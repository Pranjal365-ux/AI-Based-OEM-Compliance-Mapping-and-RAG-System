"""
Compliance Engine – Matcher Module
====================================
Evaluates compliance of one (requirement, product) pair.

Rate-limit optimisations (Groq 250K TPM / 1K RPM)
---------------------------------------------------
KEY CHANGE: batch_evaluate_requirements() groups up to BATCH_SIZE
requirements that need LLM evaluation into a single API call, returning
a JSON array of verdicts.  This cuts LLM calls by ~10x for a typical
50-requirement RFP and eliminates the most common cause of 429s.

Individual evaluate_requirement() is kept for the ranker's per-requirement
ThreadPoolExecutor interface — it now delegates to the batch path
transparently via a 1-item batch, or can be called directly for tests.

Other optimisations
-------------------
1. No-evidence short-circuit  — zero LLM calls when KB has nothing for
   this product on this requirement.
2. Numeric pre-check          — regex confirms/rejects threshold reqs
   without any LLM call (~20-25% of requirements).
3. High-score short-circuit   — cosine ≥ 0.88 with no numeric unit →
   Full Match without an LLM call.
4. generate_fast()            — routes to the fast (non-reasoning) model.
5. Tight token budget         — BATCH_MAX_TOKENS=3000 covers 10 verdicts.
6. Truncated evidence         — max EVIDENCE_PER_REQ chars per requirement.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from models.schemas import Requirement
from compliance.schemas import ComplianceStatus, EvidenceChunk, RequirementResult
from services.llm_services import llm

logger = logging.getLogger(__name__)

# ── tunables ──────────────────────────────────────────────────────────────────
EVIDENCE_PER_REQ   = 600    # chars of evidence included per requirement in a batch
BATCH_MAX_TOKENS   = 3000   # token budget for a batch LLM call (covers ~10 verdicts)
HIGH_SCORE_CUTOFF  = 0.88   # cosine threshold for no-LLM Full Match
BATCH_SIZE         = 10     # requirements per LLM call (tune down if hitting TPM limits)

# Legacy alias kept so callers that import MAX_EVIDENCE_CHARS don't break
MAX_EVIDENCE_CHARS = EVIDENCE_PER_REQ * BATCH_SIZE
VERDICT_MAX_TOKENS = BATCH_MAX_TOKENS
# ─────────────────────────────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE  = re.compile(r"```(?:json)?\s*([\s\S]*?)```")

# Unit aliases for numeric pre-check
_UNIT_ALIASES: dict[str, list[str]] = {
    "gbps":     ["gbps", "gb/s", "gbs", "gigabit", "gigabits"],
    "mbps":     ["mbps", "mb/s", "megabit", "megabits"],
    "tb":       ["tb", "terabyte", "terabytes"],
    "gb":       ["gb", "gigabyte", "gigabytes"],
    "mb":       ["mb", "megabyte", "megabytes"],
    "users":    ["users", "user", "concurrent users"],
    "sessions": ["sessions", "session", "concurrent sessions"],
    "sessions/s": ["sessions/s", "sessions per second", "session per second", "sps"],
    "connections": ["connections", "connection", "concurrent connections"],
    "cps":      ["cps", "connections per second"],
    "rps":      ["rps", "requests per second"],
    "eps":      ["eps", "events per second"],
    "m":        ["million", " m ", "m concurrent"],
}

_UNIT_INFERENCE: list[tuple[str, list[str]]] = [
    ("sessions/s", ["sessions per second", "new sessions per second"]),
    ("cps", ["connections per second", " cps"]),
    ("rps", ["requests per second", " rps"]),
    ("sessions", ["concurrent sessions", "sessions"]),
    ("connections", ["concurrent connections", "connections"]),
    ("users", ["users", "ssl vpn users"]),
    ("gb", ["hard drive", "storage", "disk"]),
]


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_requirement_no_llm(
    req:        "Requirement",
    evidence:   List[EvidenceChunk],
    vendor:     str,
    model_name: str,
) -> Optional[RequirementResult]:
    """
    Run all zero-LLM evaluation paths.

    Returns a RequirementResult if a verdict can be determined without
    an LLM call (no evidence, numeric pre-check, high-score short-circuit).
    Returns None if the LLM must be consulted.

    Used by the ranker's batch evaluation loop to separate cheap checks
    from expensive LLM calls, so the latter can be batched together.
    """
    product_evidence = [
        c for c in evidence
        if c.vendor == vendor and c.model_name == model_name
    ]

    result = RequirementResult(
        requirement_id = req.requirement_id,
        requirement    = req.requirement,
        category       = req.category,
        mandatory      = req.mandatory,
        operator       = getattr(req, "operator", None),
        value          = getattr(req, "value", None),
        unit           = getattr(req, "unit", None),
        evidence       = product_evidence[:5],
    )

    # No evidence → No Match
    if not product_evidence:
        result.status        = ComplianceStatus.NO
        result.confidence    = 0.95
        result.justification = "No relevant product specification found in the knowledge base."
        result.gap           = req.requirement
        return result

    # Numeric pre-check
    fast = _numeric_precheck(req, product_evidence)
    if fast is not None:
        result.status        = fast["status"]
        result.confidence    = fast["confidence"]
        result.justification = fast["justification"]
        result.gap           = fast.get("gap", "")
        return result

    # High-score short-circuit (no unit to verify numerically)
    top_score = product_evidence[0].score
    if top_score >= HIGH_SCORE_CUTOFF and not getattr(req, "unit", None):
        result.status        = ComplianceStatus.FULL
        result.confidence    = round(top_score, 2)
        result.justification = (
            f"Evidence strongly matches requirement (similarity={top_score:.2f}). "
            f"Supported by: {product_evidence[0].text[:120].strip()}…"
        )
        result.gap = ""
        return result

    return None   # needs LLM


def evaluate_requirement(
    req:        "Requirement",
    evidence:   List[EvidenceChunk],
    vendor:     str,
    model_name: str,
) -> RequirementResult:
    """
    Single-requirement evaluation (pre-checks + single-item LLM batch).
    Kept for backward compatibility and unit tests — the ranker now uses
    evaluate_requirement_no_llm() + batch_evaluate_requirements() instead.
    """
    result = evaluate_requirement_no_llm(req, evidence, vendor, model_name)
    if result is not None:
        return result

    # Fall through to single-item batch call
    product_evidence = [
        c for c in evidence
        if c.vendor == vendor and c.model_name == model_name
    ]
    verdicts = batch_evaluate_requirements([(req, evidence)], vendor, model_name)
    verdict = verdicts.get(req.requirement_id)
    top_score = product_evidence[0].score if product_evidence else 0.0

    result = RequirementResult(
        requirement_id = req.requirement_id,
        requirement    = req.requirement,
        category       = req.category,
        mandatory      = req.mandatory,
        operator       = getattr(req, "operator", None),
        value          = getattr(req, "value", None),
        unit           = getattr(req, "unit", None),
        evidence       = product_evidence[:5],
    )
    if verdict:
        result.status        = _parse_status(verdict.get("status", ""))
        result.confidence    = _clamp(float(verdict.get("confidence", 0.5)))
        result.justification = str(verdict.get("justification", "")).strip()
        result.gap           = str(verdict.get("gap", "")).strip()
    else:
        result.status        = _score_to_status(top_score)
        result.confidence    = round(top_score, 2)
        result.justification = "Verdict inferred from similarity score (LLM parse failed)."
        result.gap           = "" if top_score >= 0.75 else req.requirement
    return result




# ══════════════════════════════════════════════════════════════════════════════
# NUMERIC PRE-CHECK
# ══════════════════════════════════════════════════════════════════════════════

def _numeric_precheck(
    req:      Requirement,
    evidence: List[EvidenceChunk],
) -> Optional[dict]:
    """
    Try to confirm/reject a numeric threshold requirement using regex.
    Returns a verdict dict if confident, None if the LLM should decide.

    Handles K/M suffixes in both the requirement value and evidence text.
    """
    if not req.value or req.value in ("true", ""):
        return None

    try:
        threshold = _parse_number(req.value)
    except ValueError:
        return None

    unit_lower = (req.unit or _infer_unit(req.requirement) or "").lower()
    if not unit_lower:
        return None
    display_unit = req.unit or unit_lower
    unit_variants = _UNIT_ALIASES.get(unit_lower, [unit_lower])
    combined_text = "\n".join(c.text for c in evidence).lower()

    unit_pattern = "|".join(re.escape(u) for u in unit_variants)
    patterns = [
        rf"(?<![a-z])(\d[\d,\.]*\s*(?:[km]|million|thousand)?)\s*(?:{unit_pattern})",
        rf"(?:{unit_pattern})\s*(?:[:=\-]|\bis\b|\bare\b)?\s*(\d[\d,\.]*\s*(?:[km]|million|thousand)?)",
    ]
    found_values: list[float] = []
    for pattern in patterns:
        for m in re.finditer(pattern, combined_text, re.IGNORECASE):
            try:
                found_values.append(_parse_number(m.group(1)))
            except ValueError:
                pass

    if not found_values:
        return None   # No numeric evidence — let LLM decide

    best_val = max(found_values)
    op = (req.operator or ">=").lower()

    if any(x in op for x in (">=", "=>", "at least", "minimum")):
        if best_val >= threshold:
            return {
                "status":        ComplianceStatus.FULL,
                "confidence":    0.92,
                "justification": (
                    f"Evidence confirms {best_val:g} {display_unit}, "
                    f"meeting the required >= {threshold:g} {display_unit}."
                ),
                "gap": "",
            }
        elif best_val >= threshold * 0.70:
            return {
                "status":        ComplianceStatus.PARTIAL,
                "confidence":    0.80,
                "justification": (
                    f"Evidence shows {best_val:g} {display_unit}, which is below "
                    f"the required {threshold:g} {display_unit} but within 30%."
                ),
                "gap": f"Requires {threshold:g} {display_unit}; found {best_val:g} {display_unit}.",
            }
        else:
            return {
                "status":        ComplianceStatus.NO,
                "confidence":    0.88,
                "justification": (
                    f"Evidence shows only {best_val:g} {display_unit}; "
                    f"requirement is >= {threshold:g} {display_unit}."
                ),
                "gap": f"Requires {threshold:g} {display_unit}; found {best_val:g} {display_unit}.",
            }

    # Unsupported operator — let LLM handle
    return None


def _parse_number(value: str) -> float:
    raw = str(value).replace(",", "").strip().upper()
    raw = re.sub(r"\s+", "", raw)
    multiplier = 1
    if raw.endswith("K"):
        multiplier, raw = 1_000, raw[:-1]
    elif raw.endswith("M"):
        multiplier, raw = 1_000_000, raw[:-1]
    elif raw.endswith("MILLION"):
        multiplier, raw = 1_000_000, raw[:-7]
    elif raw.endswith("THOUSAND"):
        multiplier, raw = 1_000, raw[:-8]
    return float(raw) * multiplier


def _infer_unit(requirement: str) -> Optional[str]:
    lower = requirement.lower()
    for unit, signals in _UNIT_INFERENCE:
        if any(signal in lower for signal in signals):
            return unit
    return None


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_verdict_prompt(
    req:        Requirement,
    evidence:   List[EvidenceChunk],
    vendor:     str,
    model_name: str,
) -> str:
    """
    Build a compact, deterministic prompt for the fast LLM.
    Evidence is truncated to MAX_EVIDENCE_CHARS total to minimise
    generation time on a CPU-only workstation.
    """
    chars_per_chunk = max(200, MAX_EVIDENCE_CHARS // min(len(evidence), 5))
    evidence_text = ""
    for i, chunk in enumerate(evidence[:5], 1):
        snippet = chunk.text[:chars_per_chunk].replace("\n", " ").strip()
        src = Path(chunk.source_file).name if chunk.source_file else "KB"
        page = f" p.{chunk.page_start}" if chunk.page_start else ""
        evidence_text += f"[{i}] {src}{page} (sim={chunk.score:.2f}): {snippet}\n"

    threshold_line = ""
    if req.value and req.value not in ("true", "") and req.unit:
        threshold_line = f"\nThreshold: {req.operator or '>='} {req.value} {req.unit}\n"

    return (
        f"Product: {vendor} – {model_name}\n"
        f"Requirement [{req.requirement_id}] "
        f"({'MANDATORY' if req.mandatory else 'optional'}): "
        f"{req.requirement}{threshold_line}\n\n"
        f"Evidence from OEM knowledge base:\n{evidence_text.strip()}\n\n"
        f"Based ONLY on the evidence above, does this product meet the requirement?\n"
        f"Rules:\n"
        f"- Only use what is explicitly stated in the evidence.\n"
        f"- For numeric thresholds the evidence must state a value meeting the threshold.\n"
        f"- Partial Match = capability present but not at required level, or strongly implied.\n"
        f"- No Match = not mentioned or contradicted.\n\n"
        f"Reply with ONLY this JSON (no markdown, no extra text):\n"
        f'{{"status":"Full Match"|"Partial Match"|"No Match",'
        f'"confidence":<0.0-1.0>,'
        f'"justification":"<one sentence citing specific evidence>",'
        f'"gap":"<what is missing, or empty string if Full Match>"}}'
    )


# ══════════════════════════════════════════════════════════════════════════════
# LLM CALL
# ══════════════════════════════════════════════════════════════════════════════

def _call_llm(prompt: str) -> Optional[dict]:
    """Call the fast (non-reasoning) model for a yes/no compliance verdict."""
    try:
        raw = llm.generate_fast(prompt, max_tokens=BATCH_MAX_TOKENS, temperature=0)
        return _parse_verdict(raw)
    except Exception as exc:
        logger.warning(f"LLM verdict call failed: {exc}")
        return None


def batch_evaluate_requirements(
    reqs_and_evidence: List[Tuple["Requirement", List[EvidenceChunk]]],
    vendor: str,
    model_name: str,
) -> Dict[str, dict]:
    """
    Evaluate multiple requirements in a SINGLE LLM call.

    This is the primary tool for avoiding rate limits. Instead of one API
    call per requirement (the old default), this groups up to BATCH_SIZE
    requirements that all need LLM evaluation and asks for all verdicts at
    once in a JSON array response.

    Parameters
    ----------
    reqs_and_evidence : list of (Requirement, evidence_chunks) pairs —
        ONLY requirements that actually need LLM evaluation (i.e. they
        passed no-evidence and numeric pre-checks without a verdict).
    vendor, model_name : product being evaluated.

    Returns
    -------
    dict mapping requirement_id → verdict dict
        {status, confidence, justification, gap}
    """
    if not reqs_and_evidence:
        return {}

    items = []
    for req, evidence in reqs_and_evidence:
        product_ev = [c for c in evidence if c.vendor == vendor and c.model_name == model_name]
        chars = EVIDENCE_PER_REQ // max(1, min(len(product_ev), 4))
        ev_text = ""
        for i, chunk in enumerate(product_ev[:4], 1):
            snippet = chunk.text[:chars].replace("\n", " ").strip()
            src = Path(chunk.source_file).name if chunk.source_file else "KB"
            page = f" p.{chunk.page_start}" if chunk.page_start else ""
            ev_text += f"[{i}] {src}{page} (sim={chunk.score:.2f}): {snippet}\n"

        threshold = ""
        if req.value and req.value not in ("true", "") and req.unit:
            threshold = f" | threshold: {req.operator or '>='} {req.value} {req.unit}"

        items.append({
            "id": req.requirement_id,
            "mandatory": req.mandatory,
            "requirement": req.requirement + threshold,
            "evidence": ev_text.strip(),
        })

    prompt = (
        f"Product: {vendor} – {model_name}\n"
        f"Task: For each requirement below, return a compliance verdict based ONLY on the provided evidence.\n\n"
        f"Rules:\n"
        f"- Only use what is explicitly stated in the evidence.\n"
        f"- For numeric thresholds, evidence must state a value meeting the threshold.\n"
        f"- Partial Match = capability present but not at required level, or strongly implied.\n"
        f"- No Match = not mentioned or contradicted.\n\n"
        f"Requirements:\n"
    )
    for item in items:
        prompt += (
            f"\n--- REQ {item['id']} ({'MANDATORY' if item['mandatory'] else 'optional'}) ---\n"
            f"Requirement: {item['requirement']}\n"
            f"Evidence:\n{item['evidence']}\n"
        )

    prompt += (
        "\n\nReply with ONLY a JSON array, one object per requirement, in the same order:\n"
        '[{"id":"<req_id>","status":"Full Match"|"Partial Match"|"No Match",'
        '"confidence":<0.0-1.0>,"justification":"<one sentence>","gap":"<or empty>"}]\n'
        "No markdown. No extra text. JSON array only."
    )

    try:
        raw = llm.generate_fast(prompt, max_tokens=BATCH_MAX_TOKENS, temperature=0)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw).strip()
        raw = re.sub(r"\s*```$", "", raw).strip()
        # Find first [
        bracket = raw.find("[")
        if bracket != -1:
            raw = raw[bracket:]
        data = json.loads(raw)
        if not isinstance(data, list):
            return {}
        return {
            item["id"]: item
            for item in data
            if isinstance(item, dict) and "id" in item and "status" in item
        }
    except Exception as exc:
        logger.warning(f"[matcher] Batch LLM call failed: {exc}")
        return {}




def _parse_verdict(raw: str) -> Optional[dict]:
    """Strip thinking tags, extract JSON from raw LLM output."""
    raw = _THINK_RE.sub("", raw).strip()

    # Try fenced code block first
    fence = _FENCE_RE.search(raw)
    if fence:
        raw = fence.group(1).strip()

    # Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Find first {...} block
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    logger.debug(f"Could not parse LLM verdict: {raw[:200]!r}")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_status(s: str) -> ComplianceStatus:
    s = s.lower().strip()
    if "full" in s:    return ComplianceStatus.FULL
    if "partial" in s: return ComplianceStatus.PARTIAL
    return ComplianceStatus.NO


def _score_to_status(score: float) -> ComplianceStatus:
    if score >= 0.80: return ComplianceStatus.FULL
    if score >= 0.55: return ComplianceStatus.PARTIAL
    return ComplianceStatus.NO


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))

"""
Compliance Engine – Matcher Module
====================================
Evaluates compliance of one (requirement, product) pair.

Speed optimisations (CPU-only workstation, no GPU)
---------------------------------------------------
1. No-evidence short-circuit  — zero LLM calls when KB has nothing for
   this product on this requirement.
2. Numeric pre-check          — regex confirms/rejects threshold reqs
   without any LLM call (~20-25% of requirements).
3. High-score short-circuit   — cosine ≥ 0.88 with no numeric unit →
   Full Match without an LLM call.
4. generate_fast()            — routes to the fast (non-reasoning) model,
   ~15 sec/call on CPU vs ~60-90 sec for a reasoning model.
5. Tight token budget         — VERDICT_MAX_TOKENS=400 keeps generation fast.
6. Truncated evidence         — max 1 200 chars total evidence per call.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List, Optional

from models.schemas import Requirement
from compliance.schemas import ComplianceStatus, EvidenceChunk, RequirementResult
from services.llm_services import llm

logger = logging.getLogger(__name__)

# ── tunables ──────────────────────────────────────────────────────────────────
MAX_EVIDENCE_CHARS = 1200   # total chars of evidence sent per LLM call
VERDICT_MAX_TOKENS = 400    # JSON verdict is always <100 tokens; 400 = margin
HIGH_SCORE_CUTOFF  = 0.88   # cosine threshold for no-LLM Full Match
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

def evaluate_requirement(
    req:        Requirement,
    evidence:   List[EvidenceChunk],
    vendor:     str,
    model_name: str,
) -> RequirementResult:
    """
    Evaluate compliance of one requirement against OEM evidence chunks
    for a specific product.  Returns a RequirementResult with verdict.

    Evaluation order
    ----------------
    1. Filter evidence to this (vendor, model_name) pair.
    2. If no evidence → No Match (zero LLM calls).
    3. If requirement has a numeric threshold → regex pre-check.
    4. If top cosine score ≥ HIGH_SCORE_CUTOFF and no unit → Full Match.
    5. Otherwise → LLM verdict via fast model.
    """
    # Step 1 — filter to this product
    product_evidence = [
        c for c in evidence
        if c.vendor == vendor and c.model_name == model_name
    ]

    result = RequirementResult(
        requirement_id = req.requirement_id,
        requirement    = req.requirement,
        category       = req.category,
        mandatory      = req.mandatory,
        operator       = req.operator,
        value          = req.value,
        unit           = req.unit,
        evidence       = product_evidence[:5],   # store top-5 for report
    )

    # Step 2 — no-evidence short-circuit
    if not product_evidence:
        result.status        = ComplianceStatus.NO
        result.confidence    = 0.95
        result.justification = "No relevant product specification found in the knowledge base."
        result.gap           = req.requirement
        return result

    # Step 3 — numeric pre-check (avoids LLM for ~20-25% of requirements)
    fast = _numeric_precheck(req, product_evidence)
    if fast is not None:
        result.status        = fast["status"]
        result.confidence    = fast["confidence"]
        result.justification = fast["justification"]
        result.gap           = fast.get("gap", "")
        return result

    # Step 4 — high-score short-circuit (no unit to verify numerically)
    top_score = product_evidence[0].score
    if top_score >= HIGH_SCORE_CUTOFF and not req.unit:
        result.status        = ComplianceStatus.FULL
        result.confidence    = round(top_score, 2)
        result.justification = (
            f"Evidence strongly matches requirement (similarity={top_score:.2f}). "
            f"Supported by: {product_evidence[0].text[:120].strip()}…"
        )
        result.gap = ""
        return result

    # Step 5 — LLM verdict via fast (non-reasoning) model
    prompt  = _build_verdict_prompt(req, product_evidence, vendor, model_name)
    verdict = _call_llm(prompt)

    if verdict:
        result.status        = _parse_status(verdict.get("status", ""))
        result.confidence    = _clamp(float(verdict.get("confidence", 0.5)))
        result.justification = str(verdict.get("justification", "")).strip()
        result.gap           = str(verdict.get("gap", "")).strip()
    else:
        # LLM parse failed — fall back to cosine heuristic
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
        raw = llm.generate_fast(prompt, max_tokens=VERDICT_MAX_TOKENS, temperature=0)
        return _parse_verdict(raw)
    except Exception as exc:
        logger.warning(f"LLM verdict call failed: {exc}")
        return None


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

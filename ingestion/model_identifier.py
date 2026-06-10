"""
OEM Datasheet Ingestion Pipeline - Model Identification
Identifies distinct product models within a datasheet.

Architecture
------------
1. Table-based extraction - parses comparison / ordering tables   (fast)
2. Regex candidate sweep  - pattern-match model numbers in full text (fast)
3. LLM filter (if enabled)- given the candidate list, eliminate non-models
                            (cheap: sends a small JSON list, not raw text)
4. Single-model fallback  - whole doc = one product entry

The LLM is now used only as a *filter / validator* over candidates already
found by parsing.  This is much faster and cheaper than asking the LLM to
scan 6 k chars of raw text:
  - Old: LLM reads full text → produces model list from scratch
  - New: Parsing finds candidates → LLM receives only the candidate names
         and removes false positives / non-product strings.

Section enrichment  (multi-model)
----------------------------------
Per-model sections  → only added to the model(s) explicitly mentioned
Family sections     → stored ONCE on models[0]; chunker emits them once
                      so the vector DB holds N model chunks + 1 family chunk
                      instead of N copies of the same text.

This is the single most important fix for preventing chunk explosion:
previously every model received a full copy of the shared sections.
"""
from __future__ import annotations

import json
import re
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from loguru import logger

from config.settings import ModelIdentificationConfig, PipelineConfig
from models.schemas import ExtractedTable, ModelSpec
from ingestion.classifier import detect_category


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Section names that belong to the whole product family, not a single SKU.
# These are stored ONCE on models[0]; the chunker emits them once.
_FAMILY_SECTION_KEYWORDS: FrozenSet[str] = frozenset({
    "overview", "introduction", "description",
    "features", "key features", "product features", "highlights",
    "certifications", "compliance", "regulatory", "standards",
    "ordering", "ordering information", "part number", "sku",
    "environmental", "operating conditions",
    "warranty", "support", "services",
    "use cases", "solution overview",
})


def _is_family_section(name: str) -> bool:
    key = name.lower().strip()
    return any(kw in key for kw in _FAMILY_SECTION_KEYWORDS)


# ---------------------------------------------------------------------------
# Pattern compilation cache
# ---------------------------------------------------------------------------

_PATTERN_CACHE: Dict[int, List[re.Pattern]] = {}


def _compile_model_patterns(cfg: ModelIdentificationConfig) -> List[re.Pattern]:
    key = id(cfg)
    if key not in _PATTERN_CACHE:
        _PATTERN_CACHE[key] = [
            re.compile(p, re.IGNORECASE) for p in cfg.model_number_patterns
        ]
    return _PATTERN_CACHE[key]


def _build_combined_pattern(model_names: List[str]) -> re.Pattern:
    """Build one OR-pattern matching any of the given model names (longest first)."""
    sorted_names = sorted(model_names, key=len, reverse=True)
    inner = "|".join(re.escape(n) for n in sorted_names)
    return re.compile(
        r"(?<![A-Za-z0-9\-_])(" + inner + r")(?![A-Za-z0-9\-_])",
        re.IGNORECASE,
    )


# ---------------------------------------------------------------------------
# Section splitter
# ---------------------------------------------------------------------------

def split_into_sections(pages: List[dict]) -> Dict[str, List[str]]:
    """
    Walk all page texts and segment them into named sections.
    Returns {section_name: [text_lines...]}
    """
    sections: Dict[str, List[str]] = {"_preamble": []}
    current = "_preamble"

    for page in pages:
        text = page.get("cleaned_text", "")
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if _is_section_heading(stripped):
                current = stripped.upper()
                if current not in sections:
                    sections[current] = []
            else:
                sections.setdefault(current, []).append(stripped)

    return sections


def _is_section_heading(line: str) -> bool:
    line = line.strip()
    if not (3 <= len(line) <= 70):
        return False
    if line.startswith(("•", "-", "*", "o ", "+ ")):
        return False
    if line[-1] in {":", ",", ".", ";", "?", "!"}:
        return False
    words = line.split()
    if words and words[-1].lower() in {
        "with", "and", "or", "for", "in", "on", "at", "by", "to", "of"
    }:
        return False

    _UNITS = {"gbps", "mbps", "mpps", "tb", "gb", "mb", "w", "v", "a",
               "hz", "db", "btu/h", "million", "billion", "sessions", "users"}
    if words and words[-1].lower().rstrip(".,;:") in _UNITS:
        return False

    if re.match(r'^(\d+\.\d+(\.\d+)*|\d+[\.\)])\s+[A-Z]', line):
        return True

    line_lower = line.lower()

    _MULTI = {
        "technical specifications", "hardware specifications",
        "system specifications", "product specifications",
        "ordering information", "ordering info", "part numbers",
        "operating conditions", "environmental specifications",
        "key features", "features & benefits", "product features",
        "product overview", "system overview", "use cases",
    }
    if any(kw in line_lower for kw in _MULTI):
        return True

    _BRIEF = {
        "overview", "features", "specifications", "specs", "ordering",
        "compliance", "certifications", "regulatory", "standards",
        "interfaces", "connectivity", "dimensions", "physical",
        "power", "electrical", "environmental", "support", "warranty",
        "performance",
    }
    for kw in _BRIEF:
        if (line_lower == kw
                or line_lower.startswith(kw + " ")
                or line_lower.startswith(kw + ":")):
            return True

    # Strict ALL-CAPS heading
    if line.isupper() and 2 <= len(words) <= 5:
        if not re.search(r'\d', line) and not any(
            u in line_lower for u in ["gbps", "mbps", "tb", "gb", "v", "w", "hz"]
        ):
            return True

    return False


# ---------------------------------------------------------------------------
# Model number extraction (regex fallback)
# ---------------------------------------------------------------------------

def extract_candidate_model_numbers(
    full_text: str,
    cfg: ModelIdentificationConfig,
) -> Dict[str, int]:
    """Return {model_number: occurrence_count} sorted by frequency."""
    patterns = _compile_model_patterns(cfg)
    counts: Dict[str, int] = {}
    for pattern in patterns:
        for match in pattern.finditer(full_text):
            token = match.group(0).strip().upper()
            if _is_false_positive_model(token):
                continue
            counts[token] = counts.get(token, 0) + 1

    return dict(
        sorted(
            {m: c for m, c in counts.items() if c >= cfg.min_model_occurrences}.items(),
            key=lambda x: -x[1],
        )
    )


def _is_false_positive_model(token: str) -> bool:
    _FP = {
        "IEEE", "HTTP", "HTTPS", "SMTP", "SNMP", "SSH", "SSL", "TLS",
        "VLAN", "OSPF", "BGP", "LACP", "IPV4", "IPV6", "NAT", "VPN",
        "PDF", "USB", "PCB", "LED", "LCD", "CPU", "RAM", "SSD", "HDD",
        "MTBF", "MTTR", "RMA", "EOL", "EOS", "RFP", "SKU", "UPS",
        "AC", "DC", "EN", "ISO", "CE", "FCC", "UL", "CSA", "IP65",
        "ROHS", "WEEE", "TAA", "USA", "EU", "UK",
    }
    if token in _FP:
        return True
    if re.fullmatch(r"SHA[-_]?\d+", token):
        return True
    if re.fullmatch(r"NAT\d+", token):
        return True
    if len(token) <= 2:
        return True
    return False


# Matches the suffix that differentiates a specific SKU from a series root,
# e.g. "20" in PA-3200→PA-3220, or "F" in FG-100→FG-100F.
# Hardware-variant suffixes like "-DC", "-AC", "-POE" are excluded so we
# don't mistakenly treat "FG-7081F" as a prefix of "FG-7081F-DC".
_DIGIT_ONLY_SUFFIX_RE = re.compile(r"\d+$")


def _prune_family_prefixes(candidates: List[str]) -> List[str]:
    """
    Drop a candidate only when it looks like a series root, i.e.:
      - It is a string prefix of at least one other candidate, AND
      - Every longer candidate that starts with it extends with digits only
        (like PA-3200 → PA-3220/3250/3260), not with a hardware suffix
        (like FG-7081F → FG-7081F-DC).

    Example kept:   ["FG-7081F", "FG-7081F-DC"]  → both kept (-DC is hardware)
    Example pruned: ["PA-3200",  "PA-3220", "PA-3250"] → PA-3200 dropped
    """
    upper = [c.upper() for c in candidates]
    pruned = []
    for i, candidate in enumerate(candidates):
        cu = upper[i]
        longer = [upper[j] for j in range(len(upper)) if j != i and upper[j].startswith(cu) and upper[j] != cu]
        if not longer:
            pruned.append(candidate)
            continue
        # Only drop if every extension beyond the shared prefix is purely numeric
        all_digit_extensions = all(
            _DIGIT_ONLY_SUFFIX_RE.search(lon[len(cu):]) and
            not lon[len(cu):].startswith("-")
            for lon in longer
        )
        if all_digit_extensions:
            logger.debug(
                f"[model_id] Dropping '{candidate}' — series-root prefix of "
                + ", ".join(f"'{c}'" for c in longer)
            )
        else:
            pruned.append(candidate)
    return pruned


def _prune_series_names(candidates: List[str], full_text: str) -> List[str]:
    """
    Drop any candidate whose occurrences in the text are overwhelmingly as
    '<candidate> Series' (i.e. it names a product family, not a specific SKU).

    Threshold: if > 60% of the candidate's occurrences are immediately
    followed by the word 'Series', treat it as a family name and drop it.
    """
    pruned = []
    for candidate in candidates:
        escaped = re.escape(candidate)
        all_hits = re.findall(
            r"(?<![A-Za-z0-9\-_])" + escaped + r"(?![A-Za-z0-9\-_])",
            full_text,
            re.IGNORECASE,
        )
        total = len(all_hits)
        if total == 0:
            pruned.append(candidate)
            continue

        series_hits = re.findall(
            r"(?<![A-Za-z0-9\-_])" + escaped + r"\s+Series\b",
            full_text,
            re.IGNORECASE,
        )
        series_ratio = len(series_hits) / total

        if series_ratio > 0.6:
            logger.debug(
                f"[model_id] Dropping '{candidate}' — {series_ratio:.0%} of "
                f"occurrences are '<name> Series' (family name, not a SKU)"
            )
        else:
            pruned.append(candidate)
    return pruned



# ---------------------------------------------------------------------------
# Table-based model detection
# ---------------------------------------------------------------------------

def extract_models_from_tables(
    page_tables: List[dict],
    cfg: ModelIdentificationConfig,
) -> List[Dict]:
    """
    Parse comparison / ordering tables and return per-model entries.
    Returns list of {"model_name": str, "spec_row": dict}
    """
    model_entries: List[Dict] = []

    for tbl in page_tables:
        headers = [str(h).lower() for h in tbl.get("headers", [])]
        raw_headers = tbl.get("headers", [])
        rows = tbl.get("rows", [])

        if not headers:
            continue

        # Comparison table: model names IN the headers
        header_models = _extract_model_names_from_cells(raw_headers, cfg)
        if header_models:
            for mn in header_models:
                model_entries.append({"model_name": mn, "spec_row": {}})
            continue

        if not rows:
            continue

        # Comparison table: model names in the first row
        first_row_models = _extract_model_names_from_cells(rows[0], cfg)
        if len(first_row_models) >= 2:
            for mn in first_row_models:
                model_entries.append({"model_name": mn, "spec_row": {}})
            continue

        # Ordering / spec table: find the column whose header hints at model ID
        model_col = None
        for i, h in enumerate(headers):
            if any(kw in h for kw in cfg.model_header_keywords):
                model_col = i
                break
        if model_col is None and _rows_look_like_specs(rows, headers):
            model_col = 0
        if model_col is None:
            continue

        for row in rows:
            if not row or model_col >= len(row):
                continue
            mn = row[model_col].strip()
            if not _looks_like_model_number(mn, cfg):
                continue
            model_entries.append({
                "model_name": _strip_annotation_markers(mn.upper()),
                "spec_row": {
                    headers[i]: row[i]
                    for i in range(min(len(headers), len(row)))
                    if row[i].strip()
                },
            })

    return model_entries


def _extract_model_names_from_cells(
    cells: List[str], cfg: ModelIdentificationConfig
) -> List[str]:
    patterns = _compile_model_patterns(cfg)
    seen: Set[str] = set()
    result: List[str] = []
    for cell in cells:
        for part in re.split(r"[/,\n]+", str(cell or "")):
            candidate = _strip_annotation_markers(part.strip().upper())
            if not candidate or candidate in seen:
                continue
            if not any(p.fullmatch(candidate) for p in patterns):
                continue
            if _is_false_positive_model(candidate):
                continue
            seen.add(candidate)
            result.append(candidate)
    return result


def _strip_annotation_markers(value: str) -> str:
    return re.sub(r"[*†‡§#|]+$", "", value).strip()


def _looks_like_model_number(value: str, cfg: ModelIdentificationConfig) -> bool:
    candidate = _strip_annotation_markers(value.strip().upper())
    if not candidate or len(candidate) < 3 or len(candidate.split()) > 2:
        return False
    if _is_false_positive_model(candidate):
        return False
    return any(p.fullmatch(candidate) for p in _compile_model_patterns(cfg))


def _rows_look_like_specs(rows: List[List[str]], headers: List[str]) -> bool:
    if not rows:
        return False
    numeric = sum(
        1 for row in rows[:5] for cell in row if re.search(r'\d', cell)
    )
    return numeric >= len(rows[:5])


# ---------------------------------------------------------------------------
# LLM-based candidate filtering
# ---------------------------------------------------------------------------

# Provisioning-only suffixes that do NOT represent distinct hardware SKUs.
# ZTP = Zero Touch Provisioning, BDL = bundle, LENC = low-encryption export.
# DC *is* a hardware difference (DC power supply) so it is intentionally absent.
_SOFT_SUFFIX_RE = re.compile(
    r"[-_](ZTP|BDL|LENC|NFR|GOV|TAA|EDU|EVAL|DEMO|LAB|DEV|POC)$",
    re.IGNORECASE,
)


def _prune_soft_variant_suffixes(candidates: List[str]) -> List[str]:
    """
    Drop provisioning / bundle / export-control suffix variants when the
    base model is already present in the candidate list.

    Example: ["FG-7081F", "FG-7081F-ZTP", "FG-7081F-BDL"]
             → ["FG-7081F"]   (ZTP and BDL are not distinct hardware)

    DC variants are kept because they have a different power supply.
    """
    upper_set = {c.upper() for c in candidates}
    pruned = []
    for candidate in candidates:
        m = _SOFT_SUFFIX_RE.search(candidate)
        if m:
            base = candidate[: m.start()].upper()
            if base in upper_set:
                logger.debug(
                    f"[model_id] Dropping '{candidate}' — soft-suffix variant "
                    f"of '{base}' which is already in the candidate list"
                )
                continue
        pruned.append(candidate)
    return pruned


def _parse_llm_json(raw: str, candidate_set: set) -> List[Dict]:
    """
    Clean and parse the LLM's JSON response.
    Handles: empty string, markdown fences, Qwen <think> blocks,
    bare JSON array, and single-object responses.
    Returns a filtered list of valid candidate dicts, or [] on any failure.
    """
    # Strip Qwen thinking blocks
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    raw = re.sub(r"^```(?:json)?\s*", "", raw).strip()
    raw = re.sub(r"\s*```$", "", raw).strip()

    if not raw:
        return []

    # Find the first '[' or '{' — discard any preamble the model snuck in
    first_bracket = next(
        (i for i, ch in enumerate(raw) if ch in ("{", "[")), None
    )
    if first_bracket is None:
        return []
    raw = raw[first_bracket:]

    data = json.loads(raw)          # raises JSONDecodeError on bad JSON
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []

    return [
        d for d in data
        if isinstance(d, dict) and d.get("model_name", "").upper() in candidate_set
    ]


def filter_candidates_with_llm(
    candidates: List[str],
    vendor: str,
    cfg: PipelineConfig,
    context_snippet: str = "",
) -> List[Dict]:
    """
    Given a list of candidate model strings found by parsing, ask the LLM to
    remove false positives and return the confirmed real product models.

    Robustness improvements vs. the original:
    - Strips preamble text before the JSON bracket (model leak / thinking noise).
    - Retries once with a simpler prompt if the first attempt returns empty JSON.
    - Returns None (not []) to signal total failure so the caller can decide
      whether to fall back to the unfiltered candidate list.

    Returns:
        List[Dict]  — confirmed models (may be empty list if LLM found none).
        None        — LLM call failed entirely; caller should skip filtering.
    """
    if not cfg.use_llm_for_model_id:
        return None
    if not candidates:
        return None

    try:
        from services.llm_services import llm
    except Exception as e:
        logger.warning(f"[model_id] LLM init failed: {e}")
        return None

    candidate_set = {c.upper() for c in candidates}
    candidate_json = json.dumps(candidates)   # compact — fewer tokens

    context_block = ""
    if context_snippet:
        context_block = (
            f"\nCONTEXT (first 800 chars of document):\n"
            f"{context_snippet[:800]}\n"
        )

    def _make_prompt(cands_json: str) -> str:
        return (
            f'You are an OEM datasheet extraction engine for vendor "{vendor}".\n'
            f"Return ONLY a JSON array. No preamble, no markdown, no code fences.\n"
            f"{context_block}"
            f"CANDIDATES: {cands_json}\n\n"
            f"Keep only genuine product model/SKU strings. "
            f"Output schema: "
            f'[{{"model_name":"<exact string>","product_family":"<family or null>"}}]'
            f"\nIf none qualify, return []. JSON ONLY."
        )

    for attempt in range(2):
        try:
            raw = llm.generate(_make_prompt(candidate_json), temperature=0, max_tokens=2000)
            data = _parse_llm_json(raw, candidate_set)

            if data is not None:        # parsed successfully (even if empty list)
                logger.info(
                    f"[model_id] LLM filtered {len(candidates)} candidates → "
                    f"{len(data)} confirmed model(s)"
                    + (f" (attempt {attempt + 1})" if attempt else "")
                )
                return data

            if attempt == 0:
                logger.debug("[model_id] LLM attempt 1 returned unparseable output — retrying")

        except json.JSONDecodeError as exc:
            if attempt == 0:
                logger.debug(f"[model_id] LLM attempt 1 JSON error: {exc} — retrying")
            else:
                logger.warning(f"[model_id] LLM returned non-JSON on both attempts: {exc}")
                return None
        except Exception as exc:
            logger.warning(f"[model_id] LLM call failed: {exc}")
            return None

    logger.warning("[model_id] LLM filtering failed after 2 attempts — skipping")
    return None


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def identify_models(
    pages: List[dict],
    vendor: str,
    filename: str,
    cfg: PipelineConfig,
) -> List[ModelSpec]:
    """
    Master entry point.  Returns a list of ModelSpec, one per distinct SKU.

    Strategy order:
      1. Table-based extraction  – fast structural parsing
      2. Regex candidate sweep   – pattern-match across full text
      3. LLM filter (if enabled) – validate/prune the combined candidate list;
                                   prompt contains only the candidate names,
                                   NOT the raw document text
      4. Single-model fallback   – whole doc = one product entry

    The LLM is invoked AFTER parsing so it only needs to evaluate a short
    list of candidate strings rather than thousands of tokens of raw text.
    This keeps LLM latency and cost low while still catching edge cases that
    pure regex might mis-classify.
    """
    full_text = "\n".join(p.get("cleaned_text", "") for p in pages)
    all_tables = [t for p in pages for t in p.get("tables", [])]
    sections = split_into_sections(pages)

    category, confidence = detect_category(filename=filename, full_text=full_text)
    models: List[ModelSpec] = []

    # ── Stage 1: Table-based extraction ───────────────────────────────────
    table_models = extract_models_from_tables(all_tables, cfg.model_id)
    table_specs: Dict[str, dict] = {}  # model_name → spec_row
    table_names: List[str] = []
    seen_table: Set[str] = set()
    for m in table_models:
        mn = _strip_annotation_markers(m["model_name"].strip())
        if mn and mn not in seen_table:
            seen_table.add(mn)
            table_names.append(mn)
            table_specs[mn] = m.get("spec_row", {})

    logger.debug(f"[model_id] Table extraction: {len(table_names)} candidate(s)")

    # ── Stage 2: Regex candidate sweep ────────────────────────────────────
    regex_candidates = extract_candidate_model_numbers(full_text, cfg.model_id)
    # Merge: table names take priority; add regex finds not already covered
    all_candidate_names: List[str] = list(table_names)
    seen_all: Set[str] = set(table_names)
    for mn in list(regex_candidates.keys())[:20]:
        mn = _strip_annotation_markers(mn.strip())
        if mn and mn not in seen_all:
            seen_all.add(mn)
            all_candidate_names.append(mn)

    logger.debug(
        f"[model_id] Combined candidates after regex: {len(all_candidate_names)}"
    )

    # ── Stage 2b: Structural pruning (no LLM needed) ──────────────────────
    # Three cheap passes before any LLM call:
    #   1. Drop soft-suffix variants (ZTP/BDL/…) when base is present.
    #   2. Drop tokens that are string-prefixes of longer candidates.
    #   3. Drop tokens used overwhelmingly as "<name> Series" in the text.
    all_candidate_names = _prune_soft_variant_suffixes(all_candidate_names)
    all_candidate_names = _prune_family_prefixes(all_candidate_names)
    all_candidate_names = _prune_series_names(all_candidate_names, full_text)
    logger.debug(
        f"[model_id] After structural pruning: {len(all_candidate_names)} candidate(s)"
    )

    # ── Stage 3: LLM filter (optional) ────────────────────────────────────
    # Pass only the pruned candidate list to the LLM.
    # filter_candidates_with_llm returns:
    #   List[Dict]  → LLM responded; use the filtered list (may be empty).
    #   None        → LLM failed entirely; keep all structurally-pruned candidates.
    llm_confirmed: Optional[Dict[str, str]] = None  # model_name → product_family
    if cfg.use_llm_for_model_id and all_candidate_names:
        context_snippet = full_text[:800]
        llm_data = filter_candidates_with_llm(
            all_candidate_names, vendor, cfg, context_snippet
        )
        if llm_data is None:
            # LLM failed — proceed with structurally-pruned candidates as-is.
            logger.warning(
                "[model_id] LLM filter unavailable — using structural pruning results"
            )
        else:
            # LLM responded (even an empty list is authoritative).
            llm_confirmed = {
                d["model_name"].upper(): d.get("product_family")
                for d in llm_data
                if d.get("model_name")
            }
            all_candidate_names = [
                n for n in all_candidate_names
                if n.upper() in llm_confirmed
            ]
            logger.info(
                f"[model_id] After LLM filter: {len(all_candidate_names)} model(s)"
            )

    # ── Build ModelSpec list from confirmed candidates ─────────────────────
    if all_candidate_names:
        for mn in all_candidate_names:
            # Determine extraction confidence & method
            if mn in seen_table:
                conf_score = 0.85 if llm_confirmed is not None else 0.75
                method = "table+llm_filter" if llm_confirmed is not None else "table"
            else:
                conf_score = 0.65 if llm_confirmed is not None else 0.5
                method = "regex+llm_filter" if llm_confirmed is not None else "regex"

            spec_text = _spec_row_to_text(table_specs.get(mn, {}))
            family = (llm_confirmed or {}).get(mn.upper())

            models.append(ModelSpec(
                model_id=_make_model_id(vendor, mn, len(models)),
                model_name=mn,
                vendor=vendor,
                product_family=family,
                product_category=category,
                category_confidence=confidence,
                spec_sections={"Specifications": spec_text} if spec_text else {},
                source_pages=list(range(1, len(pages) + 1)),
                extraction_confidence=conf_score,
                identified_by=method,
            ))

        _enrich_models(models, sections, full_text, pages)
        return models

    # ── Stage 4: Single-model fallback ────────────────────────────────────
    logger.info("[model_id] No distinct models — treating as single-model doc")
    model_name = _guess_model_name(pages, vendor)
    models.append(ModelSpec(
        model_id=_make_model_id(vendor, model_name, 0),
        model_name=model_name,
        vendor=vendor,
        description=_extract_description(sections),
        spec_sections=_sections_to_spec_dict(sections),
        source_pages=list(range(1, len(pages) + 1)),
        extraction_confidence=0.4,
        product_category=category,
        category_confidence=confidence,
        identified_by="fallback_single",
    ))
    return models


# ---------------------------------------------------------------------------
# Enrichment  (key fix for chunk explosion)
# ---------------------------------------------------------------------------

def _enrich_models(
    models: List[ModelSpec],
    sections: Dict[str, List[str]],
    full_text: str,
    pages: List[dict],
) -> None:
    """
    Distribute document content to models.

    Single model  → gets everything.
    Multi-model   →
      - Family sections (overview, features, certs …) go to models[0] ONLY.
        The chunker emits these once tagged with the family / first model.
      - Per-model context (paragraphs that explicitly mention the SKU) goes
        to each model individually.
      - Spec sections that mention a model go only to that model.
      - Sections that mention NO known model go to all models (shared specs).
    """
    if len(models) == 1:
        models[0].spec_sections = _sections_to_spec_dict(sections)
        models[0].description = _extract_description(sections)
        _assign_model_page_ranges(models, pages)
        return

    all_names = [m.model_name for m in models]
    combined = _build_combined_pattern(all_names)
    upper_to_name: Dict[str, str] = {n.upper(): n for n in all_names}
    name_to_model: Dict[str, ModelSpec] = {m.model_name: m for m in models}

    shared_desc = _extract_description(sections)

    # Separate family vs spec sections
    family_secs: Dict[str, str] = {}
    spec_secs: Dict[str, str] = {}

    for sec_name, lines in sections.items():
        if sec_name == "_preamble":
            continue
        text = "\n".join(lines).strip()
        if not text:
            continue
        if _is_family_section(sec_name):
            family_secs[sec_name.title()] = text
        else:
            spec_secs[sec_name.title()] = text

    # Family sections → models[0] only (chunker emits once)
    for sec_name, sec_text in family_secs.items():
        if sec_name not in models[0].spec_sections:
            models[0].spec_sections[sec_name] = sec_text

    # Shared description → all models (short, won't cause explosion)
    for model in models:
        if not model.description:
            model.description = shared_desc

    # Spec sections: scan once, distribute by model mention
    for sec_name, sec_text in spec_secs.items():
        found_upper = {m.upper() for m in combined.findall(sec_text)}
        mentioned = {upper_to_name[u] for u in found_upper if u in upper_to_name}

        if not mentioned:
            # Untagged spec section → assign to all models (e.g. shared hardware)
            for model in models:
                if sec_name not in model.spec_sections:
                    model.spec_sections[sec_name] = sec_text
        else:
            for mn in mentioned:
                model = name_to_model.get(mn)
                if model and sec_name not in model.spec_sections:
                    model.spec_sections[sec_name] = sec_text

    # Per-model context windows (paragraphs that mention the SKU by name)
    paragraphs = re.split(r"\n{2,}", full_text)
    for para in paragraphs:
        para = para.strip()
        if len(para) < 30:
            continue
        found_upper = {m.upper() for m in combined.findall(para)}
        for u in found_upper:
            mn = upper_to_name.get(u)
            if not mn:
                continue
            model = name_to_model.get(mn)
            if not model:
                continue
            existing = model.spec_sections.get("Model Context", "")
            if len(existing) < 4000:
                model.spec_sections["Model Context"] = (
                    (existing + "\n\n" + para).strip()
                    if existing else para
                )

    _assign_model_page_ranges(models, pages)


# ---------------------------------------------------------------------------
# Page range assignment + sub-module detection
# ---------------------------------------------------------------------------

_SUBMODULE_PATTERN = re.compile(
    r'\b(F[A-Z]{2,3}-\d{4}[A-Z0-9\-]*)\b', re.IGNORECASE
)
_SUBMODULE_PREFIXES = ("FPM-", "FIM-", "SPM-", "FMC-", "FPC-", "FAP-")


def _is_submodule_name(name: str) -> bool:
    return any(name.upper().startswith(pfx) for pfx in _SUBMODULE_PREFIXES)


def _assign_model_page_ranges(
    models: List[ModelSpec],
    pages: List[dict],
) -> None:
    """Narrow each model's source_pages to pages that actually mention it."""
    if len(models) <= 1:
        return

    all_names = [m.model_name for m in models]
    combined = _build_combined_pattern(all_names)
    upper_map: Dict[str, str] = {n.upper(): n for n in all_names}

    page_hits: List[Set[str]] = []
    for page in pages:
        text = page.get("cleaned_text", "")
        found = {m.upper() for m in combined.findall(text)}
        page_hits.append({upper_map[u] for u in found if u in upper_map})

    for model in models:
        hits = [idx + 1 for idx, s in enumerate(page_hits) if model.model_name in s]
        if hits:
            model.source_pages = list(range(min(hits), max(hits) + 1))
        else:
            logger.debug(f"[model_id] '{model.model_name}': no page hits, keeping all")

    # Sub-module detection (FPM/FIM/etc.)
    existing_upper = {m.model_name.upper() for m in models}
    vendor = models[0].vendor if models else "Unknown"
    category = models[0].product_category if models else "Unknown"
    conf = models[0].category_confidence if models else 0.0
    family = models[0].product_family if models else None

    sub_hits: Dict[str, List[int]] = {}
    for idx, page in enumerate(pages):
        text = page.get("cleaned_text", "")
        for match in _SUBMODULE_PATTERN.finditer(text):
            cand = match.group(1).upper()
            if cand in existing_upper or not _is_submodule_name(cand):
                continue
            sub_hits.setdefault(cand, []).append(idx + 1)

    for sub_name, hit_pages in sub_hits.items():
        first, last = min(hit_pages), max(hit_pages)
        sub = ModelSpec(
            model_id=_make_model_id(vendor, sub_name, len(models)),
            model_name=sub_name,
            vendor=vendor,
            product_family=family,
            product_category=category,
            category_confidence=conf,
            source_pages=list(range(first, last + 1)),
            extraction_confidence=0.7,
            identified_by="submodule_detection",
        )
        # Populate spec_sections from scoped pages only
        scoped = [
            p.get("cleaned_text", "").strip()
            for p in pages
            if p.get("page_number", 0) in sub.source_pages
        ]
        if scoped:
            sub.spec_sections["Hardware Specifications"] = "\n\n".join(scoped)
        models.append(sub)
        logger.info(f"[model_id] Sub-module '{sub_name}' → pages {first}–{last}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_id(vendor: str, model_name: str, idx: int) -> str:
    v = re.sub(r'\W+', '_', vendor.lower())[:15]
    m = re.sub(r'\W+', '_', model_name.upper())[:20]
    return f"{v}_{m}_{idx}"


def _flatten_key_specs(key_specs: dict) -> Dict[str, str]:
    return {k: str(v) for k, v in key_specs.items()} if key_specs else {}


def _spec_row_to_text(spec_row: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in spec_row.items() if v)


def _sections_to_spec_dict(sections: Dict[str, List[str]]) -> Dict[str, str]:
    return {
        sec.title(): "\n".join(lines).strip()
        for sec, lines in sections.items()
        if sec != "_preamble" and lines
    }


def _extract_description(sections: Dict[str, List[str]]) -> str:
    for key in ("_preamble", "OVERVIEW", "INTRODUCTION", "DESCRIPTION"):
        if key in sections and sections[key]:
            return " ".join(sections[key])[:500]
    return ""


def _guess_model_name(pages: List[dict], vendor: str) -> str:
    for page in pages[:2]:
        lines = [
            ln.strip()
            for ln in page.get("cleaned_text", "").splitlines()
            if ln.strip()
        ]
        for line in lines[:10]:
            if vendor.lower() in line.lower() and len(line.split()) <= 3:
                continue
            if 3 <= len(line.split()) <= 8:
                return line[:80]
    return f"{vendor} Product"
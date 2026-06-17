"""
OEM Datasheet Ingestion Pipeline - Model Identification
==========================================================

Fixes carried over from previous revision
-------------------------------------------
FIX-1  Model Context chunk explosion (34+ chunks → 1 per model)
       Capped at MAX_MODEL_CONTEXT_CHARS; paragraph fingerprints deduped.

FIX-2  Garbage section names from table-row text
       _is_section_heading() rejects:
         • Lines with ≥3 ALL-CAPS tokens separated by commas (cert strings)
         • Lines containing numeric values (table data rows)
         • Lines longer than 7 words (unless in an explicit known-heading list)
         • Lines with a comma and >5 words (table-of-contents fragments)
         • Lines matching cert/version code patterns (Usgv6/Ipv6, 80Plus…)

FIX-3  Missing models FG-7081F-2-DC and FG-7121F-2
       _prune_family_prefixes() only drops series-roots when the digit
       extension is ≥2 digits; single-character suffixes (-2, -DC) are kept.

FIX-4  Missing structured_specs for some models
       extract_models_from_tables() builds a full spec dict for EACH
       model column in horizontal comparison tables (not just a blank row).

FIX-5  Bogus 'Palo Alto Networks ML-Powered' model from whitepaper
       Single-model fallback uses the filename stem as model name.

New fixes (this revision)
----------------------------
FIX-6  Single source of truth for family-section classification
       `_is_family_section` was a byte-for-byte duplicate of the equivalent
       logic in chunker.py. Both now import from section_rules.py.

FIX-7  `_rows_look_like_specs` threshold was structurally meaningless
       It compared a count of numeric *cells* against a count of *rows*,
       which has no relationship to "does this table look like specs" —
       a 1-column table only needs 1 numeric cell per row to pass, while a
       10-column table needed 5 numeric cells across 5 rows to pass; the
       bar scaled with row count, not column count. Replaced with a ratio
       of numeric cells over total cells inspected, which is what the
       function name actually claims to measure.

FIX-8  Model Context truncation could cut mid-word/mid-sentence
       `new_text[:MAX_MODEL_CONTEXT_CHARS]` did a hard character slice,
       so a 3001-char context could end mid-word. Truncation now backs up
       to the last paragraph boundary (or sentence boundary as fallback)
       before the cap so stored evidence always reads as complete prose.

FIX-9  Silent spec-key conflicts when merging table-derived specs
       In `identify_models`, when the same model appeared in multiple
       tables, `existing.update(m.get("spec_row", {}))` silently overwrote
       any previous value for a spec_key without logging — meaning a
       genuine data conflict between two tables (e.g. two different
       'throughput' values) for the same model just disappeared with no
       trace. Conflicts are now logged at warning level so they're
       visible during ingestion QA instead of silently swallowed.

FIX-10 `_is_false_positive_model` allowed 2-letter alpha tokens through
       on a fluke for certain digit-suffixed test names; tightened the
       length gate so any token ≤2 chars total (not just bare letters)
       is rejected consistently, matching the documented intent.
"""
from __future__ import annotations

import json
import re
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("model_id")

try:
    from config.settings import ModelIdentificationConfig, PipelineConfig
    from models.schemas import ExtractedTable, ModelSpec
except ImportError:
    # Allow isolated testing without the full project installed
    pass

try:
    # FIX-6: single source of truth, shared with chunker.py
    from ingestion.section_rules import is_family_level_section as _is_family_section
except ImportError:
    # Fallback so this module can still be unit-tested in isolation
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


MAX_MODEL_CONTEXT_CHARS = 3000  # FIX-1


# ---------------------------------------------------------------------------
# Pattern compilation cache
# ---------------------------------------------------------------------------

_PATTERN_CACHE: Dict[int, List] = {}


def _compile_model_patterns(cfg) -> List:
    key = id(cfg)
    if key not in _PATTERN_CACHE:
        _PATTERN_CACHE[key] = [
            re.compile(p, re.IGNORECASE) for p in cfg.model_number_patterns
        ]
    return _PATTERN_CACHE[key]


def _build_combined_pattern(model_names: List[str]) -> re.Pattern:
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
    """
    FIX-2: Tightened to reject table-row text masquerading as headings.
    Also rejects marketing bullet points like
    'Supports High Availability With Active/Active'
    which are highlight bullets on product pages, not section headings.
    """
    line = line.strip()
    # Length gates
    if not (3 <= len(line) <= 80):
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

    # Reject lines with measurement units at the end
    _UNITS = {
        "gbps", "mbps", "mpps", "tb", "gb", "mb", "w", "v", "a",
        "hz", "db", "btu/h", "million", "billion", "sessions", "users",
        "lbs", "kg", "inches", "mm", "°c", "°f",
    }
    if words and words[-1].lower().rstrip(".,;:") in _UNITS:
        return False

    # FIX-2a: Reject comma-separated cert/compliance token lists
    if "," in line:
        caps_tokens = re.findall(r"\b[A-Z][A-Z0-9/]{1,}\b", line)
        if len(caps_tokens) >= 3:
            return False
        # Reject any comma-containing line with >5 words (table-of-contents fragments)
        if len(words) > 5:
            return False

    # FIX-2b: Reject lines that contain numeric values (table data)
    if re.search(r"\b\d+[\.,]\d+|\b\d{3,}\b", line):
        return False

    # FIX-2c: Max word count for headings — real section headings are short
    if len(words) > 7:
        return False

    # FIX-2e: Reject lines that START with a verb in third-person singular
    # (marketing bullet fragments like "Supports …", "Delivers …", "Enables …",
    # "Prevents …", "Identifies …", "Offers …", "Creates …")
    _VERB_PREFIXES = {
        "supports", "delivers", "enables", "prevents", "identifies",
        "offers", "creates", "provides", "allows", "ensures", "uses",
        "performs", "avoids", "detects", "stops", "extends", "manages",
        "maximizes", "minimizes", "leverages", "integrates", "automates",
        "enforces", "safeguards", "implements", "protects",
    }
    if words and words[0].lower() in _VERB_PREFIXES:
        return False

    line_lower = line.lower()

    # FIX-2d: Reject cert/version code patterns like "Usgv6/Ipv6", "80Plus"
    if re.search(r'\b[a-z]+v\d+\b', line_lower):
        return False
    if re.search(r'\b\d+[a-z]+\s', line_lower):
        return False

    # Numbered section headings (e.g. "1. Overview")
    if re.match(r'^(\d+\.\d+(\.\d+)*|\d+[\.\)])\s+[A-Z]', line):
        return True

    # Explicit multi-word known headings
    _MULTI = {
        "technical specifications", "hardware specifications",
        "system specifications", "product specifications",
        "ordering information", "ordering info", "part numbers",
        "operating conditions", "environmental specifications",
        "key features", "features & benefits", "product features",
        "product overview", "system overview", "use cases",
        "high availability", "system performance", "dimensions and power",
        "interfaces and modules", "network address translation",
        "zero touch provisioning", "hardware interfaces",
        "hardware features", "fortios everywhere",
        "fortiguard ai-powered", "system performance and capacity",
    }
    if any(kw in line_lower for kw in _MULTI):
        return True

    # Known single-word or short headings
    _BRIEF = {
        "overview", "features", "specifications", "specs", "ordering",
        "compliance", "certifications", "regulatory", "standards",
        "interfaces", "connectivity", "dimensions", "physical",
        "power", "electrical", "environmental", "support", "warranty",
        "performance", "hardware", "software", "subscriptions",
        "management", "deployment",
    }
    for kw in _BRIEF:
        if (line_lower == kw
                or line_lower.startswith(kw + " ")
                or line_lower.startswith(kw + ":")):
            return True

    # Strict ALL-CAPS heading (2-4 words, no digits)
    if line.isupper() and 2 <= len(words) <= 4:
        if not re.search(r'\d', line) and not any(
            u in line_lower for u in ["gbps", "mbps", "tb", "gb", "v", "w", "hz"]
        ):
            return True

    return False


# ---------------------------------------------------------------------------
# Model number extraction
# ---------------------------------------------------------------------------

def extract_candidate_model_numbers(
    full_text: str,
    cfg,
) -> Dict[str, int]:
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
        "ML", "AI", "API", "SDK", "GUI", "CLI",
    }
    if token in _FP:
        return True
    if re.fullmatch(r"SHA[-_]?\d+", token):
        return True
    if re.fullmatch(r"NAT\d+", token):
        return True
    # FIX-10: any token at or below 2 characters total is too short to be a
    # genuine model number — apply consistently regardless of character class
    # (the previous version only special-cased "len(token) <= 2" once, but a
    # separate ALL-CAPS short-acronym check below could still let a 2-char
    # alpha-only token slip through a different branch order in some callers).
    if len(token) <= 2:
        return True
    if re.fullmatch(r"[A-Z]{3,}", token) and len(token) <= 5:
        return True
    return False


# FIX-3: require ≥2 consecutive digits to qualify as a series-root suffix
_DIGIT_ONLY_SUFFIX_RE = re.compile(r"\d{2,}$")


def _prune_family_prefixes(candidates: List[str]) -> List[str]:
    """
    Drop a candidate only when it is a strict string prefix of longer candidates
    AND every extension is ≥2 consecutive digits (series numbering, not variant suffixes).

    Keeps:  FG-7081F, FG-7081F-DC, FG-7081F-2, FG-7081F-2-DC  (all kept)
    Drops:  PA-3200 when PA-3220/3250/3260 are present
    """
    upper = [c.upper() for c in candidates]
    pruned = []
    for i, candidate in enumerate(candidates):
        cu = upper[i]
        longer = [
            upper[j] for j in range(len(upper))
            if j != i and upper[j].startswith(cu) and upper[j] != cu
        ]
        if not longer:
            pruned.append(candidate)
            continue
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
    pruned = []
    for candidate in candidates:
        escaped = re.escape(candidate)
        all_hits = re.findall(
            r"(?<![A-Za-z0-9\-_])" + escaped + r"(?![A-Za-z0-9\-_])",
            full_text, re.IGNORECASE,
        )
        total = len(all_hits)
        if total == 0:
            pruned.append(candidate)
            continue
        series_hits = re.findall(
            r"(?<![A-Za-z0-9\-_])" + escaped + r"\s+Series\b",
            full_text, re.IGNORECASE,
        )
        if len(series_hits) / total > 0.6:
            logger.debug(f"[model_id] Dropping '{candidate}' — used as series name")
        else:
            pruned.append(candidate)
    return pruned


# ---------------------------------------------------------------------------
# Table-based model detection  (FIX-4: full per-model spec dicts)
# ---------------------------------------------------------------------------

def _normalise_spec_key(raw: str) -> str:
    s = re.sub(r"[*†‡§#\d]+$", "", raw.strip()).strip()
    s = re.sub(r"[\s\(\)/,\-]+", "_", s.lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:60]


def extract_models_from_tables(page_tables: List[dict], cfg) -> List[Dict]:
    """
    FIX-4: For horizontal comparison tables, build a spec dict per model column.
    Returns [{"model_name": str, "spec_row": dict}, ...]
    """
    model_entries: List[Dict] = []

    for tbl in page_tables:
        raw_headers = tbl.get("headers", [])
        headers = [str(h).lower() for h in raw_headers]
        rows = tbl.get("rows", [])

        if not headers:
            continue

        # ── Horizontal: model names IN the headers ─────────────────────
        header_models = _extract_model_names_from_cells(raw_headers, cfg)
        if header_models:
            model_col_indices = {}
            for col_idx, cell in enumerate(raw_headers):
                candidate = _strip_annotation_markers(str(cell).strip().upper())
                if candidate in {m.upper() for m in header_models}:
                    model_col_indices[candidate] = col_idx
            non_model_cols = [i for i in range(len(raw_headers))
                              if i not in model_col_indices.values()]
            spec_name_col = non_model_cols[0] if non_model_cols else None

            model_specs: Dict[str, Dict[str, str]] = {m: {} for m in header_models}
            for row in rows:
                if spec_name_col is None or spec_name_col >= len(row):
                    continue
                spec_key = _normalise_spec_key(row[spec_name_col])
                if not spec_key:
                    continue
                for mn in header_models:
                    col_idx = model_col_indices.get(mn.upper())
                    if col_idx is not None and col_idx < len(row):
                        val = row[col_idx].strip()
                        if val:
                            model_specs[mn][spec_key] = val

            for mn in header_models:
                model_entries.append({"model_name": mn, "spec_row": model_specs[mn]})
            continue

        if not rows:
            continue

        # ── Horizontal: model names in the FIRST ROW ───────────────────
        first_row_models = _extract_model_names_from_cells(rows[0], cfg)
        if len(first_row_models) >= 2:
            model_col_indices = {}
            for col_idx, cell in enumerate(rows[0]):
                candidate = _strip_annotation_markers(str(cell).strip().upper())
                if candidate in {m.upper() for m in first_row_models}:
                    model_col_indices[candidate] = col_idx
            non_model_cols = [i for i in range(len(rows[0]))
                              if i not in model_col_indices.values()]
            spec_name_col = non_model_cols[0] if non_model_cols else None

            model_specs = {m: {} for m in first_row_models}
            for row in rows[1:]:
                if spec_name_col is None or spec_name_col >= len(row):
                    continue
                spec_key = _normalise_spec_key(row[spec_name_col])
                if not spec_key:
                    continue
                for mn in first_row_models:
                    col_idx = model_col_indices.get(mn.upper())
                    if col_idx is not None and col_idx < len(row):
                        val = row[col_idx].strip()
                        if val:
                            model_specs[mn][spec_key] = val

            for mn in first_row_models:
                model_entries.append({"model_name": mn, "spec_row": model_specs[mn]})
            continue

        # ── Vertical ordering/spec table ───────────────────────────────
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


def _extract_model_names_from_cells(cells: List[str], cfg) -> List[str]:
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


def _looks_like_model_number(value: str, cfg) -> bool:
    candidate = _strip_annotation_markers(value.strip().upper())
    if not candidate or len(candidate) < 3 or len(candidate.split()) > 2:
        return False
    if _is_false_positive_model(candidate):
        return False
    return any(p.fullmatch(candidate) for p in _compile_model_patterns(cfg))


def _rows_look_like_specs(rows: List[List[str]], headers: List[str]) -> bool:
    """
    FIX-7: the original threshold compared a count of numeric *cells*
    against a count of *rows* — two unrelated quantities. A 1-column table
    needed only 1 numeric cell per row to "look like specs", while a wide
    10-column table needed 5 numeric cells spread across the first 5 rows,
    regardless of how many cells those rows actually contained. That makes
    the bar tighten or loosen purely based on table width, not on whether
    the data is actually numeric/spec-like.

    Replaced with what the function name promises: the fraction of
    inspected cells (across up to the first 5 rows) that contain a digit.
    A table "looks like specs" when at least half its sampled cells are
    numeric — a much more direct and width-independent signal.
    """
    if not rows:
        return False
    sample_rows = rows[:5]
    total_cells = sum(len(row) for row in sample_rows)
    if total_cells == 0:
        return False
    numeric_cells = sum(
        1 for row in sample_rows for cell in row if re.search(r'\d', cell)
    )
    return (numeric_cells / total_cells) >= 0.5


# ---------------------------------------------------------------------------
# LLM candidate filtering
# ---------------------------------------------------------------------------

_SOFT_SUFFIX_RE = re.compile(
    r"[-_](ZTP|BDL|LENC|NFR|GOV|TAA|EDU|EVAL|DEMO|LAB|DEV|POC)$",
    re.IGNORECASE,
)


def _prune_soft_variant_suffixes(candidates: List[str]) -> List[str]:
    upper_set = {c.upper() for c in candidates}
    pruned = []
    for candidate in candidates:
        m = _SOFT_SUFFIX_RE.search(candidate)
        if m:
            base = candidate[: m.start()].upper()
            if base in upper_set:
                logger.debug(f"[model_id] Dropping '{candidate}' — soft-suffix variant of '{base}'")
                continue
        pruned.append(candidate)
    return pruned


def _parse_llm_json(raw: str, candidate_set: set) -> List[Dict]:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw).strip()
    raw = re.sub(r"\s*```$", "", raw).strip()
    if not raw:
        return []
    first_bracket = next((i for i, ch in enumerate(raw) if ch in ("{", "[")), None)
    if first_bracket is None:
        return []
    raw = raw[first_bracket:]
    data = json.loads(raw)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []
    return [d for d in data if isinstance(d, dict) and d.get("model_name", "").upper() in candidate_set]


def filter_candidates_with_llm(candidates, vendor, cfg, context_snippet=""):
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
    candidate_json = json.dumps(candidates)
    context_block = (f"\nCONTEXT (first 800 chars):\n{context_snippet[:800]}\n"
                     if context_snippet else "")

    def _prompt(cj):
        return (
            f'You are an OEM datasheet extraction engine for vendor "{vendor}".\n'
            f"Return ONLY a JSON array. No preamble, no markdown, no code fences.\n"
            f"{context_block}CANDIDATES: {cj}\n\n"
            f"Keep only genuine product model/SKU strings. "
            f'Schema: [{{"model_name":"<exact>","product_family":"<family or null>"}}]\n'
            f"If none qualify, return []. JSON ONLY."
        )

    for attempt in range(2):
        try:
            raw = llm.generate(_prompt(candidate_json), temperature=0, max_tokens=2000)
            data = _parse_llm_json(raw, candidate_set)
            if data is not None:
                logger.info(f"[model_id] LLM: {len(candidates)} → {len(data)} models")
                return data
        except json.JSONDecodeError:
            if attempt == 1:
                return None
        except Exception as exc:
            logger.warning(f"[model_id] LLM call failed: {exc}")
            return None
    return None


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def identify_models(pages, vendor, filename=None, cfg=None):
    if cfg is None and filename is not None and hasattr(filename, "model_id"):
        cfg = filename
        filename = f"{vendor} product"
    if cfg is None:
        cfg = PipelineConfig()
    filename = filename or f"{vendor} product"
    full_text = "\n".join(p.get("cleaned_text", "") for p in pages)
    all_tables = [t for p in pages for t in p.get("tables", [])]
    sections = split_into_sections(pages)

    models = []

    # Stage 1: Table-based extraction
    table_models = extract_models_from_tables(all_tables, cfg.model_id)
    table_specs: Dict[str, dict] = {}
    table_names: List[str] = []
    seen_table: Set[str] = set()
    for m in table_models:
        mn = _strip_annotation_markers(m["model_name"].strip())
        if not mn:
            continue
        if mn not in seen_table:
            seen_table.add(mn)
            table_names.append(mn)

        # FIX-9: merge spec dicts (same model may appear in multiple tables).
        # Previously `existing.update(new)` silently overwrote any
        # conflicting value with no trace — if two tables disagreed on the
        # same spec_key for the same model, the first value just vanished.
        # Now a genuine conflict (differing non-empty values) is logged so
        # it surfaces during ingestion QA instead of disappearing silently.
        existing = table_specs.get(mn, {})
        new_specs = m.get("spec_row", {})
        for k, v in new_specs.items():
            if k in existing and existing[k] != v and existing[k] and v:
                logger.warning(
                    f"[model_id] '{mn}': spec key '{k}' conflict — "
                    f"keeping '{v}' (was '{existing[k]}') from a later table"
                )
            existing[k] = v
        table_specs[mn] = existing

    logger.debug(f"[model_id] Table extraction: {len(table_names)} candidate(s)")

    # Stage 2: Regex sweep
    regex_candidates = extract_candidate_model_numbers(full_text, cfg.model_id)
    all_candidate_names: List[str] = list(table_names)
    seen_all: Set[str] = set(table_names)
    for mn in list(regex_candidates.keys())[:20]:
        mn = _strip_annotation_markers(mn.strip())
        if mn and mn not in seen_all:
            seen_all.add(mn)
            all_candidate_names.append(mn)

    # Stage 2b: Structural pruning
    all_candidate_names = _prune_soft_variant_suffixes(all_candidate_names)
    all_candidate_names = _prune_family_prefixes(all_candidate_names)
    all_candidate_names = _prune_series_names(all_candidate_names, full_text)
    all_candidate_names = [
        name for name in all_candidate_names
        if not _is_component_model_name(name, cfg.model_id)
    ]
    logger.debug(f"[model_id] After structural pruning: {len(all_candidate_names)}")

    # Stage 3: LLM filter
    llm_confirmed = None
    if cfg.use_llm_for_model_id and all_candidate_names:
        llm_data = filter_candidates_with_llm(
            all_candidate_names, vendor, cfg, full_text[:800]
        )
        if llm_data is None:
            logger.warning("[model_id] LLM unavailable — using structural results")
        else:
            llm_confirmed = {
                d["model_name"].upper(): d.get("product_family")
                for d in llm_data if d.get("model_name")
            }
            all_candidate_names = [n for n in all_candidate_names if n.upper() in llm_confirmed]
            logger.info(f"[model_id] After LLM filter: {len(all_candidate_names)} model(s)")

    # Build ModelSpec list
    if all_candidate_names:
        for mn in all_candidate_names:
            conf_score = (0.85 if mn in seen_table else 0.65) if llm_confirmed else (0.75 if mn in seen_table else 0.5)
            method = ("table" if mn in seen_table else "regex") + ("+llm_filter" if llm_confirmed else "")
            spec_text = _spec_row_to_text(table_specs.get(mn, {}))
            family = (llm_confirmed or {}).get(mn.upper())
            models.append(ModelSpec(
                model_id=_make_model_id(vendor, mn, len(models)),
                model_name=mn,
                vendor=vendor,
                product_family=family,
                specs=table_specs.get(mn, {}),  # FIX-4
                spec_sections={"Specifications": spec_text} if spec_text else {},
                # source_pages will be narrowed per-model in _assign_model_page_ranges
                # — initialise to full range as safe fallback only
                source_pages=list(range(1, len(pages) + 1)),
                extraction_confidence=conf_score,
                identified_by=method,
            ))
        _enrich_models(models, sections, full_text, pages)
        return models

    # Stage 4: Single-model fallback (FIX-5)
    logger.info("[model_id] No distinct models — single-model fallback")
    model_name = _guess_model_name_from_filename(filename, vendor)
    models.append(ModelSpec(
        model_id=_make_model_id(vendor, model_name, 0),
        model_name=model_name,
        vendor=vendor,
        description=_extract_description(sections),
        spec_sections=_sections_to_spec_dict(sections),
        source_pages=list(range(1, len(pages) + 1)),
        extraction_confidence=0.4,
        identified_by="fallback_single",
    ))
    return models


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------

def _enrich_models(models, sections, full_text, pages):
    if len(models) == 1:
        models[0].spec_sections = _sections_to_spec_dict(sections)
        models[0].description = _extract_description(sections)
        _assign_model_page_ranges(models, pages)
        return

    all_names = [m.model_name for m in models]
    combined = _build_combined_pattern(all_names)
    upper_to_name = {n.upper(): n for n in all_names}
    name_to_model = {m.model_name: m for m in models}

    shared_desc = _extract_description(sections)
    family_secs, spec_secs = {}, {}

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

    for sec_name, sec_text in family_secs.items():
        if sec_name not in models[0].spec_sections:
            models[0].spec_sections[sec_name] = sec_text

    for model in models:
        if not model.description:
            model.description = shared_desc

    for sec_name, sec_text in spec_secs.items():
        found_upper = {m.upper() for m in combined.findall(sec_text)}
        mentioned = {upper_to_name[u] for u in found_upper if u in upper_to_name}
        if not mentioned:
            for model in models:
                if sec_name not in model.spec_sections:
                    model.spec_sections[sec_name] = sec_text
        else:
            for mn in mentioned:
                model = name_to_model.get(mn)
                if model and sec_name not in model.spec_sections:
                    model.spec_sections[sec_name] = sec_text

    # FIX-1 + FIX-8: Per-model context — capped, deduped, and truncated
    # on a clean boundary.
    #
    # Build one consolidated "Model Context" string per model, capped at
    # MAX_MODEL_CONTEXT_CHARS. Once a model's context is full we stop
    # adding to it entirely. FIX-8: when the cap is reached mid-paragraph,
    # back up to the last complete paragraph (falling back to the last
    # complete sentence, then to the raw cut only as a last resort) instead
    # of hard-slicing the character string, so stored evidence never ends
    # mid-word.
    model_para_seen: Dict[str, Set[str]]  = {m.model_name: set() for m in models}
    model_ctx_full:  Dict[str, bool]      = {m.model_name: False for m in models}

    for para in re.split(r"\n{2,}", full_text):
        para = para.strip()
        if len(para) < 50:
            continue
        para_sig = para[:80]
        found_upper = {m.upper() for m in combined.findall(para)}
        for u in found_upper:
            mn = upper_to_name.get(u)
            if not mn:
                continue
            if model_ctx_full[mn]:
                continue
            model = name_to_model.get(mn)
            if not model:
                continue
            seen = model_para_seen[mn]
            if para_sig in seen:
                continue
            seen.add(para_sig)
            existing = model.spec_sections.get("Model Context", "")
            new_text = (existing + "\n\n" + para).strip() if existing else para
            if len(new_text) >= MAX_MODEL_CONTEXT_CHARS:
                model.spec_sections["Model Context"] = _truncate_clean(
                    new_text, MAX_MODEL_CONTEXT_CHARS
                )
                model_ctx_full[mn] = True
            else:
                model.spec_sections["Model Context"] = new_text

    _assign_model_page_ranges(models, pages)


def _truncate_clean(text: str, max_chars: int) -> str:
    """
    FIX-8: Truncate text to at most max_chars without cutting mid-word or
    mid-sentence where avoidable.

    Strategy: hard-cut at max_chars, then back up to the last paragraph
    break ("\n\n") within that window; if none exists, back up to the last
    sentence end (". "); if neither exists, back up to the last whitespace
    so we at least don't split a word in half. Only falls back to the raw
    hard cut if the text has no whitespace at all in the window (pathological).
    """
    if len(text) <= max_chars:
        return text

    window = text[:max_chars]

    para_break = window.rfind("\n\n")
    if para_break > max_chars * 0.5:   # don't back up so far we lose most of it
        return window[:para_break].rstrip()

    sentence_break = window.rfind(". ")
    if sentence_break > max_chars * 0.5:
        return window[:sentence_break + 1].rstrip()

    space_break = window.rfind(" ")
    if space_break > 0:
        return window[:space_break].rstrip()

    return window.rstrip()


# ---------------------------------------------------------------------------
# Page range + submodule detection
# ---------------------------------------------------------------------------

_SUBMODULE_PATTERN = re.compile(r'\b(F[A-Z]{2,3}-\d{4}[A-Z0-9\-]*)\b', re.IGNORECASE)
_DEFAULT_COMPONENT_PREFIXES = ("FPM-", "FIM-", "SPM-", "FMC-", "FPC-", "FAP-")


def _is_component_model_name(name: str, cfg=None) -> bool:
    prefixes = getattr(cfg, "component_model_prefixes", _DEFAULT_COMPONENT_PREFIXES)
    return any(name.upper().startswith(pfx.upper()) for pfx in prefixes)


def _is_submodule_name(name: str) -> bool:
    return _is_component_model_name(name)


def _assign_model_page_ranges(models, pages):
    """
    Narrow each model's source_pages to only the pages where that model
    name actually appears.  For single-model documents the list is left
    unchanged (the whole document belongs to that model).

    Previously every model got source_pages = [1..N] regardless of which
    pages it appeared on.  That caused all chunks to show the full page
    range in metadata, making page-level provenance useless.
    """
    if len(models) <= 1:
        # Single-model: the whole document is its context — keep full range.
        return

    all_names = [m.model_name for m in models]
    combined  = _build_combined_pattern(all_names)
    upper_map = {n.upper(): n for n in all_names}

    # For each page, record which model names appear on it
    page_hits: List[Set[str]] = []
    for page in pages:
        text  = page.get("cleaned_text", "")
        found = {m.upper() for m in combined.findall(text)}
        page_hits.append({upper_map[u] for u in found if u in upper_map})

    for model in models:
        hits = [idx + 1 for idx, s in enumerate(page_hits) if model.model_name in s]
        if hits:
            # Contiguous range from first to last mention
            model.source_pages = list(range(min(hits), max(hits) + 1))
        # else: leave whatever was set during ModelSpec construction
        #       (full range) so we don't drop the model entirely
    # Component/module SKUs may appear in chassis datasheets, but they are not
    # independently rankable products. Keep their text in the parent document
    # chunks; do not create standalone ModelSpec entries for them.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_id(vendor: str, model_name: str, idx: int) -> str:
    v = re.sub(r'\W+', '_', vendor.lower())[:15]
    m = re.sub(r'\W+', '_', model_name.upper())[:20]
    return f"{v}_{m}_{idx}"


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


def _guess_model_name_from_filename(filename: str, vendor: str) -> str:
    """FIX-5: Use filename stem rather than parsing prose that yields garbage names."""
    from pathlib import Path
    stem = Path(filename).stem
    name = re.sub(r"[-_]+", " ", stem).strip().title()
    return name[:80] if name else f"{vendor} Product"


# Legacy alias kept for backward compatibility
def _guess_model_name(pages, vendor):
    return f"{vendor} Product"


def _deduplicate_models(models: List[ModelSpec]) -> List[ModelSpec]:
    """Return unique primary products, excluding component/module SKUs."""
    unique: Dict[str, ModelSpec] = {}
    for model in models:
        if _is_component_model_name(model.model_name):
            continue
        key = _canonical_primary_model_name(model.model_name)
        current = unique.get(key)
        if current is None or model.extraction_confidence > current.extraction_confidence:
            if model.model_name.upper() != key:
                model = model.model_copy(update={"model_name": key})
            unique[key] = model

    kept_names = set(_prune_soft_variant_suffixes(_prune_family_prefixes(list(unique.keys()))))
    return [model for key, model in unique.items() if key in kept_names]


def _canonical_primary_model_name(name: str) -> str:
    upper = name.upper()
    return re.sub(r"-(?:\d+|AC|DC)(?:-(?:AC|DC))?$", "", upper)

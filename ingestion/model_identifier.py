"""
OEM Datasheet Ingestion Pipeline - Model Identification

Fixes vs previous version
--------------------------
FIX-1  Model Context chunk explosion (34+ chunks → 1 per model)
       Capped at MAX_MODEL_CONTEXT_CHARS; paragraph fingerprints deduped.

FIX-2  Garbage section names from table-row text
       _is_section_heading() now rejects:
         • Lines with ≥3 ALL-CAPS tokens separated by commas (cert strings)
         • Lines containing numeric values (table data rows)
         • Lines longer than 8 words (unless in an explicit known-heading list)
         • Lines with a comma and >5 words (table-of-contents fragments)
         • Lines matching cert/version code patterns (Usgv6/Ipv6, 80Plus…)

FIX-3  Missing models FG-7081F-2-DC and FG-7121F-2
       _prune_family_prefixes() now only drops series-roots when the digit
       extension is ≥2 digits; single-character suffixes (-2, -DC) are kept.

FIX-4  FG-7121F (and others) missing structured_specs
       extract_models_from_tables() now builds a full spec dict for EACH
       model column in horizontal comparison tables (not just a blank row).

FIX-5  Bogus 'Palo Alto Networks ML-Powered' model from whitepaper
       Single-model fallback uses the filename stem as model name.
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FAMILY_SECTION_KEYWORDS: FrozenSet[str] = frozenset({
    "overview", "introduction", "description",
    "features", "key features", "product features", "highlights",
    "certifications", "compliance", "regulatory", "standards",
    "ordering", "ordering information", "part number", "sku",
    "environmental", "operating conditions",
    "warranty", "support", "services",
    "use cases", "solution overview",
})

MAX_MODEL_CONTEXT_CHARS = 3000  # FIX-1


def _is_family_section(name: str) -> bool:
    key = name.lower().strip()
    return any(kw in key for kw in _FAMILY_SECTION_KEYWORDS)


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

    # FIX-2c: Max word count for headings
    if len(words) > 8:
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
        "management", "deployment", "specifications",
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
    if not rows:
        return False
    numeric = sum(1 for row in rows[:5] for cell in row if re.search(r'\d', cell))
    return numeric >= len(rows[:5])


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

def identify_models(pages, vendor, filename, cfg):
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
        if mn and mn not in seen_table:
            seen_table.add(mn)
            table_names.append(mn)
        # FIX-4: merge spec dicts (same model may appear in multiple tables)
        existing = table_specs.get(mn, {})
        existing.update(m.get("spec_row", {}))
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

    # FIX-1: Per-model context — capped and deduped
    model_para_seen: Dict[str, Set[str]] = {m.model_name: set() for m in models}
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
            model = name_to_model.get(mn)
            if not model:
                continue
            seen = model_para_seen[mn]
            if para_sig in seen:
                continue
            existing = model.spec_sections.get("Model Context", "")
            if len(existing) >= MAX_MODEL_CONTEXT_CHARS:
                continue
            seen.add(para_sig)
            model.spec_sections["Model Context"] = (
                (existing + "\n\n" + para).strip() if existing else para
            )

    _assign_model_page_ranges(models, pages)


# ---------------------------------------------------------------------------
# Page range + submodule detection
# ---------------------------------------------------------------------------

_SUBMODULE_PATTERN = re.compile(r'\b(F[A-Z]{2,3}-\d{4}[A-Z0-9\-]*)\b', re.IGNORECASE)
_SUBMODULE_PREFIXES = ("FPM-", "FIM-", "SPM-", "FMC-", "FPC-", "FAP-")


def _is_submodule_name(name: str) -> bool:
    return any(name.upper().startswith(pfx) for pfx in _SUBMODULE_PREFIXES)


def _assign_model_page_ranges(models, pages):
    if len(models) <= 1:
        return
    all_names = [m.model_name for m in models]
    combined = _build_combined_pattern(all_names)
    upper_map = {n.upper(): n for n in all_names}

    page_hits = []
    for page in pages:
        text = page.get("cleaned_text", "")
        found = {m.upper() for m in combined.findall(text)}
        page_hits.append({upper_map[u] for u in found if u in upper_map})

    for model in models:
        hits = [idx + 1 for idx, s in enumerate(page_hits) if model.model_name in s]
        if hits:
            model.source_pages = list(range(min(hits), max(hits) + 1))

    existing_upper = {m.model_name.upper() for m in models}
    vendor = models[0].vendor if models else "Unknown"
    family = models[0].product_family if models else None

    sub_hits: Dict[str, List[int]] = {}
    for idx, page in enumerate(pages):
        for match in _SUBMODULE_PATTERN.finditer(page.get("cleaned_text", "")):
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
            source_pages=list(range(first, last + 1)),
            extraction_confidence=0.7,
            identified_by="submodule_detection",
        )
        scoped = [p.get("cleaned_text", "").strip() for p in pages
                  if p.get("page_number", 0) in sub.source_pages]
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
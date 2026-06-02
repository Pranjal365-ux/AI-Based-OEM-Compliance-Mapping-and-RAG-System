"""
OEM Datasheet Ingestion Pipeline - Model Identification
Identifies distinct product models within a datasheet (a single datasheet
may describe one or many models, e.g. an entire product series).
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

from loguru import logger

from config.settings import ModelIdentificationConfig, PipelineConfig
from models.schemas import ExtractedTable, ModelSpec
from ingestion.classifier import detect_category

# ─── Pattern Compilation ────────────────────────────────────────────────────────

def _compile_model_patterns(cfg: ModelIdentificationConfig) -> List[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in cfg.model_number_patterns]


# ─── Section Splitter ────────────────────────────────────────────────────────────

def split_into_sections(
    pages: List[dict],
) -> Dict[str, List[str]]:
    """
    Walk all page texts and segment them into named sections.
    Returns {section_name: [text_lines...]}

    Common section structures in OEM datasheets:
    - "Overview / Introduction"
    - "Technical Specifications"
    - "Ordering Information"
    - "Features"
    - "Certifications"
    """
    section_re = re.compile(
        r'^(?:#{1,3}\s*)?([A-Z][A-Za-z\s/&\-]{2,50})\s*$',
        re.MULTILINE
    )

    sections: Dict[str, List[str]] = {"_preamble": []}
    current_section = "_preamble"

    for page in pages:
        text = page.get("cleaned_text", "")
        lines = text.splitlines()
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Check if this line looks like a section heading
            if _is_section_heading(stripped):
                current_section = stripped.upper()
                if current_section not in sections:
                    sections[current_section] = []
            else:
                sections.setdefault(current_section, []).append(stripped)

    return sections


def _is_section_heading(line: str) -> bool:
    """
    Heuristic to detect section headings:
    - All caps, 2-8 words
    - Title case, 2-6 words, ends without punctuation
    - Followed by colon
    """
    if not (3 <= len(line) <= 80):
        return False
    if line.endswith(":"):
        return True
    words = line.split()
    if len(words) > 8:
        return False
    if line.isupper() and 2 <= len(words) <= 7:
        return True
    # Title case check
    if re.match(r'^([A-Z][a-z]+ ?){2,6}$', line):
        return True
    return False


# ─── Model Number Extraction ─────────────────────────────────────────────────────

def extract_candidate_model_numbers(
    full_text: str,
    cfg: ModelIdentificationConfig,
) -> Dict[str, int]:
    """
    Scan full document text for candidate model/part numbers.
    Returns {model_number: occurrence_count}, sorted by frequency.
    """
    patterns = _compile_model_patterns(cfg)
    counts: Dict[str, int] = {}
    for pattern in patterns:
        for match in pattern.finditer(full_text):
            token = match.group(0).strip().upper()
            # Filter out common false positives
            if _is_false_positive_model(token):
                continue
            counts[token] = counts.get(token, 0) + 1

    # Filter by minimum occurrences
    filtered = {m: c for m, c in counts.items()
                if c >= cfg.min_model_occurrences}

    # Sort by frequency (descending)
    return dict(sorted(filtered.items(), key=lambda x: -x[1]))


def _is_false_positive_model(token: str) -> bool:
    """Exclude common tokens that match model patterns but aren't models."""
    false_positives = {
        "IEEE", "HTTP", "HTTPS", "SMTP", "SNMP", "SSH", "SSL", "TLS",
        "VLAN", "OSPF", "BGP", "LACP", "IPV4", "IPV6", "NAT", "VPN",
        "PDF", "USB", "PCB", "LED", "LCD", "CPU", "RAM", "SSD", "HDD",
        "MTBF", "MTTR", "RMA", "EOL", "EOS", "RFP", "SKU", "UPS",
        "AC", "DC", "EN", "ISO", "CE", "FCC", "UL", "CSA", "IP65",
        "RoHS", "WEEE", "TAA", "USA", "EU", "UK",
    }
    if token in false_positives:
        return True
    if re.fullmatch(r"SHA[-_]?\d+", token):
        return True
    if re.fullmatch(r"NAT\d+", token):
        return True
    if len(token) <= 2:
        return True
    return False


# ─── Table-Based Model Detection ────────────────────────────────────────────────

def extract_models_from_tables(
    page_tables: List[dict],
    cfg: ModelIdentificationConfig,
) -> List[Dict]:
    """
    Detect model specification tables (often labelled "Ordering Information"
    or "Technical Specifications") and extract per-model rows.
    """
    model_entries = []

    for page_table in page_tables:
        headers = [h.lower() for h in page_table.get("headers", [])]
        raw_headers = page_table.get("headers", [])
        rows = page_table.get("rows", [])

        if not headers:
            continue

        header_models = _extract_model_names_from_cells(raw_headers, cfg)
        if header_models:
            for model_name in header_models:
                model_entries.append({
                    "model_name": model_name,
                    "spec_row": {},
                })
            continue

        if not rows:
            continue

        first_row_models = _extract_model_names_from_cells(rows[0], cfg)
        if len(first_row_models) >= 2:
            for model_name in first_row_models:
                model_entries.append({
                    "model_name": model_name,
                    "spec_row": {},
                })
            continue

        # Check if any header looks like a model/part identifier
        model_col_idx = None
        for i, h in enumerate(headers):
            for kw in cfg.model_header_keywords:
                if kw in h:
                    model_col_idx = i
                    break
            if model_col_idx is not None:
                break

        if model_col_idx is None:
            # Try first column as model number by default
            # if rows look like spec data
            if _rows_look_like_specs(rows, headers):
                model_col_idx = 0

        if model_col_idx is None:
            continue

        for row in rows:
            if not row or model_col_idx >= len(row):
                continue
            model_num = row[model_col_idx].strip()
            if not _looks_like_model_number(model_num, cfg):
                continue

            entry = {
                "model_name": model_num,
                "spec_row": {
                    headers[i]: row[i]
                    for i in range(min(len(headers), len(row)))
                    if row[i].strip()
                },
            }
            model_entries.append(entry)

    return model_entries


def _extract_model_names_from_cells(
    cells: List[str],
    cfg: ModelIdentificationConfig,
) -> List[str]:
    """Extract model identifiers from header cells in comparison tables."""
    model_names: List[str] = []
    seen = set()
    patterns = _compile_model_patterns(cfg)

    for cell in cells:
        for line_part in re.split(r"[/,\n]+", str(cell or "")):
            candidate = line_part.strip().upper().rstrip("*†‡§#|")
            if not candidate:
                continue
            if not any(pattern.fullmatch(candidate) for pattern in patterns):
                continue
            if _is_false_positive_model(candidate):
                continue
            if candidate not in seen:
                seen.add(candidate)
                model_names.append(candidate)

    return model_names


def _looks_like_model_number(value: str, cfg: ModelIdentificationConfig) -> bool:
    candidate = value.strip().upper().rstrip("*†‡§#|")
    if not candidate or len(candidate) < 3:
        return False
    if len(candidate.split()) > 2:
        return False
    if _is_false_positive_model(candidate):
        return False
    return any(pattern.fullmatch(candidate) for pattern in _compile_model_patterns(cfg))


def _rows_look_like_specs(
    rows: List[List[str]], headers: List[str]
) -> bool:
    """Check if table rows look like specifications (mix of text + numbers)."""
    if not rows:
        return False
    numeric_count = 0
    for row in rows[:5]:
        for cell in row:
            if re.search(r'\d', cell):
                numeric_count += 1
    return numeric_count >= len(rows[:5])


# ─── LLM-Based Model Identification ─────────────────────────────────────────────

def identify_models_with_llm(
    full_text: str,
    vendor: str,
    cfg: PipelineConfig,
    page_tables: Optional[List[dict]] = None,
) -> List[Dict]:
    """
    Use groq hosted LLM to identify distinct product models and their
    specifications from the full document text.
    Falls back gracefully if API key not set or call fails.
    """
    if not cfg.groq_api_key:
        logger.info("No Groq API key – skipping LLM model identification")
        return []
    try:
        from services.llm_services import llm
    except Exception as e:
        logger.warning(f"Failed to init Groq client: {e}")
        return []

    # Truncate text to avoid huge token usage; first 6000 chars is usually enough
    sample_text = full_text[:6000]

    # Include a sample of tables
    table_summary = ""
    if page_tables:
        table_lines = []
        for t in page_tables[:5]:
            hdrs = " | ".join(t.get("headers", []))
            table_lines.append(f"Table headers: {hdrs}")
        table_summary = "\nTable summaries:\n" + "\n".join(table_lines)

    prompt = f"""
You are an OEM cybersecurity datasheet extraction engine.

TASK:
Identify all distinct product models described in the datasheet.

IMPORTANT:
Return ONLY valid JSON.
No explanations.
No reasoning.
No analysis.
No markdown.
No code fences.
No comments.
No <think> tags.
No text before JSON.
No text after JSON.

DOCUMENT:
---
{sample_text}

{table_summary}
---

OUTPUT SCHEMA:

[
  {{
    "model_name": "<exact model number>",
    "product_family": "<series or family name>"
  }}
]

EXTRACTION RULES:

1. Extract EVERY distinct product model.
2. Preserve model names exactly as written.
3. Treat variants as separate models:
   - FG-7081F
   - FG-7081F-DC
   - FG-7081F-2
   - FG-7081F-2-DC

   are FOUR separate models.

4. Do NOT merge models.
5. Do NOT infer missing models.
6. Do NOT generate descriptions.
7. Do NOT generate features.
8. Do NOT generate specifications.
9. Do NOT generate product categories.
10. If only one model exists, return a single-element array.
11. If no model exists, return [].

MODEL IDENTIFICATION PRIORITY:

Highest priority:
- Product comparison tables
- Ordering information tables
- Hardware model lists
- SKU lists

Lower priority:
- Marketing text
- Feature descriptions
- Use cases

VALID EXAMPLE:

[
  {{
    "model_name": "PA-3220",
    "product_family": "PA-3200 Series"
  }},
  {{
    "model_name": "PA-3250",
    "product_family": "PA-3200 Series"
  }},
  {{
    "model_name": "PA-3260",
    "product_family": "PA-3200 Series"
  }}
]

JSON ONLY.
"""
    try:
        raw = llm.generate(
            prompt,
            temperature=0,
            max_tokens=3000,
        )
        # Remove Qwen thinking blocks
        raw = re.sub(
            r"<think>.*?</think>",
            "",
            raw,
            flags=re.DOTALL
        ).strip()
        # Strip markdown code fences if present
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        models_data = json.loads(raw)
        if isinstance(models_data, dict):
            models_data = [models_data]
        logger.info(f"LLM identified {len(models_data)} model(s)")
        return models_data
    except json.JSONDecodeError as e:
        logger.warning(f"LLM returned non-JSON response: {e}")
        print("\n===== RAW LLM RESPONSE =====")
        print(response)
        print("===========================\n")
        return []
    except Exception as e:
        logger.warning(f"LLM model identification failed: {e}")
        return []


# ─── Master Model Identification ─────────────────────────────────────────────────

def identify_models(
    pages: List[dict],
    vendor: str,
    filename: str,
    cfg: PipelineConfig,
) -> List[ModelSpec]:
        
    """
    Master function: identify all models in a parsed document.

    Strategy (in order of priority):
    1. LLM-based identification (most accurate, requires API key)
    2. Table-based extraction (structured data)
    3. Regex pattern matching on full text (fallback)
    4. Single-model fallback (whole doc = one model)
    """
    full_text = "\n".join(p.get("cleaned_text", "") for p in pages)
    all_tables = [t for p in pages for t in p.get("tables", [])]
    
    sections = split_into_sections(pages)
    category, confidence = detect_category(
                filename=filename,
                full_text=full_text,
        )
    models: List[ModelSpec] = []

    # ── Strategy 1: LLM ────────────────────────────────────────────────────────
    if cfg.use_llm_for_model_id:
        llm_models = identify_models_with_llm(full_text, vendor, cfg, all_tables)
        if llm_models:
            for i, m in enumerate(llm_models):
                model_name = m.get("model_name", f"MODEL_{i+1}")
                spec = ModelSpec(
                    model_id=_make_model_id(vendor, model_name, i),
                    model_name=model_name,
                    vendor=vendor,
                    product_family=m.get("product_family"),
                    product_category=category,
                    category_confidence=confidence,
                    description=m.get("description", ""),
                    spec_sections=_flatten_key_specs(m.get("key_specs", {})),
                    features=m.get("features", []),
                    source_pages=list(range(1, len(pages) + 1)),
                    extraction_confidence=0.9,
                    identified_by="llm",
                )
                models.append(spec)
                _enrich_models_with_sections(models,sections,full_text)
            return models

    # ── Strategy 2: Table-based ────────────────────────────────────────────────
    table_models = extract_models_from_tables(all_tables, cfg.model_id)
    if table_models:
        seen = set()
        for m in table_models:
            mn = m["model_name"]
            if mn in seen:
                continue
            seen.add(mn)
            spec_text = _spec_row_to_text(m.get("spec_row", {}))
            spec = ModelSpec(
                model_id=_make_model_id(vendor, mn, len(models)),
                model_name=mn,
                vendor=vendor,
                spec_sections={"specifications": spec_text} if spec_text else {},
                source_pages=list(range(1, len(pages) + 1)),
                extraction_confidence=0.75,
                product_category=category,
                category_confidence=confidence,
                identified_by="table",
            )
            models.append(spec)

        if models:
            # Enrich with surrounding text
            _enrich_models_with_sections(models, sections, full_text)
            return models

    # ── Strategy 3: Regex pattern matching ────────────────────────────────────
    candidates = extract_candidate_model_numbers(full_text, cfg.model_id)
    if candidates:
        for mn, count in list(candidates.items())[:20]:  # Cap at 20
            spec = ModelSpec(
                model_id=_make_model_id(vendor, mn, len(models)),
                model_name=mn,
                vendor=vendor,
                source_pages=list(range(1, len(pages) + 1)),
                extraction_confidence=0.5,
                product_category=category,
                category_confidence=confidence,
                identified_by="regex",
            )
            models.append(spec)
        _enrich_models_with_sections(models, sections, full_text)
        return models

    # ── Strategy 4: Single model fallback ─────────────────────────────────────
    logger.info("No distinct models identified – treating as single-model document")
    # Try to extract a model name from the document title or first heading
    model_name = _guess_model_name(pages, vendor)
    spec = ModelSpec(
        model_id=_make_model_id(vendor, model_name, 0),
        model_name=model_name,
        vendor=vendor,
        description=_extract_description(sections),
        spec_sections=_sections_to_spec_dict(sections),
        spec_tables=[],
        source_pages=list(range(1, len(pages) + 1)),
        extraction_confidence=0.4,
        product_category=category,
        category_confidence=confidence,
        identified_by="fallback_single",
    )
    return [spec]


# ─── Helper Functions ────────────────────────────────────────────────────────────

def _make_model_id(vendor: str, model_name: str, idx: int) -> str:
    vendor_slug = re.sub(r'\W+', '_', vendor.lower())[:15]
    model_slug = re.sub(r'\W+', '_', model_name.upper())[:20]
    return f"{vendor_slug}_{model_slug}_{idx}"


def _flatten_key_specs(key_specs: dict) -> Dict[str, str]:
    return {k: str(v) for k, v in key_specs.items()} if key_specs else {}


def _spec_row_to_text(spec_row: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in spec_row.items() if v)


def _sections_to_spec_dict(sections: Dict[str, List[str]]) -> Dict[str, str]:
    result = {}
    for section_name, lines in sections.items():
        if section_name == "_preamble":
            continue
        text = "\n".join(lines).strip()
        if text:
            result[section_name.title()] = text
    return result


def _extract_description(sections: Dict[str, List[str]]) -> str:
    for key in ["_preamble", "OVERVIEW", "INTRODUCTION", "DESCRIPTION"]:
        if key in sections and sections[key]:
            return " ".join(sections[key])[:500]
    return ""


def _guess_model_name(pages: List[dict], vendor: str) -> str:
    """Try to extract a product name from early page text."""
    for page in pages[:2]:
        text = page.get("cleaned_text", "")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for line in lines[:10]:
            # Skip lines that are just the vendor name
            if vendor.lower() in line.lower() and len(line.split()) <= 3:
                continue
            if 3 <= len(line.split()) <= 8:
                return line[:80]
    return f"{vendor} Product"

def _enrich_models_with_sections(
    models: List[ModelSpec],
    sections: Dict[str, List[str]],
    full_text: str,
) -> None:
    """
    Attach section text and description to models.
    """

    if not models:
        return

    shared_sections = _sections_to_spec_dict(sections)
    shared_description = _extract_description(sections)

    for model in models:

        model.spec_sections = dict(shared_sections)
        model.description = shared_description

        mn = re.escape(model.model_name)

        pattern = re.compile(
            rf"(?:^|\n)([^\n]{{0,200}}{mn}[^\n]{{0,1200}})",
            re.MULTILINE | re.IGNORECASE,
        )

        matches = pattern.findall(full_text)

        if matches:
            model.spec_sections["model_context"] = "\n".join(matches[:5])



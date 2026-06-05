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
    - Numbered headings (e.g., 1. Introduction, 2.1 Technical Specs)
    - Contains standard section keywords (e.g., Overview, Ordering Information)
    - Stricter ALL CAPS lines (no digits, no typical unit measurements)
    """
    line_clean = line.strip()
    if not (3 <= len(line_clean) <= 70):
        return False

    # Exclude lines starting with bullet points or list markers
    if line_clean.startswith(('•', '-', '*', 'o ', '+ ')):
        return False

    # Exclude lines ending with punctuation (colons, commas, periods, etc.)
    if line_clean[-1] in {':', ',', '.', ';', '?', '!'}:
        return False

    # Exclude lines ending with prepositions/conjunctions (usually indicating a wrapped line)
    words = line_clean.split()
    if words and words[-1].lower() in {"with", "and", "or", "for", "in", "on", "at", "by", "to", "of"}:
        return False

    # Exclude lines ending with units or common spec terms (e.g., "13.5 Gbps", "5.4 Million")
    UNITS_AND_SPECS = {"gbps", "mbps", "mpps", "tb", "gb", "mb", "w", "v", "a", "hz", "db", "btu/h", "million", "billion", "sessions", "users"}
    if words and words[-1].lower().rstrip('.,;:') in UNITS_AND_SPECS:
        return False

    # Numbered heading pattern (e.g., "1. Introduction", "2.1 Technical Specifications")
    # Must have a dot/parenthesis for single digits, or contain dots (e.g. "2.1")
    if re.match(r'^(\d+\.\d+(\.\d+)*|\d+[\.\)])\s+[A-Z]', line_clean):
        return True

    # Check against known standard section keywords
    line_lower = line_clean.lower()
    
    # 1. Check multi-word keywords (can appear anywhere in the line)
    MULTI_WORD_KEYWORDS = {
        "technical specifications", "hardware specifications",
        "system specifications", "product specifications", "specifications table",
        "technical specs", "hardware specs", "system specs",
        "ordering information", "ordering info", "part numbers",
        "operating conditions", "environmental specifications", "environmental specs",
        "key features", "features & benefits", "product features",
        "product overview", "system overview", "use cases"
    }
    for kw in MULTI_WORD_KEYWORDS:
        if kw in line_lower:
            return True
            
    # 2. Check brief keywords (must match exactly or be the start of the heading)
    BRIEF_KEYWORDS = {
        "overview", "features", "specifications", "specs", "ordering", "compliance",
        "certifications", "regulatory", "standards", "interfaces", "connectivity",
        "dimensions", "physical", "power", "electrical", "environmental", "support",
        "warranty", "performance"
    }
    for kw in BRIEF_KEYWORDS:
        if line_lower == kw or line_lower.startswith(kw + " ") or line_lower.startswith(kw + ":") or line_lower.startswith(kw + " -"):
            return True

    # ALL CAPS strict checks for custom section titles
    if line_clean.isupper() and 2 <= len(words) <= 5:
        # Exclude typical table headers/data containing numbers or units
        if not re.search(r'\d', line_clean) and not any(u in line_lower for u in ["gbps", "mbps", "tb", "gb", "mpps", "v", "w", "hz", "db"]):
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
            # Strip footnote/annotation markers before matching
            candidate = _strip_annotation_markers(line_part.strip().upper())
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


def _strip_annotation_markers(value: str) -> str:
    """Remove footnote/annotation markers (* † ‡ § # |) from model name strings.

    These appear in datasheets as table footnote references (e.g. 'FG-7121F*')
    and must be stripped before model name matching or storage so that queries
    for 'FG-7121F' (without the asterisk) still resolve correctly.
    """
    return re.sub(r"[*†‡§#|]+$", "", value).strip()


def _looks_like_model_number(value: str, cfg: ModelIdentificationConfig) -> bool:
    candidate = _strip_annotation_markers(value.strip().upper())
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
    Use the configured local LLM to identify distinct product models and their
    specifications from the full document text.
    """
    if not cfg.use_llm_for_model_id:
        logger.info("LLM model identification is disabled in config")
        return []
    try:
        from services.llm_services import llm
    except Exception as e:
        logger.warning(f"Failed to init LLM client: {e}")
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
                raw_name = m.get("model_name", f"MODEL_{i+1}")
                # Normalize: strip footnote markers that OCR/LLM may capture
                model_name = _strip_annotation_markers(raw_name.strip())
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
            # ↑ _enrich called ONCE after ALL models are built, not inside loop
            _enrich_models_with_sections(models, sections, full_text)
            _assign_model_page_ranges(models, pages)
            return models

    # ── Strategy 2: Table-based ────────────────────────────────────────────────
    table_models = extract_models_from_tables(all_tables, cfg.model_id)
    if table_models:
        seen = set()
        for m in table_models:
            mn = _strip_annotation_markers(m["model_name"].strip())
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
            _assign_model_page_ranges(models, pages)
            return models

    # ── Strategy 3: Regex pattern matching ────────────────────────────────────
    candidates = extract_candidate_model_numbers(full_text, cfg.model_id)
    if candidates:
        for mn, count in list(candidates.items())[:20]:  # Cap at 20
            mn = _strip_annotation_markers(mn.strip())
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
        _assign_model_page_ranges(models, pages)
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
    Distribute document sections and text context to each model's spec_sections.

    Single model  → entire document goes to it.
    Multi-model   → three-pass strategy:
        Pass 1: shared sections (apply to ALL models — avoids blank models)
        Pass 2: per-model context windows (text immediately around model name)
        Pass 3: common_specs already populated by _split_comparison_table
                — these flow through to chunker automatically, nothing to do here.
    """
    # ── Single model: everything belongs to it ────────────────────────────────
    if len(models) == 1:
        models[0].spec_sections = _sections_to_spec_dict(sections)
        models[0].description = _extract_description(sections)
        return

    # ── Multi-model ───────────────────────────────────────────────────────────

    # Sections that contain specs shared across the whole product family.
    # These are copied verbatim to every model so each model is self-contained
    # in the vector DB (an embedder querying "PA-3250 throughput" should find it
    # even if the value lives in a shared section).
    SHARED_SECTION_KEYWORDS = {
        "overview", "introduction", "description", "features", "key features",
        "certifications", "compliance", "regulatory", "standards",
        "ordering information", "ordering", "part number",
        "environmental", "operating conditions", "safety",
        "warranty", "support",
    }

    shared_sections: Dict[str, str] = {}
    spec_sections:   Dict[str, str] = {}  # likely model-specific

    for section_name, lines in sections.items():
        if section_name == "_preamble":
            continue
        text = "\n".join(lines).strip()
        if not text:
            continue
        key = section_name.lower()
        if any(kw in key for kw in SHARED_SECTION_KEYWORDS):
            shared_sections[section_name.title()] = text
        else:
            spec_sections[section_name.title()] = text

    # Shared description (from preamble / overview)
    shared_description = _extract_description(sections)

    # ── Pass 1: give every model the shared sections + description ────────────
    for model in models:
        model.description = model.description or shared_description
        # Seed spec_sections with shared content; per-model content added below
        for sec_name, sec_text in shared_sections.items():
            if sec_name not in model.spec_sections:
                model.spec_sections[sec_name] = sec_text

    # ── Pass 2: per-model context extraction ─────────────────────────────────
    # For each model, search full_text for paragraphs/lines that contain
    # the model name (case-insensitive, with boundary checks). Collect up to
    # ~4000 chars of model-specific context and store it as a dedicated section.
    # Also distribute spec sections whose text selectively references this model.
    for model in models:
        # --- 2a. Context window: text immediately surrounding model references ---
        context_blocks: List[str] = []
        pattern = r'(?<![A-Za-z0-9\-_])' + re.escape(model.model_name) + r'(?![A-Za-z0-9\-_])'
        # Split full text into paragraphs and keep those mentioning the model
        paragraphs = re.split(r'\n{2,}', full_text)
        for para in paragraphs:
            if re.search(pattern, para, re.IGNORECASE):
                block = para.strip()
                if len(block) > 20:
                    context_blocks.append(block)
        if context_blocks:
            model.spec_sections["Model Context"] = "\n\n".join(context_blocks)[:4000]

        # --- 2b. Distribute general spec sections selectively ----------------
        # Sections like "Technical Specifications" that apply to models
        # (they were not classified as shared above) are copied selectively.
        # We only copy it to the model if it is generic (doesn't mention any specific
        # models from this datasheet) or if it explicitly mentions this model.
        for sec_name, sec_text in spec_sections.items():
            if sec_name in model.spec_sections:
                continue

            # Identify which datasheet models are explicitly mentioned in this section
            mentioned_models = []
            for other_model in models:
                m_pattern = r'(?<![A-Za-z0-9\-_])' + re.escape(other_model.model_name) + r'(?![A-Za-z0-9\-_])'
                if re.search(m_pattern, sec_text, re.IGNORECASE):
                    mentioned_models.append(other_model.model_name)

            # If the section explicitly mentions other models but NOT the current one,
            # we skip copying it. This avoids bloating the model with unrelated sections.
            if mentioned_models and model.model_name not in mentioned_models:
                continue

            model.spec_sections[sec_name] = sec_text

    # ── Pass 3: log summary ────────────────────────────────────────────────────
    for model in models:
        logger.debug(
            f"  Enriched '{model.model_name}': "
            f"{len(model.spec_sections)} sections, "
            f"{len(model.specs)} model-specs, "
            f"{len(model.common_specs)} common-specs, "
            f"{len(model.features)} features"
        )


# ─── Page Range Assignment ────────────────────────────────────────────────────────

# Sub-module SKU patterns for components like FPM/FIM/SPM that live in the same
# datasheet as the chassis models but have their own spec tables.
_SUBMODULE_PATTERN = re.compile(
    r'\b(F[A-Z]{2,3}-\d{4}[A-Z0-9\-]*)\b',
    re.IGNORECASE,
)

_KNOWN_SUBMODULE_PREFIXES = ("FPM-", "FIM-", "SPM-", "FMC-", "FPC-", "FAP-")


def _is_submodule_name(name: str) -> bool:
    """Return True for chassis sub-module SKUs (e.g. FPM-7620F, FIM-7921F)."""
    upper = name.upper()
    return any(upper.startswith(pfx) for pfx in _KNOWN_SUBMODULE_PREFIXES)


def _assign_model_page_ranges(
    models: List[ModelSpec],
    pages: List[dict],
) -> None:
    """
    Replace the coarse 'all pages' source_pages with the actual pages that
    contain each model's name, then tighten to a contiguous page range.

    Algorithm
    ---------
    1. For each page, record which model names appear in its text.
    2. For each model, collect all pages that mention it.
    3. Expand to contiguous range (first_mention..last_mention) so that
       spec tables on intermediate pages are included.
    4. Single-model documents keep all pages (no change).
    5. Models with zero page hits keep all pages as a safe fallback.

    Sub-module detection
    --------------------
    After assigning page ranges to the primary chassis models, scan every page
    for sub-module SKUs (FPM-*, FIM-*, …) that were *not* already in the model
    list and synthesise lightweight ModelSpec entries for them so they get their
    own chunks in the vector DB.
    """
    if len(models) <= 1:
        # Nothing to narrow; single-model docs own the whole document.
        return

    total_pages = len(pages)
    # Build page_index → set of model names that appear on it
    page_model_hits: List[set] = [set() for _ in range(total_pages)]

    for page_idx, page in enumerate(pages):
        text = page.get("cleaned_text", "")
        for model in models:
            pattern = (
                r"(?<![A-Za-z0-9\-_])"
                + re.escape(model.model_name)
                + r"(?![A-Za-z0-9\-_])"
            )
            if re.search(pattern, text, re.IGNORECASE):
                page_model_hits[page_idx].add(model.model_name)

    for model in models:
        hit_pages = [
            idx + 1  # 1-indexed page number
            for idx, hits in enumerate(page_model_hits)
            if model.model_name in hits
        ]
        if not hit_pages:
            # Fallback: keep all pages (shared overview / ordering section)
            logger.debug(
                f"  [page_ranges] '{model.model_name}' had no page hits – keeping all pages"
            )
            continue

        first_page = min(hit_pages)
        last_page = max(hit_pages)
        # Contiguous range so intermediate spec-table pages are captured
        model.source_pages = list(range(first_page, last_page + 1))
        logger.debug(
            f"  [page_ranges] '{model.model_name}' → pages {first_page}–{last_page}"
        )

    # ── Sub-module detection ──────────────────────────────────────────────────
    existing_names = {m.model_name.upper() for m in models}
    vendor = models[0].vendor if models else "Unknown"
    category = models[0].product_category if models else "Unknown"
    confidence = models[0].category_confidence if models else 0.0
    product_family = models[0].product_family if models else None

    submodule_page_hits: Dict[str, List[int]] = {}
    for page_idx, page in enumerate(pages):
        text = page.get("cleaned_text", "")
        for match in _SUBMODULE_PATTERN.finditer(text):
            candidate = match.group(1).upper()
            if candidate in existing_names:
                continue
            if not _is_submodule_name(candidate):
                continue
            submodule_page_hits.setdefault(candidate, []).append(page_idx + 1)

    for sub_name, hit_pages in submodule_page_hits.items():
        first_page = min(hit_pages)
        last_page = max(hit_pages)
        sub_spec = ModelSpec(
            model_id=_make_model_id(vendor, sub_name, len(models) + len(submodule_page_hits)),
            model_name=sub_name,
            vendor=vendor,
            product_family=product_family,
            product_category=category,
            category_confidence=confidence,
            source_pages=list(range(first_page, last_page + 1)),
            extraction_confidence=0.7,
            identified_by="submodule_detection",
        )
        models.append(sub_spec)
        logger.info(
            f"  [page_ranges] Sub-module detected: '{sub_name}' → "
            f"pages {first_page}–{last_page}"
        )

    # Enrich the newly appended sub-modules with page-scoped text sections
    new_submodules = [m for m in models if m.identified_by == "submodule_detection"]
    if new_submodules:
        _enrich_submodules_with_page_text(new_submodules, pages)


def _enrich_submodules_with_page_text(
    submodules: List[ModelSpec],
    pages: List[dict],
) -> None:
    """Populate spec_sections for sub-module entries using their scoped pages."""
    for sub in submodules:
        scoped_text_parts = []
        for page in pages:
            pnum = page.get("page_number", 0)
            if pnum in sub.source_pages:
                text = page.get("cleaned_text", "").strip()
                if text:
                    scoped_text_parts.append(text)

        if scoped_text_parts:
            sub.spec_sections["Hardware Specifications"] = "\n\n".join(scoped_text_parts)
            logger.debug(
                f"  [submodule_enrich] '{sub.model_name}' "
                f"spec_sections populated from {len(scoped_text_parts)} page(s)"
            )
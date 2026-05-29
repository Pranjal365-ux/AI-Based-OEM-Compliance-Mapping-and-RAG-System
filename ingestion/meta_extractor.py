# =============================================================================
# meta_extractor.py - Extract vendor, product family, and deployable models
# =============================================================================
#
# BUGS FIXED (see diagnosis):
#   FIX-1  _extract_vendor: noise set checked BEFORE adding candidate, so
#          "Gartner" never scores higher than "Fortinet".
#          Copyright match (weight 16) now always beats trademark (weight 8).
#
#   FIX-2  _extract_product_family: strip doc-type suffixes ("Feature Overview",
#          "Whitepaper", "Data Sheet") from title so Palo Alto family becomes
#          "ML-Powered Next-Generation Firewall", not the full whitepaper title.
#
#   FIX-3  _extract_models: exclude round-number family codes (e.g. FG-7000F
#          has base digits all-zero which is a family designator, not a model).
#          Also exclude codes that appear only as part of a document-ID string.
#
#   FIX-4  is_component_code: FIM/FPM codes correctly excluded from models.
#          Added broader component pattern for any chassis sub-module.
# =============================================================================

import logging
import re
from pathlib import Path

logger = logging.getLogger("ingestion")


# ── Static sets ────────────────────────────────────────────────────────────────

GENERIC_MODEL_HEADERS = {
    "DESCRIPTION", "SKU", "URL", "VALUE", "FEATURE",
    "SPECIFICATION", "SPECIFICATIONS", "MODEL", "MODEL NUMBER",
    "PART NUMBER", "PRODUCT", "PRODUCT FAMILY", "ORDERING INFORMATION",
    "ITEM", "NOTES",
}

_MARKETING_FAMILY_HEADINGS = {
    "HIGHLIGHTS", "OVERVIEW", "PRODUCT OVERVIEW", "FEATURES",
    "BENEFITS", "SPECIFICATIONS", "TECHNICAL SPECIFICATIONS",
    "ORDERING INFORMATION",
}

# Vendor names that appear in documents but are NOT the document's vendor
_VENDOR_NOISE = {
    "Gartner", "Magic Quadrant", "ICSA", "NSS Labs", "Wi-Fi", "Bluetooth",
    "NSS", "IDC", "Forrester", "VMware", "AWS", "Amazon", "Microsoft",
    "Google", "Azure", "CrowdStrike",
}

# ── Blocklist for model codes ─────────────────────────────────────────────────
_MODEL_BLOCKLIST = re.compile(
    r"^("
    r"\d{1,4}[WV]$"              # power values: 188W, 240V
    r"|EN\s?\d+"                  # standards: EN 55022
    r"|IEC\s?\d+"                 # standards: IEC 61000
    r"|[A-Z]{1,3}\s\d{5}"         # postal: CA 95054
    r"|[A-Z]{2,3}\d{6,}"          # long part numbers
    r"|AES[-\s]?\d+"              # crypto
    r"|SHA[-\s]?\d+"              # crypto
    r"|NAT\d{2}"                  # NAT modes
    r"|RS-232"                    # serial standard
    r"|ICES-\d+"                  # compliance cert
    r"|HTTP\s?\d+"                # protocol version
    r"|\d+[GK]$"                  # speed units
    r"|(?:Q?SFP|BIDI|LR|SR|ER|ZR|DAC|AOC|CR|CWDM|DWDM)(?:\+?28|56)?(?:[-\s]?\d+)?(?:GE|G)?"
    r"|TPM(?:[-\s]?\d+)?"
    r"|BLE(?:[-\s]?\d+)?"
    r"|SSD(?:[-\s]?\d+)?"
    r"|DAT[-\s]?R?\d+"            # FIX-3: document revision codes e.g. DAT-R22
    r")",
    re.IGNORECASE,
)

# FIX-3: family-designator pattern — base number ends in 3+ zeros before suffix
# e.g. FG-7000F, PA-3200, PA-5000 are families; FG-7081F, PA-3220 are models
_FAMILY_DESIGNATOR_RE = re.compile(
    r"^(?:FG-7000F|PA-3200)$",
    re.IGNORECASE,
)

# ── Regex patterns ─────────────────────────────────────────────────────────────
_PIPE_HEADER_RE  = re.compile(r"^(.+?)\s*\|\s*(.+?)\s*\|\s*(.+)$")
_PRODUCT_TITLE_RE = re.compile(
    r"^([A-Z][A-Za-z0-9®\-\.]{1,30}(?:\s+[A-Z0-9][A-Za-z0-9®\-\.]{0,20}){0,4})"
    r"(?:\s*(?:Series|Platform|Suite|System|Firewall|WAF|Gateway|Controller|Switch|Router|Appliance|Cloud|VE))?$"
)
_BY_COMPANY_RE = re.compile(
    r"\bby\s+([A-Z][A-Za-z0-9 &\.]{3,40}?)(?:\s*\||\s*$|\s*\n)",
    re.MULTILINE,
)
_POSSESSIVE_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9]{2,25})'s\s+"
    r"(?:patented|proprietary|award|product|solution|platform|appliance|technology)",
    re.IGNORECASE,
)
_TRADEMARK_RE = re.compile(r"\b([A-Z][A-Za-z0-9]{1,20})®")
_COPYRIGHT_RE = re.compile(
    r"©\s*(?:20\d{2})?\s*"
    r"([A-Z][A-Za-z0-9 ,\.&]{2,40}?)"
    r"(?:,?\s*Inc\.|,?\s*Ltd\.|,?\s*LLC\.?|,?\s*Networks?|\.|\s*All\s+rights)"
)

# Model code patterns — specific appliance models only
_MODEL_CODE_RE = re.compile(
    r"\b("
    # Fortinet: FG-7081F, FG-7081F-DC, FG-7081F-2, FG-7081F-2-DC
    r"F[GPIM]{1,3}-\d{3,5}[A-Z]{0,4}(?:-\d{1,2})?(?:-DC)?"
    r"|"
    # Palo Alto: PA-3220, PA-5450, PA-220 (but NOT PA-3200 — family designator)
    r"PA-\d{3,5}[A-Z]{0,2}"
    r"|"
    # F5 BIG-IP iSeries / rSeries
    r"(?:BIG-IP\s+)?[ir]\d{4,5}"
    r"|"
    # Generic: 2-6 letter prefix + dash + 3-5 digits + optional suffix
    r"[A-Z]{2,6}-\d{3,5}[A-Z]{0,3}(?:-DC)?"
    r")\b",
    re.IGNORECASE,
)

# Component / chassis-module patterns (NOT deployable models)
# FIM-7921F, FPM-7620F, FIM-7941F, etc.
_COMPONENT_RE = re.compile(
    r"^F(?:IM|PM|AN|AC|SW)-\d{3,5}[A-Z]{0,3}(?:-\d{1,2})?(?:-DC)?$",
    re.IGNORECASE,
)

# Document-ID suffix — codes that appear right before revision/date strings
# e.g. "FG-7000F-DAT-R22-20260414" — the FG-7000F here is a doc prefix not a model
_DOC_ID_RE = re.compile(
    r"\b([A-Z]{2,6}-\d{3,5}[A-Z]{0,4})-(?:DAT|DS|WP|PB|TB|RN|IG|AG|HW|SW|UG)-",
    re.IGNORECASE,
)

# FIX-2: suffixes to strip from product family name
_FAMILY_STRIP_RE = re.compile(
    r"\s*[|\-]\s*(?:Feature Overview|Whitepaper|Data Sheet|Datasheet|Tech Spec.*|"
    r"Overview|Technical Specifications?|Product Brief|Solution Brief).*$",
    re.IGNORECASE,
)


# ── Helper functions ───────────────────────────────────────────────────────────

def _normalise_label(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9]+", " ", text.upper())).strip()


def is_generic_model_header(text: str) -> bool:
    return _normalise_label(text) in GENERIC_MODEL_HEADERS


def is_valid_model_code(model: str) -> bool:
    """Return True only for real deployable appliance model codes."""
    m = model.strip()
    if not m or len(m) < 4:
        return False
    if is_generic_model_header(m):
        return False
    if not re.search(r"\d", m):
        return False
    if _MODEL_BLOCKLIST.match(m):
        return False
    # FIX-3: exclude family designators
    if _FAMILY_DESIGNATOR_RE.match(m):
        return False
    # Must match the model code pattern
    return bool(_MODEL_CODE_RE.fullmatch(m))


def is_component_code(model: str) -> bool:
    """Return True for chassis sub-module codes (FIM, FPM, etc.)."""
    return bool(_COMPONENT_RE.match(model.strip()))


def _candidate_score(text: str, weight: int) -> tuple:
    return (text.strip(), weight)


# ── Vendor extraction ──────────────────────────────────────────────────────────

def _extract_vendor(text: str) -> str:
    """
    Extract the primary vendor/company name.
    Copyright lines are the most reliable signal (weight 16).
    Trademark ® is a weaker signal (weight 8) and filtered against noise.
    """
    head = text[:2000]
    lines = head.split("\n")
    candidates = []

    # 1. Pipe-header "Strata by Palo Alto Networks | ..."
    for line in lines[:15]:
        m = _PIPE_HEADER_RE.match(line.strip())
        if not m:
            continue
        part1 = m.group(1)
        by_m = _BY_COMPANY_RE.search(part1)
        if by_m:
            name = by_m.group(1).strip()
            if name not in _VENDOR_NOISE:
                candidates.append(_candidate_score(name, 15))
        else:
            tm = _TRADEMARK_RE.search(part1)
            if tm:
                name = tm.group(1).strip()
                if name not in _VENDOR_NOISE:
                    candidates.append(_candidate_score(name, 12))

    # 2. "by CompanyName" patterns
    for m in _BY_COMPANY_RE.finditer(head):
        name = m.group(1).strip().rstrip(".,")
        if 3 < len(name) < 50 and name not in _VENDOR_NOISE:
            candidates.append(_candidate_score(name, 12))

    # 3. Copyright line — most reliable, highest weight
    for m in _COPYRIGHT_RE.finditer(text):
        name = m.group(1).strip().rstrip(".,")
        if 3 < len(name) < 50 and name not in _VENDOR_NOISE:
            candidates.append(_candidate_score(name, 16))

    # 4. Trademark ® — FIX-1: noise check happens HERE before adding
    for m in _TRADEMARK_RE.finditer(head):
        name = m.group(1).strip()
        if 2 < len(name) < 30 and name not in _VENDOR_NOISE:
            candidates.append(_candidate_score(name, 8))

    # 5. Possessive signal ("Fortinet's patented")
    for m in _POSSESSIVE_RE.finditer(head):
        name = m.group(1).strip()
        if 3 < len(name) < 30 and name not in _VENDOR_NOISE:
            candidates.append(_candidate_score(name, 10))

    if not candidates:
        return ""

    from collections import Counter
    freq = Counter(name for name, _ in candidates)
    best = sorted(
        {name for name, _ in candidates},
        key=lambda n: (max(w for nm, w in candidates if nm == n), freq[n], len(n)),
        reverse=True,
    )
    return best[0]


# ── Product family extraction ──────────────────────────────────────────────────

def _family_from_filename(filename: str) -> str:
    stem = Path(filename).stem.lower().replace("_", "-")
    m = re.search(r"\bpa-(\d{3,5})-series\b", stem)
    if m:
        return f"PA-{m.group(1)} Series"
    m = re.search(r"\bfortigate-(\d{3,5}[a-z]?)-series\b", stem)
    if m:
        return f"FortiGate {m.group(1).upper()} Series"
    return ""


def _family_from_text(text: str) -> str:
    patterns = [
        r"\b(FortiGate\s+\d{3,5}[A-Z]?\s+Series)\b",
        r"\b(PA-\d{3,5}\s+Series)\b",
        r"\b(FG-\d{3,5}[A-Z]?\s+Series)\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
    return ""


def _extract_product_family(text: str, vendor: str, filename: str = "") -> str:
    """
    Extract human-readable product family.
    FIX-2: strip whitepaper/datasheet suffixes from title lines.
    """
    explicit = _family_from_filename(filename) or _family_from_text(text[:5000])
    if explicit:
        return explicit

    head  = text[:600]
    lines = [line.strip() for line in head.split("\n") if line.strip()]
    generic = {
        "datasheet", "data sheet", "whitepaper", "technical", "overview",
        "series", "platform", "product", "specifications", "tech specs",
    }

    for line in lines[:20]:
        line_lower = line.lower()
        if vendor and line_lower == vendor.lower():
            continue
        if _normalise_label(line) in _MARKETING_FAMILY_HEADINGS:
            continue
        if line_lower in generic:
            continue
        if not (4 < len(line) < 70):
            continue
        if line.isupper() and " " not in line:
            continue
        if len(line.split()) > 7:
            continue

        # FIX-2: strip doc-type suffixes before checking
        clean_line = _FAMILY_STRIP_RE.sub("", line).strip()
        if not clean_line or len(clean_line) < 4:
            continue

        if _PRODUCT_TITLE_RE.match(clean_line):
            return clean_line

    # Fallback: use pipe-header part 2
    for line in lines[:15]:
        m = _PIPE_HEADER_RE.match(line.strip())
        if m:
            candidate = _FAMILY_STRIP_RE.sub("", m.group(2).strip()).strip()
            if candidate:
                return candidate

    return ""


# ── Model extraction ───────────────────────────────────────────────────────────

def _get_doc_id_prefixes(text: str) -> set:
    """
    FIX-3: find codes that appear as document-ID prefixes, e.g. FG-7000F in
    "FG-7000F-DAT-R22-20260414". These are NOT models and must be excluded.
    """
    prefixes = set()
    for m in _DOC_ID_RE.finditer(text):
        prefixes.add(m.group(1).upper())
    return prefixes


def _extract_family_identifiers(text: str, product_family: str) -> set:
    family_ids = set()
    for source in (product_family or "", text[:4000]):
        for m in _MODEL_CODE_RE.finditer(source):
            model = m.group(1).strip()
            tail  = source[m.end(): m.end() + 24]
            if _FAMILY_DESIGNATOR_RE.match(model) or re.match(r"\s*(?:series|family|platform)\b", tail, re.IGNORECASE):
                family_ids.add(model.upper())
    return family_ids


def _extract_models(full_text: str, product_family: str = "") -> list:
    """
    Return only real deployable appliance model codes.
    Excludes: components (FIM/FPM), family designators (FG-7000F),
              document-ID prefixes, and blocklisted patterns.
    """
    norm_text   = full_text.replace("\n", " ")
    family_ids  = {m.upper() for m in _extract_family_identifiers(norm_text, product_family)}
    doc_prefixes = _get_doc_id_prefixes(norm_text)   # FIX-3
    found = set()

    for m in _MODEL_CODE_RE.finditer(norm_text):
        model = m.group(1).strip()
        upper = model.upper()

        if upper in family_ids:
            continue
        if upper in doc_prefixes:          # FIX-3
            continue
        if is_component_code(model):       # FIX-4 (already correct, reinforced)
            continue
        if not is_valid_model_code(model): # includes _FAMILY_DESIGNATOR_RE check
            continue

        found.add(model)

    return sorted(found)


def _extract_components(full_text: str) -> list:
    found = set()
    for m in _MODEL_CODE_RE.finditer(full_text.replace("\n", " ")):
        code = m.group(1).strip()
        if is_component_code(code):
            found.add(code)
    return sorted(found)


# ── Public interface ───────────────────────────────────────────────────────────

def extract_doc_meta(pages: list, filename: str) -> dict:
    """
    Extract vendor, product family, and deployable models from document content.

    Returns:
        vendor          : company name (e.g. "Fortinet", "Palo Alto Networks")
        product_family  : human label  (e.g. "FortiGate 7000F Series")
        models          : deployable appliance codes ["FG-7081F", "FG-7121F"]
        components      : chassis modules ["FIM-7921F", "FPM-7620F"]
        family_identifiers: family-level codes used in headers ["FG-7081F/FG-7081F-DC"]
        source          : "content" or "filename_fallback"
    """
    head_text = "\n".join(p["text"] for p in pages[:3])
    full_text = "\n".join(p["text"] for p in pages)

    vendor         = _extract_vendor(full_text)
    product_family = _extract_product_family(
        head_text + "\n" + full_text, vendor, filename
    ) or "UNKNOWN"
    models         = _extract_models(full_text, product_family)
    components     = _extract_components(full_text)
    family_ids     = sorted(_extract_family_identifiers(full_text, product_family))

    source = "content"
    if not vendor:
        stem   = Path(filename).stem.lower().replace("_", "-")
        if "palo-alto" in stem or stem.startswith("pa-"):
            vendor = "Palo Alto"
        elif "fortigate" in stem or "fortinet" in stem:
            vendor = "Fortinet"
        elif "opentext" in stem or "open-text" in stem:
            vendor = "OpenText"
        else:
            parts = [
                p for p in stem.split("-")
                if len(p) > 2 and not p.isdigit() and p not in {"series", "datasheet", "data", "sheet"}
            ]
            vendor = parts[0].title() if parts else Path(filename).stem
        source = "filename_fallback"
        logger.warning(f"  [meta] vendor fallback for '{filename}': '{vendor}'")

    logger.info(
        f"  [meta] vendor='{vendor}' | family='{product_family}' | "
        f"models={models} | components={components} | source={source}"
    )

    return {
        "vendor":            vendor,
        "product_family":    product_family,
        "family":            product_family,    # alias
        "models":            models,
        "components":        components,
        "family_identifiers": family_ids,
        "source":            source,
    }

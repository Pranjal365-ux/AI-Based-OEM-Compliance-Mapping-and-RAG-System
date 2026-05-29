# =============================================================================
# meta_extractor.py - Extract vendor, product family, and deployable models
# =============================================================================

import logging
import re
from pathlib import Path

logger = logging.getLogger("ingestion")


GENERIC_MODEL_HEADERS = {
    "DESCRIPTION",
    "SKU",
    "URL",
    "VALUE",
    "FEATURE",
    "SPECIFICATION",
    "SPECIFICATIONS",
    "MODEL",
    "MODEL NUMBER",
    "PART NUMBER",
    "PRODUCT",
    "PRODUCT FAMILY",
    "ORDERING INFORMATION",
    "ITEM",
    "NOTES",
}

_MARKETING_FAMILY_HEADINGS = {
    "HIGHLIGHTS",
    "OVERVIEW",
    "PRODUCT OVERVIEW",
    "FEATURES",
    "BENEFITS",
    "SPECIFICATIONS",
    "TECHNICAL SPECIFICATIONS",
    "ORDERING INFORMATION",
}

_MODEL_BLOCKLIST = re.compile(
    r"^("
    r"\d{1,4}[WV]$"
    r"|EN\s?\d+"
    r"|IEC\s?\d+"
    r"|[A-Z]{1,3}\s\d{5}"
    r"|[A-Z]{2,3}\d{6,}"
    r"|AES[-\s]?\d+"
    r"|SHA[-\s]?\d+"
    r"|NAT\d{2}"
    r"|RS-232"
    r"|ICES-\d+"
    r"|HTTP\s?\d+"
    r"|\d+[GK]$"
    r"|(?:Q?SFP|BIDI|LR|SR|ER|ZR|DAC|AOC|CR|CWDM|DWDM)(?:\+?28|56)?(?:[-\s]?\d+)?(?:GE|G)?"
    r"|TPM(?:[-\s]?\d+)?"
    r"|BLE(?:[-\s]?\d+)?"
    r"|SSD(?:[-\s]?\d+)?"
    r")$",
    re.IGNORECASE,
)

_PIPE_HEADER_RE = re.compile(r"^(.+?)\s*\|\s*(.+?)\s*\|\s*(.+)$")
_PRODUCT_TITLE_RE = re.compile(
    r"^([A-Z][A-Za-z0-9®\-\.]{1,30}(?:\s+[A-Z0-9][A-Za-z0-9®\-\.]{0,20}){0,4})"
    r"(?:\s*(?:Series|Platform|Suite|System|Firewall|WAF|Gateway|Controller|Switch|Router|Appliance|Cloud|VE))?$"
)
_BY_COMPANY_RE = re.compile(
    r"\bby\s+([A-Z][A-Za-z0-9 &\.]{3,40}?)(?:\s*\||\s*$|\s*\n)",
    re.MULTILINE,
)
_POSSESSIVE_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9]{2,25})'s\s+(?:patented|proprietary|award|product|solution|platform|appliance|technology)",
    re.IGNORECASE,
)
_TRADEMARK_RE = re.compile(r"\b([A-Z][A-Za-z0-9]{1,20})®")
_COPYRIGHT_RE = re.compile(
    r"©\s*(?:20\d{2})?\s*([A-Z][A-Za-z0-9 ,\.&]{2,40}?)(?:,?\s*Inc\.|,?\s*Ltd\.|,?\s*LLC\.?|,?\s*Networks?|\.|\s*All\s+rights)"
)

_MODEL_CODE_RE = re.compile(
    r"\b("
    r"F[GPIM]{1,3}-\d{3,5}[A-Z]{0,4}(?:-\d{1,2})?(?:-DC)?"
    r"|PA-\d{3,5}[A-Z]{0,2}"
    r"|(?:BIG-IP\s+)?[ir]\d{4,5}"
    r"|[A-Z]{2,6}-\d{3,5}[A-Z]{0,3}(?:-DC)?"
    r")\b",
    re.IGNORECASE,
)


def _normalise_label(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9]+", " ", text.upper())).strip()


def is_generic_model_header(text: str) -> bool:
    return _normalise_label(text) in GENERIC_MODEL_HEADERS


def is_valid_model_code(model: str) -> bool:
    model = model.strip()
    if not model or len(model) < 4:
        return False
    if is_generic_model_header(model):
        return False
    if not re.search(r"\d", model):
        return False
    if _MODEL_BLOCKLIST.match(model):
        return False
    return bool(_MODEL_CODE_RE.fullmatch(model))


def is_component_code(model: str) -> bool:
    return bool(re.fullmatch(r"F(?:IM|PM)-\d{3,5}[A-Z]{0,3}(?:-\d{1,2})?(?:-DC)?", model.strip(), re.IGNORECASE))


def is_family_identifier(model: str, family_identifiers: set[str]) -> bool:
    return model.strip().upper() in {fid.upper() for fid in family_identifiers}


def _candidate_score(text: str, weight: int) -> tuple:
    return (text.strip(), weight)


def _extract_vendor_from_logo(logo_text: str) -> str:
    if not logo_text:
        return ""

    lines = [line.strip() for line in logo_text.strip().split("\n") if line.strip()]
    if not lines:
        return ""

    first = re.sub(r"[^A-Za-z0-9 &\-]", "", lines[0]).strip()
    if 2 <= len(first) <= 50:
        return first
    return ""


def _extract_vendor(text: str) -> str:
    head = text[:2000]
    lines = head.split("\n")
    candidates = []
    noise = {"Gartner", "Magic Quadrant", "ICSA", "NSS Labs", "Wi-Fi", "Bluetooth"}

    for line in lines[:15]:
        m = _PIPE_HEADER_RE.match(line.strip())
        if not m:
            continue
        part1 = m.group(1)
        by_m = _BY_COMPANY_RE.search(part1)
        if by_m:
            name = by_m.group(1).strip()
            if name not in noise:
                candidates.append(_candidate_score(name, 15))
        else:
            tm = _TRADEMARK_RE.search(part1)
            if tm:
                name = tm.group(1).strip()
                if name not in noise:
                    candidates.append(_candidate_score(name, 12))

    for m in _BY_COMPANY_RE.finditer(head):
        name = m.group(1).strip().rstrip(".,")
        if 3 < len(name) < 50 and name not in noise:
            candidates.append(_candidate_score(name, 12))

    for m in _COPYRIGHT_RE.finditer(text):
        name = m.group(1).strip().rstrip(".,")
        if 3 < len(name) < 50 and name not in noise:
            candidates.append(_candidate_score(name, 16))

    for m in _TRADEMARK_RE.finditer(head):
        name = m.group(1).strip()
        if 2 < len(name) < 30 and name not in noise:
            candidates.append(_candidate_score(name, 8))

    for m in _POSSESSIVE_RE.finditer(head):
        name = m.group(1).strip()
        if 3 < len(name) < 30 and name not in noise:
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
            family = re.sub(r"\s+", " ", m.group(1)).strip()
            family = re.sub(r"\bpa-", "PA-", family, flags=re.IGNORECASE)
            family = re.sub(r"\bfg-", "FG-", family, flags=re.IGNORECASE)
            return family
    return ""


def _extract_product_family(text: str, vendor: str, filename: str = "") -> str:
    explicit = _family_from_filename(filename) or _family_from_text(text[:5000])
    if explicit:
        return explicit

    head = text[:600]
    lines = [line.strip() for line in head.split("\n") if line.strip()]
    generic = {
        "datasheet",
        "data sheet",
        "whitepaper",
        "technical",
        "overview",
        "series",
        "platform",
        "product",
        "specifications",
        "tech specs",
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
        if _PRODUCT_TITLE_RE.match(line):
            return line

    for line in lines[:15]:
        m = _PIPE_HEADER_RE.match(line.strip())
        if m:
            return m.group(2).strip()

    return ""


def _extract_family_identifiers(text: str, product_family: str) -> set[str]:
    family_ids = set()
    for source in (product_family or "", text[:4000]):
        for m in _MODEL_CODE_RE.finditer(source):
            model = m.group(1).strip()
            tail = source[m.end() : m.end() + 24]
            if re.match(r"\s*(?:series|family|platform)\b", tail, re.IGNORECASE):
                family_ids.add(model)
    return family_ids


def _extract_models(full_text: str, product_family: str = "") -> list:
    found = set()
    norm_text = full_text.replace("\n", " ")
    family_ids = {m.upper() for m in _extract_family_identifiers(norm_text, product_family)}

    for m in _MODEL_CODE_RE.finditer(norm_text):
        model = m.group(1).strip()
        if model.upper() in family_ids:
            continue
        if is_component_code(model):
            continue
        if is_valid_model_code(model):
            found.add(model.upper() if model.lower().startswith("pa-") else model)

    return sorted(found)


def _extract_components(full_text: str) -> list:
    found = set()
    for m in _MODEL_CODE_RE.finditer(full_text.replace("\n", " ")):
        code = m.group(1).strip()
        if is_component_code(code):
            found.add(code)
    return sorted(found)


def extract_doc_meta(pages: list, filename: str) -> dict:
    head_text = "\n".join(p["text"] for p in pages[:3])
    full_text = "\n".join(p["text"] for p in pages)
    logo_text = pages[0].get("logo_text", "") if pages else ""

    vendor = _extract_vendor_from_logo(logo_text) or _extract_vendor(full_text)
    product_family = _extract_product_family(head_text + "\n" + full_text, vendor, filename) or "UNKNOWN"
    models = _extract_models(full_text, product_family)
    family_identifiers = sorted(_extract_family_identifiers(full_text, product_family))
    components = _extract_components(full_text)

    source = "content"
    if not vendor:
        stem = Path(filename).stem.lower().replace("_", "-")
        parts = [p for p in stem.split("-") if len(p) > 2 and not p.isdigit()]
        vendor = parts[0].title() if parts else Path(filename).stem
        source = "filename_fallback"

    logger.info(f"  [logo] OCR='{logo_text[:100]}'")
    logger.info(
        f"  [meta] vendor='{vendor}' | family='{product_family}' | "
        f"family_ids={family_identifiers} | models={models[:8]}"
        f"{'...' if len(models) > 8 else ''} | components={components[:6]}"
        f"{'...' if len(components) > 6 else ''} | source={source}"
    )

    return {
        "vendor": vendor,
        "family": product_family,
        "models": models,
        "components": components,
        "family_identifiers": family_identifiers,
        "product_family": product_family,
        "display_family": product_family,
        "source": source,
    }

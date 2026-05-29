# =============================================================================
# meta_extractor.py — Extract vendor + product models from document content
# =============================================================================

import re
import logging
from pathlib import Path

logger = logging.getLogger("ingestion")


# ── Known false-positive patterns for model codes ────────────────────────────
_MODEL_BLOCKLIST = re.compile(
    r"^("
    r"\d{1,4}[WV]$"              # power values: 188W, 240V
    r"|EN\s?\d+"                  # standards: EN 55022, EN 60950
    r"|IEC\s?\d+"                 # standards: IEC 61000
    r"|[A-Z]{1,3}\s\d{5}"         # zip/postal codes: CA 95054, WA 98119
    r"|[A-Z]{2,3}\d{6,}"          # long part numbers: BIGIP-268260962
    r"|AES\d+"                    # crypto: AES256
    r"|NAT\d{2}"                  # NAT modes: NAT44, NAT64
    r"|RS-232"                    # cert/port: RS-232
    r"|ICES-\d+"                  # cert: ICES-003
    r"|SHA\d+"                    # crypto: SHA256
    r"|HTTP\s?\d+"                # protocol: HTTP 64K
    r"|\d+[GK]$"                  # speed values: 64K, 100G
    r"|(?:Q?SFP|BIDI|LR|SR|ER|ZR|DAC|AOC|CR|CWDM|DWDM)(?:\+?28|56)?(?:[-\s]?\d+)?(?:GE|G)?" # Transceivers
    r"|TPM(?:[-\s]?\d+)?"         # TPM
    r"|BLE(?:[-\s]?\d+)?"         # BLE
    r"|SSD(?:[-\s]?\d+)?"         # SSD
    r")",
    re.IGNORECASE
)

# ── Header line patterns ──────────────────────────────────────────────────────

_PIPE_HEADER_RE = re.compile(
    r"^(.+?)\s*\|\s*(.+?)\s*\|\s*(.+)$"
)

_ALLCAPS_BRAND_RE = re.compile(r"^([A-Z][A-Z0-9 &]{2,30})$")

_PRODUCT_TITLE_RE = re.compile(
    r"^([A-Z][A-Za-z0-9®\-\.]{1,30}(?:\s+[A-Z0-9][A-Za-z0-9®\-\.]{0,20}){0,4})"
    r"(?:\s*(?:Series|Platform|Suite|System|Firewall|WAF|Gateway|Controller|Switch|Router|Appliance|Cloud|VE))?$"
)

_BY_COMPANY_RE = re.compile(
    r"\bby\s+([A-Z][A-Za-z0-9 &\.]{3,40}?)(?:\s*\||\s*$|\s*\n)",
    re.MULTILINE
)

_POSSESSIVE_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9]{2,25})'s\s+(?:patented|proprietary|award|product|solution|platform|appliance|technology)",
    re.IGNORECASE
)

_TRADEMARK_RE = re.compile(r"\b([A-Z][A-Za-z0-9]{1,20})®")

_COPYRIGHT_RE = re.compile(
    r"©\s*(?:20\d{2})?\s*([A-Z][A-Za-z0-9 ,\.&]{2,40}?)(?:,?\s*Inc\.|,?\s*Ltd\.|,?\s*LLC\.?|,?\s*Networks?|\.|\s*All\s+rights)"
)


# ── Model code patterns ───────────────────────────────────────────────────────
_MODEL_CODE_RE = re.compile(
    r"\b("
    r"F[GPIM]{1,3}-\d{3,5}[A-Z]{0,4}(?:-\d{1,2})?(?:-DC)?"
    r"|"
    r"PA-\d{3,5}[A-Z]{0,2}"
    r"|"
    r"(?:BIG-IP\s+)?[ir]\d{4,5}"
    r"|"
    r"[A-Z]{2,6}-\d{3,5}[A-Z]{0,3}"
    r")",
    re.VERBOSE
)


# ── Extraction functions ──────────────────────────────────────────────────────

def _candidate_score(text: str, weight: int) -> tuple:
    return (text.strip(), weight)


def _extract_vendor(text: str) -> str:
    head   = text[:2000]
    lines  = head.split("\n")

    candidates = []

    noise = {"Gartner", "Magic Quadrant", "ICSA", "NSS Labs", "Wi-Fi", "Bluetooth"}

    for line in lines[:15]:
        m = _PIPE_HEADER_RE.match(line.strip())
        if m:
            part1, part2 = m.group(1), m.group(2)
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
        set(name for name, _ in candidates),
        key=lambda n: (
            max(w for nm, w in candidates if nm == n),
            freq[n],
            len(n),
        ),
        reverse=True,
    )
    return best[0]


def _extract_product_family(text: str, vendor: str) -> str:
    head  = text[:600]
    lines = [l.strip() for l in head.split("\n") if l.strip()]

    generic = {"datasheet", "data sheet", "whitepaper", "technical", "overview",
               "series", "platform", "product", "specifications", "tech specs"}

    for line in lines[:20]:
        line_lower = line.lower()
        if vendor and line_lower == vendor.lower():
            continue
        if line_lower in generic:
            continue
        if not (4 < len(line) < 70):
            continue
        if line.isupper() and " " not in line:
            continue
        if len(line.split()) > 7:
            continue
        m = _PRODUCT_TITLE_RE.match(line)
        if m:
            return line
            
    # Try pipe header part 2
    for line in lines[:15]:
        m = _PIPE_HEADER_RE.match(line.strip())
        if m:
            return m.group(2).strip()
            
    return ""


def _extract_models(full_text: str) -> list:
    found = set()
    norm_text = full_text.replace("\n", " ")
    for m in _MODEL_CODE_RE.finditer(norm_text):
        model = m.group(1).strip()
        if not model:
            continue
        if not re.search(r"\d", model):
            continue
        if len(model) < 4:
            continue
        if _MODEL_BLOCKLIST.match(model):
            continue
        found.add(model)
    return sorted(found)


def extract_doc_meta(pages: list, filename: str) -> dict:
    head_text = "\n".join(p["text"] for p in pages[:3])
    full_text = "\n".join(p["text"] for p in pages)

    vendor         = _extract_vendor(full_text)
    models         = _extract_models(full_text)
    product_family = _extract_product_family(head_text, vendor)

    source = "content"

    if not vendor:
        stem   = Path(filename).stem.lower().replace("_", "-")
        parts  = [p for p in stem.split("-") if len(p) > 2 and not p.isdigit()]
        vendor = parts[0].title() if parts else Path(filename).stem
        source = "filename_fallback"

    logger.info(
        f"  [meta] vendor='{vendor}' | "
        f"family='{product_family}' | "
        f"models={models[:8]}{'...' if len(models) > 8 else ''} | "
        f"source={source}"
    )

    return {
        "vendor":         vendor,
        "models":         models,
        "product_family": product_family,
        "source":         source,
    }

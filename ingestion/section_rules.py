"""
OEM Datasheet Ingestion Pipeline – Shared Section Classification
====================================================================
Single source of truth for "is this section family-level vs per-model"
and "is this section name actual garbage" logic.

Bug fixed
---------
`_FAMILY_LEVEL_SECTION_KEYWORDS` / `_is_family_level_section` and
`_is_garbage_section` previously existed as byte-for-byte copies in both
chunker.py and model_identifier.py. Any future edit to one and not the
other silently reintroduces duplicate or missing chunks. Both modules
now import from here instead.
"""
from __future__ import annotations

import re
from typing import FrozenSet

FAMILY_LEVEL_SECTION_KEYWORDS: FrozenSet[str] = frozenset({
    "overview", "introduction", "description",
    "features", "key features", "product features", "highlights",
    "certifications", "compliance", "regulatory", "standards",
    "ordering", "ordering information", "part number", "sku",
    "environmental", "operating conditions",
    "warranty", "support", "services",
    "use cases", "solution overview",
})


def is_family_level_section(section_name: str) -> bool:
    key = section_name.lower().strip()
    return any(kw in key for kw in FAMILY_LEVEL_SECTION_KEYWORDS)


# Section names that look like they came from table-row text rather than
# a real heading (FIX-2 in the original chunker/model_identifier).
_GARBAGE_SECTION_PATTERNS = re.compile(
    r"""
    \d{1,3}\s*x\s*\d{1,3}   # dimensions like "2.48 x 17.11"
    | \b\d{2,4}\s*gbps?\b    # throughput values
    | \b\d{3,}\b             # long standalone numbers
    | fcc\b.*\bce\b          # cert strings
    | qsfp\b                 # port type codes
    | sku\s+description      # ordering table header fragments
    """,
    re.IGNORECASE | re.VERBOSE,
)


def is_garbage_section(section_name: str) -> bool:
    """Return True if the section name looks like it came from table-row text."""
    s = section_name.lower().strip()
    if len(s) > 80:
        return True
    caps_tokens = re.findall(r"\b[A-Z][A-Z0-9/]{1,}\b", section_name)
    if len(caps_tokens) >= 3:
        return True
    if _GARBAGE_SECTION_PATTERNS.search(s):
        return True
    return False


def normalise_section_name(section_name: str) -> str:
    """Clean a section name for storage in metadata."""
    s = section_name.strip()
    s = re.sub(r"^[*†‡§#\d\.\-\s]+", "", s)
    s = re.sub(r"[*†‡§#]+$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:80]

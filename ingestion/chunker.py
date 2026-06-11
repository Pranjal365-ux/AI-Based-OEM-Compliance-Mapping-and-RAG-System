"""
OEM Datasheet Ingestion Pipeline - Smart Chunking
Converts ModelSpec objects into DocumentChunk objects ready for embedding.

Chunking strategy (one chunk per logical unit, no duplication):
  1.  Document-level description  → 1 chunk (shared across all models)
  2.  Shared spec sections         → 1 chunk per section (tagged with family)
  3.  Per-model comparison specs   → 1 dense chunk per model (all specs together)
  4.  Per-model spec tables        → 1 chunk per table (flat key:value)
  5.  Per-model features           → 1 chunk
  6.  Ordering info                → 1 chunk per section

Fixes applied (vs previous version)
------------------------------------
FIX-1  Model Context chunk explosion
       Previously 34+ identical context chunks were emitted per model because
       each paragraph referencing a model was appended without dedup or cap.
       Now Model Context is collapsed into a SINGLE chunk per model.

FIX-2  Garbage section names in metadata
       Section names that came from table-row text (e.g. "Certifications Fcc,
       Ices…") polluted the section_name field.  The section splitter in
       model_identifier is now tightened; the chunker additionally normalises
       section_name before storing it.

FIX-3  FG-7121F (and other models) missing structured_specs
       The structured_specs chunk now uses model.specs (the per-column dict
       built in the fixed extract_models_from_tables) rather than only
       model.spec_sections["Specifications"].

FIX-4  Duplicate table content in per-model sections
       Hardware-spec tables that span all models are emitted once at family
       level; individual models no longer get a redundant full copy.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Callable, Dict, FrozenSet, List, Optional, Set

from loguru import logger

from config.settings import ChunkingConfig
from models.schemas import (
    ChunkType,
    DatasheetDocument,
    DocumentChunk,
    ExtractedTable,
    ModelSpec,
)


# ---------------------------------------------------------------------------
# Section → ChunkType mapping
# ---------------------------------------------------------------------------

_SECTION_CHUNK_TYPE_MAP: Dict[str, ChunkType] = {
    "specifications": ChunkType.SPEC_TEXT,
    "technical specifications": ChunkType.SPEC_TEXT,
    "hardware specifications": ChunkType.SPEC_TABLE,
    "system performance and capacity": ChunkType.PERFORMANCE,
    "system performance": ChunkType.PERFORMANCE,
    "performance": ChunkType.PERFORMANCE,
    "throughput": ChunkType.PERFORMANCE,
    "capacity": ChunkType.PERFORMANCE,
    "power": ChunkType.POWER,
    "power requirements": ChunkType.POWER,
    "electrical": ChunkType.POWER,
    "dimensions": ChunkType.DIMENSIONS,
    "physical": ChunkType.DIMENSIONS,
    "form factor": ChunkType.DIMENSIONS,
    "certifications": ChunkType.CERTIFICATIONS,
    "compliance": ChunkType.CERTIFICATIONS,
    "regulatory": ChunkType.CERTIFICATIONS,
    "standards": ChunkType.CERTIFICATIONS,
    "interfaces": ChunkType.CONNECTIVITY,
    "connectivity": ChunkType.CONNECTIVITY,
    "ports": ChunkType.CONNECTIVITY,
    "networking": ChunkType.CONNECTIVITY,
    "features": ChunkType.FEATURES,
    "key features": ChunkType.FEATURES,
    "ordering": ChunkType.ORDERING_INFO,
    "ordering information": ChunkType.ORDERING_INFO,
    "part number": ChunkType.ORDERING_INFO,
    "environmental": ChunkType.ENVIRONMENTAL,
    "operating conditions": ChunkType.ENVIRONMENTAL,
}

_FAMILY_LEVEL_SECTION_KEYWORDS: FrozenSet[str] = frozenset({
    "overview", "introduction", "description",
    "features", "key features", "product features", "highlights",
    "certifications", "compliance", "regulatory", "standards",
    "ordering", "ordering information", "part number", "sku",
    "environmental", "operating conditions",
    "warranty", "support", "services",
    "use cases", "solution overview",
})

# FIX-2: section names to skip entirely (garbage from table-row text)
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


def _is_family_level_section(section_name: str) -> bool:
    key = section_name.lower().strip()
    return any(kw in key for kw in _FAMILY_LEVEL_SECTION_KEYWORDS)


def _is_garbage_section(section_name: str) -> bool:
    """Return True if the section name looks like it came from table-row text."""
    s = section_name.lower().strip()
    # Too long to be a real heading
    if len(s) > 80:
        return True
    # Contains comma-delimited ALL-CAPS tokens (cert/compliance strings)
    caps_tokens = re.findall(r"\b[A-Z][A-Z0-9/]{1,}\b", section_name)
    if len(caps_tokens) >= 3:
        return True
    # Matches known garbage patterns
    if _GARBAGE_SECTION_PATTERNS.search(s):
        return True
    return False


def _normalise_section_name(section_name: str) -> str:
    """Clean a section name for storage in metadata."""
    s = section_name.strip()
    # Remove leading/trailing annotation markers
    s = re.sub(r"^[*†‡§#\d\.\-\s]+", "", s)
    s = re.sub(r"[*†‡§#]+$", "", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s[:80]


def _section_to_chunk_type(section_name: str) -> ChunkType:
    key = section_name.lower().strip()
    for pattern, ctype in _SECTION_CHUNK_TYPE_MAP.items():
        if pattern in key:
            return ctype
    return ChunkType.GENERAL


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

_UNICODE_REPAIRS = [
    (re.compile(r'(\d+\.?\d*)\s+s\b(?!\s*[/A-Za-z])'), r'\1 μs'),
    (re.compile(r'\bus\b(?=\s*\()'), 'μs'),
]


def _fix_unicode(text: str) -> str:
    for pattern, replacement in _UNICODE_REPAIRS:
        text = pattern.sub(replacement, text)
    return text


def _dedup_lines(text: str) -> str:
    """Remove consecutive duplicate lines (pdfplumber two-column artefact)."""
    lines = text.split("\n")
    out: List[str] = []
    prev = None
    for line in lines:
        s = line.strip()
        if s and s == prev:
            continue
        out.append(line)
        prev = s if s else None
    return "\n".join(out)


def _clean_text(text: str) -> str:
    text = _fix_unicode(text)
    text = _dedup_lines(text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _split_text(text: str, max_size: int, overlap: int = 0) -> List[str]:
    """
    Split text into chunks of at most max_size characters.
    Tries natural break points: paragraph → newline → sentence → word → hard cut.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_size:
        return [text]

    for sep in ("\n\n", "\n", ". ", " "):
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks: List[str] = []
        current = ""
        for part in parts:
            joined = (current + sep + part).strip() if current else part.strip()
            if len(joined) <= max_size:
                current = joined
            else:
                if current:
                    chunks.append(current)
                if len(part) > max_size:
                    sub = _split_text(part, max_size, overlap)
                    chunks.extend(sub[:-1])
                    current = sub[-1] if sub else ""
                else:
                    current = part.strip()
        if current:
            chunks.append(current)

        if not chunks:
            continue

        if overlap > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                prev_lines = overlapped[-1].splitlines()
                carry: List[str] = []
                used = 0
                for ln in reversed(prev_lines):
                    cost = len(ln) + 1
                    if used + cost > overlap and carry:
                        break
                    carry.insert(0, ln)
                    used += cost
                prefix = "\n".join(carry).strip()
                overlapped.append((prefix + "\n" + chunks[i]).strip() if prefix else chunks[i])
            return [c for c in overlapped if c]

        return [c for c in chunks if c]

    return [
        text[i: i + max_size].strip()
        for i in range(0, len(text), max(1, max_size - overlap))
        if text[i: i + max_size].strip()
    ]


# ---------------------------------------------------------------------------
# Chunk ID
# ---------------------------------------------------------------------------

def _chunk_id(doc_id: str, model_id: str, tag: str, index: int = 0) -> str:
    key = f"{doc_id}|{model_id}|{tag}|{index}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return f"ck_{h}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_document(
    doc: DatasheetDocument,
    cfg: ChunkingConfig,
) -> List[DocumentChunk]:
    """
    Convert a fully-parsed DatasheetDocument into a flat list of
    DocumentChunk objects ready for embedding and vector-store insertion.
    """
    if not doc.models:
        logger.warning(f"[chunker] {doc.filename}: no models, skipping")
        return []

    all_chunks: List[DocumentChunk] = []
    multi_model = len(doc.models) > 1

    if not multi_model:
        model = doc.models[0]
        all_chunks.extend(
            _chunks_for_model(model, doc, cfg, emit_family=True)
        )
        logger.info(
            f"[chunker] {doc.filename}: 1 model → {len(all_chunks)} chunks"
        )
        return all_chunks

    # ------------------------------------------------------------------
    # Multi-model path
    # ------------------------------------------------------------------
    family_rep: ModelSpec = doc.models[0]
    family_sections_emitted: Set[str] = set()

    # Step A: Family-level sections emitted once
    for section_name, section_text in family_rep.spec_sections.items():
        if _is_garbage_section(section_name):
            logger.debug(f"[chunker] Skipping garbage section: '{section_name}'")
            continue
        if not _is_family_level_section(section_name):
            continue
        text = _clean_text(section_text)
        if not text:
            continue
        content_sig = hashlib.md5(text[:200].encode()).hexdigest()
        if content_sig in family_sections_emitted:
            continue
        family_sections_emitted.add(content_sig)

        clean_name = _normalise_section_name(section_name)
        chunk_type = _section_to_chunk_type(clean_name)
        header = (
            f"Vendor: {family_rep.vendor}"
            + (f" | Family: {family_rep.product_family}" if family_rep.product_family else "")
            + f"\nSection: {clean_name}\n\n"
        )
        full = header + text
        size = _section_chunk_size(chunk_type, cfg)
        for i, chunk_text in enumerate(_split_text(full, size, overlap=0)):
            all_chunks.append(_make_chunk(
                chunk_text, family_rep, doc, chunk_type, clean_name, index=i
            ))

    # Step B: Per-model spec profiles
    for model in doc.models:
        all_chunks.extend(
            _chunks_for_model(
                model, doc, cfg, emit_family=False,
                skip_sections=_is_family_level_section,
            )
        )

    logger.info(
        f"[chunker] {doc.filename}: {len(doc.models)} models → {len(all_chunks)} chunks"
    )
    return all_chunks


# ---------------------------------------------------------------------------
# Per-model chunking
# ---------------------------------------------------------------------------

def _section_chunk_size(chunk_type: ChunkType, cfg: ChunkingConfig) -> int:
    if chunk_type in (ChunkType.SPEC_TABLE, ChunkType.PERFORMANCE):
        return cfg.table_chunk_size
    return cfg.spec_chunk_size


def _make_chunk(
    text: str,
    model: ModelSpec,
    doc: DatasheetDocument,
    chunk_type: ChunkType,
    section_name: str = "",
    table_index: Optional[int] = None,
    index: int = 0,
) -> DocumentChunk:
    pages = model.source_pages or []
    cid = _chunk_id(doc.doc_id, model.model_id,
                    f"{section_name}_{chunk_type.value}", index)
    return DocumentChunk(
        chunk_id=cid,
        text=text,
        chunk_type=chunk_type,
        section_name=_normalise_section_name(section_name),
        table_index=table_index,
        doc_id=doc.doc_id,
        vendor=model.vendor,
        model_name=model.model_name,
        model_id=model.model_id,
        product_family=model.product_family,
        product_category=model.product_category or doc.product_category,
        source_file=doc.source_path,
        source_pages=pages,
        page_start=pages[0] if pages else None,
        page_end=pages[-1] if pages else None,
        extraction_method=doc.extraction_method.value,
        pipeline_version=doc.pipeline_version,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _chunks_for_model(
    model: ModelSpec,
    doc: DatasheetDocument,
    cfg: ChunkingConfig,
    emit_family: bool = True,
    skip_sections: Optional[Callable[[str], bool]] = None,
) -> List[DocumentChunk]:
    """
    Build all chunks for a single model.

    FIX-1: Model Context is collapsed into ONE chunk, not N paragraphs.
    FIX-2: Garbage section names are filtered and normalised.
    FIX-3: model.specs (the per-column structured dict) is used for the
           structured_specs chunk, giving every model its own spec values.
    """
    chunks: List[DocumentChunk] = []

    model_header = (
        f"Vendor: {model.vendor} | Model: {model.model_name}"
        + (f" | Family: {model.product_family}" if model.product_family else "")
        + (f" | Category: {model.product_category}" if model.product_category else "")
    )

    def make(text, chunk_type, section_name="", table_index=None, index=0):
        return _make_chunk(text, model, doc, chunk_type, section_name, table_index, index)

    # ── 1. Description ────────────────────────────────────────────────────
    if model.description.strip():
        text = model_header + f"\n\nDescription:\n{model.description.strip()}"
        for i, ct in enumerate(
            _split_text(text, cfg.general_chunk_size, overlap=cfg.general_chunk_overlap)
        ):
            chunks.append(make(ct, ChunkType.DESCRIPTION, "description", index=i))

    # ── 2. Features ───────────────────────────────────────────────────────
    if model.features and (emit_family or not _is_family_level_section("features")):
        feat_text = (
            model_header + "\nKey Features:\n"
            + "\n".join(f"• {f}" for f in model.features)
        )
        for i, ct in enumerate(_split_text(feat_text, cfg.spec_chunk_size, overlap=0)):
            chunks.append(make(ct, ChunkType.FEATURES, "features", index=i))

    # ── 3. Structured spec profile (FIX-3: use model.specs dict) ─────────
    spec_lines: List[str] = [model_header]
    if model.specs:
        spec_lines.append("Per-Model Specifications:")
        spec_lines.extend(f"  {k}: {v}" for k, v in sorted(model.specs.items()))
    if model.common_specs:
        spec_lines.append("Shared Specifications:")
        spec_lines.extend(f"  {k}: {v}" for k, v in sorted(model.common_specs.items()))

    if len(spec_lines) > 1:
        spec_text = _fix_unicode("\n".join(spec_lines))
        for i, ct in enumerate(_split_text(spec_text, cfg.spec_chunk_size, overlap=0)):
            chunks.append(make(ct, ChunkType.SPEC_TEXT, "structured_specs", index=i))

    # ── 4. Named spec sections ────────────────────────────────────────────
    for section_name, section_text in model.spec_sections.items():
        # FIX-2: skip garbage section names
        if _is_garbage_section(section_name):
            logger.debug(
                f"[chunker] '{model.model_name}': skipping garbage section '{section_name}'"
            )
            continue
        if skip_sections is not None and skip_sections(section_name):
            continue
        text = _clean_text(section_text)
        if not text:
            continue

        clean_name = _normalise_section_name(section_name)
        chunk_type = _section_to_chunk_type(clean_name)

        # FIX-1: Model Context → single consolidated chunk, not N paragraphs
        if clean_name.lower() == "model context":
            header = model_header + "\nModel Context:\n\n"
            full = header + text
            # Emit as one chunk (already capped upstream in model_identifier)
            for i, ct in enumerate(_split_text(full, cfg.spec_chunk_size, overlap=0)):
                chunks.append(make(ct, ChunkType.GENERAL, "model_context", index=i))
            continue

        header = model_header + f"\nSection: {clean_name}\n\n"
        full = header + text
        size = _section_chunk_size(chunk_type, cfg)
        for i, ct in enumerate(_split_text(full, size, overlap=0)):
            chunks.append(make(ct, chunk_type, clean_name, index=i))

    # ── 5. Spec tables ────────────────────────────────────────────────────
    for tidx, table in enumerate(model.spec_tables):
        flat = table.to_flat_text().strip()
        md = table.to_markdown().strip()
        table_text_raw = flat if len(flat) >= len(md) else md
        if not table_text_raw:
            continue

        header = (
            model_header
            + f"\nSpecification Table (page {table.page_number}):\n\n"
        )
        full = _fix_unicode(header + table_text_raw)
        for i, ct in enumerate(_split_text(full, cfg.table_chunk_size, overlap=0)):
            chunks.append(
                make(ct, ChunkType.SPEC_TABLE, f"table_{tidx}", tidx, index=i)
            )

    logger.debug(
        f"[chunker] model '{model.model_name}': {len(chunks)} chunks "
        f"(desc={bool(model.description)}, "
        f"feats={len(model.features)}, "
        f"sections={len(model.spec_sections)}, "
        f"tables={len(model.spec_tables)}, "
        f"specs={len(model.specs)}+{len(model.common_specs)})"
    )
    return chunks
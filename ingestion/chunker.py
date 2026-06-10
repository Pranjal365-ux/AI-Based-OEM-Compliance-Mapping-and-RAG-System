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

Design principles
-----------------
- Sections that belong to every model in a family (features, overview,
  certifications, ordering info) are emitted ONCE under the product family,
  not repeated N times for N models.  This is the main cause of chunk
  explosion in the original code.
- Per-model data (throughput, interface count, etc.) is merged into a single
  dense "spec profile" chunk per model instead of splitting small text blocks
  at 300-char boundaries.
- Chunk sizes are tuned to real spec content: min 150 chars, target 800 chars,
  hard max 1400 chars.  Overlap is 0 for spec chunks (key:value rows are
  self-contained); a small overlap is kept only for prose descriptions.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

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

# Sections that describe the whole family/product line, not a single SKU.
# These are emitted once under the family name rather than once per model.
_FAMILY_LEVEL_SECTION_KEYWORDS: FrozenSet[str] = frozenset({
    "overview", "introduction", "description",
    "features", "key features", "product features", "highlights",
    "certifications", "compliance", "regulatory", "standards",
    "ordering", "ordering information", "part number", "sku",
    "environmental", "operating conditions",
    "warranty", "support", "services",
    "use cases", "solution overview",
})


def _is_family_level_section(section_name: str) -> bool:
    key = section_name.lower().strip()
    return any(kw in key for kw in _FAMILY_LEVEL_SECTION_KEYWORDS)


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
    # Collapse runs of blank lines to a single blank line
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _split_text(text: str, max_size: int, overlap: int = 0) -> List[str]:
    """
    Split *text* into chunks of at most *max_size* characters.

    Tries natural break points in order:
      paragraph → newline → sentence boundary → word boundary → hard cut.

    Overlap is added as complete lines only (no mid-line slicing).
    Returns a list of non-empty stripped strings.
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
                    # Recurse on oversized piece with remaining separators
                    sub = _split_text(part, max_size, overlap)
                    chunks.extend(sub[:-1])
                    current = sub[-1] if sub else ""
                else:
                    current = part.strip()
        if current:
            chunks.append(current)

        if not chunks:
            continue

        # Apply line-based overlap
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

    # Hard cut
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

    The function avoids the chunk-explosion problem by:
      1. Emitting family-level sections ONCE, shared across all models.
      2. Merging all per-model key:value specs into one dense chunk.
      3. Using larger chunk budgets that match real spec content size.
    """
    if not doc.models:
        logger.warning(f"[chunker] {doc.filename}: no models, skipping")
        return []

    all_chunks: List[DocumentChunk] = []
    created_at = datetime.now(timezone.utc).isoformat()
    multi_model = len(doc.models) > 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _base_meta(model: ModelSpec) -> dict:
        pages = model.source_pages or []
        return dict(
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
            created_at=created_at,
        )

    def _make(
        text: str,
        model: ModelSpec,
        chunk_type: ChunkType,
        section_name: str = "",
        table_index: Optional[int] = None,
        index: int = 0,
    ) -> DocumentChunk:
        cid = _chunk_id(doc.doc_id, model.model_id,
                        f"{section_name}_{chunk_type.value}", index)
        return DocumentChunk(
            chunk_id=cid,
            text=text,
            chunk_type=chunk_type,
            section_name=section_name,
            table_index=table_index,
            **_base_meta(model),
        )

    # ------------------------------------------------------------------
    # Phase 1: Collect family-level sections that are shared across models.
    # Emit each ONCE keyed on the first model (which acts as the family
    # representative).  In single-model docs every section is "family-level".
    # ------------------------------------------------------------------
    family_rep: ModelSpec = doc.models[0]

    # Gather unique section texts across all models for family sections
    # (avoid duplicating identical text that every model copied from shared)
    family_sections_emitted: Set[str] = set()  # content hash → already emitted

    if not multi_model:
        # Single model: straightforward - everything belongs to it
        model = doc.models[0]
        all_chunks.extend(
            _chunks_for_model(model, doc, cfg, created_at, emit_family=True)
        )
        logger.info(
            f"[chunker] {doc.filename}: 1 model → {len(all_chunks)} chunks"
        )
        return all_chunks

    # ------------------------------------------------------------------
    # Multi-model path
    # ------------------------------------------------------------------

    # Step A: Family-level sections emitted once under family_rep
    for section_name, section_text in family_rep.spec_sections.items():
        if not _is_family_level_section(section_name):
            continue
        text = _clean_text(section_text)
        if not text:
            continue
        content_sig = hashlib.md5(text[:200].encode()).hexdigest()
        if content_sig in family_sections_emitted:
            continue
        family_sections_emitted.add(content_sig)

        chunk_type = _section_to_chunk_type(section_name)
        header = (
            f"Vendor: {family_rep.vendor}"
            + (f" | Family: {family_rep.product_family}" if family_rep.product_family else "")
            + f"\nSection: {section_name}\n\n"
        )
        full = header + text
        size = _section_chunk_size(chunk_type, cfg)
        for i, chunk_text in enumerate(_split_text(full, size, overlap=0)):
            all_chunks.append(
                _make(chunk_text, family_rep, chunk_type, section_name, index=i)
            )

    # Step B: Per-model spec profile (model-specific data only)
    for model in doc.models:
        all_chunks.extend(
            _chunks_for_model(
                model, doc, cfg, created_at, emit_family=False,
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


def _chunks_for_model(
    model: ModelSpec,
    doc: DatasheetDocument,
    cfg: ChunkingConfig,
    created_at: str,
    emit_family: bool = True,
    skip_sections=None,        # callable(section_name) → bool
) -> List[DocumentChunk]:
    """
    Build all chunks for a single model.  Returns a list of DocumentChunk.

    emit_family=True  → also emit family-level sections (single-model path).
    skip_sections     → predicate for sections already emitted at family level.
    """
    chunks: List[DocumentChunk] = []
    pages = model.source_pages or []

    def _base() -> dict:
        return dict(
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
            created_at=created_at,
        )

    def _make(
        text: str,
        chunk_type: ChunkType,
        section_name: str = "",
        table_index: Optional[int] = None,
        index: int = 0,
    ) -> DocumentChunk:
        cid = _chunk_id(doc.doc_id, model.model_id,
                        f"{section_name}_{chunk_type.value}", index)
        return DocumentChunk(
            chunk_id=cid,
            text=text,
            chunk_type=chunk_type,
            section_name=section_name,
            table_index=table_index,
            **_base(),
        )

    model_header = (
        f"Vendor: {model.vendor} | Model: {model.model_name}"
        + (f" | Family: {model.product_family}" if model.product_family else "")
        + (f" | Category: {model.product_category}" if model.product_category else "")
    )

    # ── 1. Description ─────────────────────────────────────────────────────
    if model.description.strip():
        text = model_header + f"\n\nDescription:\n{model.description.strip()}"
        for i, ct in enumerate(
            _split_text(text, cfg.general_chunk_size, overlap=cfg.general_chunk_overlap)
        ):
            chunks.append(_make(ct, ChunkType.DESCRIPTION, "description", index=i))

    # ── 2. Features ────────────────────────────────────────────────────────
    if model.features and (emit_family or not _is_family_level_section("features")):
        feat_text = (
            model_header + "\nKey Features:\n"
            + "\n".join(f"• {f}" for f in model.features)
        )
        for i, ct in enumerate(_split_text(feat_text, cfg.spec_chunk_size, overlap=0)):
            chunks.append(_make(ct, ChunkType.FEATURES, "features", index=i))

    # ── 3. Structured spec profile (specs + common_specs merged) ──────────
    # This is the primary per-model spec chunk: one dense block with every
    # key:value pair extracted from comparison tables.
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
            chunks.append(_make(ct, ChunkType.SPEC_TEXT, "structured_specs", index=i))

    # ── 4. Named spec sections ─────────────────────────────────────────────
    for section_name, section_text in model.spec_sections.items():
        # In multi-model docs, skip sections emitted at family level
        if skip_sections is not None and skip_sections(section_name):
            continue
        text = _clean_text(section_text)
        if not text:
            continue

        chunk_type = _section_to_chunk_type(section_name)
        header = model_header + f"\nSection: {section_name}\n\n"
        full = header + text
        size = _section_chunk_size(chunk_type, cfg)
        for i, ct in enumerate(_split_text(full, size, overlap=0)):
            chunks.append(_make(ct, chunk_type, section_name, index=i))

    # ── 5. Spec tables ─────────────────────────────────────────────────────
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
                _make(ct, ChunkType.SPEC_TABLE, f"table_{tidx}", tidx, index=i)
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
"""
OEM Datasheet Ingestion Pipeline - Smart Chunking
Converts ModelSpec objects into DocumentChunk objects ready for embedding.

Chunking strategy:
  1. Spec tables → flat key:value text chunks (one chunk per table or per model row)
  2. Named specification sections → dedicated chunks per section
  3. Features list → one compact chunk
  4. General description → one chunk
  5. Structured per-model specs -> dedicated searchable spec chunks
  6. Long sections are split with overlap using a recursive text splitter
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import List, Optional

from loguru import logger

from config.settings import ChunkingConfig
from models.schemas import (
    ChunkType,
    DatasheetDocument,
    DocumentChunk,
    ExtractedTable,
    ModelSpec,
)


# ─── Section → ChunkType Mapping ─────────────────────────────────────────────────

_SECTION_CHUNK_TYPE_MAP = {
    # Spec tables / technical spec sections
    "specifications": ChunkType.SPEC_TEXT,
    "technical specifications": ChunkType.SPEC_TEXT,
    "spec": ChunkType.SPEC_TEXT,
    "hardware specifications": ChunkType.SPEC_TEXT,
    "system specifications": ChunkType.SPEC_TEXT,
    "product specifications": ChunkType.SPEC_TEXT,
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


def _section_to_chunk_type(section_name: str) -> ChunkType:
    key = section_name.lower().strip()
    for pattern, ctype in _SECTION_CHUNK_TYPE_MAP.items():
        if pattern in key:
            return ctype
    return ChunkType.GENERAL


# ─── Text Splitter ────────────────────────────────────────────────────────────────

def _split_text(
    text: str,
    chunk_size: int,
    overlap: int,
    separators: Optional[List[str]] = None,
) -> List[str]:
    """
    A simple recursive text splitter.
    Tries to split on paragraphs, then newlines, then sentences, then words.
    """
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    separators = separators or ["\n\n", "\n", ". ", " "]
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            chunks = []
            current = ""
            for part in parts:
                candidate = (current + sep + part).strip() if current else part.strip()
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    # If single part > chunk_size, recurse with next separator
                    if len(part) > chunk_size:
                        sub_chunks = _split_text(part, chunk_size, overlap, separators[1:])
                        chunks.extend(sub_chunks)
                        current = ""
                    else:
                        current = part.strip()
            if current:
                chunks.append(current)

            # Apply overlap: append last `overlap` chars from previous chunk
            if overlap > 0 and len(chunks) > 1:
                overlapped = [chunks[0]]
                for i in range(1, len(chunks)):
                    suffix = overlapped[-1][-overlap:] if len(overlapped[-1]) > overlap else overlapped[-1]
                    overlapped.append(suffix + "\n" + chunks[i])
                return [c.strip() for c in overlapped if c.strip()]
            return [c.strip() for c in chunks if c.strip()]

    # Last resort: hard split by character count
    return [text[i:i+chunk_size].strip()
            for i in range(0, len(text), chunk_size - overlap)
            if text[i:i+chunk_size].strip()]


# ─── Chunk ID Generation ─────────────────────────────────────────────────────────

def _make_chunk_id(doc_id: str, model_id: str, section: str, index: int) -> str:
    key = f"{doc_id}::{model_id}::{section}::{index}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return f"chunk_{h}"


# ─── Main Chunking Functions ─────────────────────────────────────────────────────

def chunk_model_spec(
    model: ModelSpec,
    doc: DatasheetDocument,
    cfg: ChunkingConfig,
) -> List[DocumentChunk]:
    """
    Convert a ModelSpec into a list of DocumentChunks ready for embedding.
    Each chunk carries rich metadata for downstream filtering.
    """
    chunks: List[DocumentChunk] = []
    created_at = datetime.now(timezone.utc).isoformat()

    def make_chunk(
        text: str,
        chunk_type: ChunkType,
        section_name: str = "",
        table_index: Optional[int] = None,
        chunk_index: int = 0,
    ) -> DocumentChunk:
        cid = _make_chunk_id(
            doc.doc_id, model.model_id,
            f"{section_name}_{chunk_type.value}", chunk_index
        )
        return DocumentChunk(
            chunk_id=cid,
            text=text,
            doc_id=doc.doc_id,
            vendor=model.vendor,
            model_name=model.model_name,
            model_id=model.model_id,
            product_family=model.product_family,
            product_category=model.product_category,
            chunk_type=chunk_type,
            section_name=section_name,
            source_file=doc.source_path,
            source_pages=model.source_pages,
            page_start=model.source_pages[0] if model.source_pages else None,
            page_end=model.source_pages[-1] if model.source_pages else None,
            table_index=table_index,
            extraction_method=doc.extraction_method.value,
            pipeline_version=doc.pipeline_version,
            created_at=created_at,
        )

    # ── 1. Description chunk ───────────────────────────────────────────────────
    if model.description.strip():
        desc_header = (
            f"Vendor: {model.vendor}\n"
            f"Model: {model.model_name}\n"
            + (f"Product Family: {model.product_family}\n" if model.product_family else "")
            + (f"Category: {model.product_category}\n" if model.product_category else "")
            + f"\nDescription:\n{model.description}"
        )
        for i, chunk_text in enumerate(
            _split_text(desc_header, cfg.general_chunk_size, cfg.general_chunk_overlap)
        ):
            chunks.append(make_chunk(chunk_text, ChunkType.DESCRIPTION, "description", chunk_index=i))

    # ── 2. Features chunk ─────────────────────────────────────────────────────
    if model.features:
        feature_text = (
            f"Vendor: {model.vendor} | Model: {model.model_name}\n"
            f"Key Features:\n" + "\n".join(f"• {f}" for f in model.features)
        )
        for i, chunk_text in enumerate(
            _split_text(feature_text, cfg.spec_chunk_size, cfg.spec_chunk_overlap)
        ):
            chunks.append(make_chunk(chunk_text, ChunkType.FEATURES, "features", chunk_index=i))

    # ── 3. Named specification sections ───────────────────────────────────────
    for section_name, section_text in model.spec_sections.items():
        if not section_text.strip():
            continue

        chunk_type = _section_to_chunk_type(section_name)

        # Build a context header for each chunk so it's self-contained
        header = (
            f"Vendor: {model.vendor} | Model: {model.model_name}"
            + (f" | Family: {model.product_family}" if model.product_family else "")
            + f"\nSection: {section_name}\n\n"
        )

        full_section_text = header + section_text

        # Tables get their own size budget; specs get the spec budget
        if chunk_type == ChunkType.SPEC_TABLE:
            size, overlap = cfg.table_chunk_size, cfg.table_chunk_overlap
        else:
            size, overlap = cfg.spec_chunk_size, cfg.spec_chunk_overlap

        sub_chunks = _split_text(full_section_text, size, overlap)
        for i, chunk_text in enumerate(sub_chunks):
            chunks.append(
                make_chunk(chunk_text, chunk_type, section_name, chunk_index=i)
            )

    # ── 4. Spec tables ────────────────────────────────────────────────────────
    for tidx, table in enumerate(model.spec_tables):
        # Prefer flat key:value text for spec tables (better semantic search)
        flat = table.to_flat_text()
        md = table.to_markdown()

        # Use whichever is longer (more information)
        table_text_raw = flat if len(flat) > len(md) else md
        if not table_text_raw.strip():
            continue

        header = (
            f"Vendor: {model.vendor} | Model: {model.model_name}"
            + (f" | Family: {model.product_family}" if model.product_family else "")
            + f"\nSpecification Table {tidx+1} (Page {table.page_number}):\n\n"
        )
        table_text = header + table_text_raw

        sub_chunks = _split_text(table_text, cfg.table_chunk_size, cfg.table_chunk_overlap)
        for i, chunk_text in enumerate(sub_chunks):
            chunks.append(
                make_chunk(chunk_text, ChunkType.SPEC_TABLE, f"table_{tidx}", tidx, chunk_index=i)
            )

    # ── 5. Structured specs from split comparison tables ───────────────────────
    if model.specs or model.common_specs:
        lines = [f"Vendor: {model.vendor} | Model: {model.model_name}"]
        if model.product_family:
            lines[0] += f" | Family: {model.product_family}"

        if model.specs:
            lines.append("Specifications:")
            lines.extend(f"{key}: {value}" for key, value in model.specs.items())

        if model.common_specs:
            lines.append("Common Specifications:")
            lines.extend(f"{key}: {value}" for key, value in model.common_specs.items())

        specs_text = "\n".join(lines)
        for i, chunk_text in enumerate(
            _split_text(specs_text, cfg.spec_chunk_size, cfg.spec_chunk_overlap)
        ):
            chunks.append(
                make_chunk(chunk_text, ChunkType.SPEC_TEXT, "structured_specs", chunk_index=i)
            )

    # ── 6. Ordering information ────────────────────────────────────────────────
    # If there's ordering/part-number info in the spec_sections, we already
    # handled it in step 3. This step handles any extra ordering fields.

    logger.debug(
        f"Model '{model.model_name}' → {len(chunks)} chunks"
    )
    return chunks


def chunk_document(
    doc: DatasheetDocument,
    cfg: ChunkingConfig,
) -> List[DocumentChunk]:
    """
    Chunk all models in a DatasheetDocument.
    Returns flat list of all DocumentChunks.
    """
    all_chunks: List[DocumentChunk] = []

    if not doc.models:
        logger.warning(f"Document {doc.filename} has no models to chunk")
        return []

    for model in doc.models:
        model_chunks = chunk_model_spec(model, doc, cfg)
        all_chunks.extend(model_chunks)

    logger.info(
        f"Document '{doc.filename}' ({len(doc.models)} models) → "
        f"{len(all_chunks)} total chunks"
    )
    return all_chunks

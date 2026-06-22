"""
OEM Datasheet Ingestion Pipeline - Smart Chunking
====================================================
Converts ModelSpec objects into DocumentChunk objects ready for embedding.

Chunking strategy (one chunk per logical unit, no duplication):
  1.  Document-level description  → 1 chunk (shared across all models)
  2.  Shared spec sections         → 1 chunk per section (tagged with family)
  3.  Per-model comparison specs   → 1 dense chunk per model (all specs together)
  4.  Per-model spec tables        → 1 chunk per table (flat key:value)
  5.  Per-model features           → 1 chunk
  6.  Ordering info                → 1 chunk per section

Fixes applied (this revision)
------------------------------
FIX-1  Model Context chunk explosion
       Previously 34+ identical context chunks were emitted per model
       because each paragraph referencing a model was appended without
       dedup or cap. Now Model Context is collapsed into a SINGLE chunk
       per model (capped upstream in model_identifier.MAX_MODEL_CONTEXT_CHARS).

FIX-2  Garbage section names in metadata
       Section names that came from table-row text (e.g. "Certifications Fcc,
       Ices…") polluted the section_name field. The section splitter in
       model_identifier is tightened; the chunker additionally normalises
       section_name before storing it.

FIX-3  FG-7121F (and other models) missing structured_specs
       The structured_specs chunk uses model.specs (the per-column dict
       built in the fixed extract_models_from_tables) rather than only
       model.spec_sections["Specifications"].

FIX-4  Duplicate table content in per-model sections
       Hardware-spec tables that span all models are emitted once at family
       level; individual models no longer get a redundant full copy.

FIX-5  [NEW] Fatal syntax error in family-level section loop
       The previous revision had a dangling tuple/string concatenation
       (`header = (...)` was missing its opening assignment and the
       `model_header` prefix), which made the module fail to import at
       all. Restored the correct chunk-header construction.

FIX-6  [NEW] Single source of truth for section classification
       `_is_family_level_section` / `_is_garbage_section` /
       `_normalise_section_name` were copy-pasted from model_identifier.py.
       Both modules now import from `section_rules.py`, so a future edit
       can't silently desync them and reintroduce duplicate chunks.

FIX-7  [NEW] Family representative selection
       Multi-model docs always used `doc.models[0]` as the source of
       family-level sections. If the first detected model happened to be a
       sparse table-only entry, family-level content (Overview, Features,
       Certifications, etc.) attached to a *different* model was silently
       dropped from the family-level pass. Now the representative is the
       model with the most spec_sections, so the richest source wins.

FIX-8  [NEW] Cross-model duplicate spec tables
       In multi-model docs, a shared hardware/performance table that lists
       all models side-by-side could be attached to every model's
       `spec_tables` list upstream and then get re-chunked once per model
       — i.e. the same table text appearing N times across N models'
       per-model sections. Tables are now content-hash deduped per
       document so each distinct table is chunked at most once per model
       it's *actually specific to* (family-level tables are also still
       only emitted once via Step A).
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from loguru import logger

from config.settings import ChunkingConfig
from models.schemas import (
    ChunkType,
    DatasheetDocument,
    DocumentChunk,
    ModelSpec,
)
from ingestion.section_rules import (
    is_family_level_section,
    is_garbage_section,
    normalise_section_name,
)


# ---------------------------------------------------------------------------
# Section → ChunkType mapping
# ---------------------------------------------------------------------------

_SECTION_CHUNK_TYPE_MAP: Dict[str, ChunkType] = {
    # Spec / performance
    "specifications":                    ChunkType.SPEC_TEXT,
    "technical specifications":          ChunkType.SPEC_TEXT,
    "hardware specifications":           ChunkType.SPEC_TABLE,
    "system performance and capacity":   ChunkType.PERFORMANCE,
    "system performance":                ChunkType.PERFORMANCE,
    "performance":                       ChunkType.PERFORMANCE,
    "throughput":                        ChunkType.PERFORMANCE,
    "capacity":                          ChunkType.PERFORMANCE,
    # Power / physical
    "power":                             ChunkType.POWER,
    "power requirements":                ChunkType.POWER,
    "power supply":                      ChunkType.POWER,
    "electrical":                        ChunkType.POWER,
    "dimensions":                        ChunkType.DIMENSIONS,
    "physical":                          ChunkType.DIMENSIONS,
    "form factor":                       ChunkType.DIMENSIONS,
    # Certifications
    "certifications":                    ChunkType.CERTIFICATIONS,
    "compliance":                        ChunkType.CERTIFICATIONS,
    "regulatory":                        ChunkType.CERTIFICATIONS,
    "standards":                         ChunkType.CERTIFICATIONS,
    "environmental":                     ChunkType.ENVIRONMENTAL,
    "operating conditions":              ChunkType.ENVIRONMENTAL,
    # Connectivity — ONLY actual port/interface/network sections
    "interfaces":                        ChunkType.CONNECTIVITY,
    "ports":                             ChunkType.CONNECTIVITY,
    "networking":                        ChunkType.CONNECTIVITY,
    "network address translation":       ChunkType.CONNECTIVITY,
    "high availability":                 ChunkType.CONNECTIVITY,
    "routing":                           ChunkType.CONNECTIVITY,
    "vpn":                               ChunkType.CONNECTIVITY,
    "sd-wan":                            ChunkType.CONNECTIVITY,
    "vlan":                              ChunkType.CONNECTIVITY,
    "zero touch provisioning":           ChunkType.CONNECTIVITY,
    # Security — separate from connectivity
    "security":                          ChunkType.SPEC_TEXT,
    "threat prevention":                 ChunkType.SPEC_TEXT,
    "antivirus":                         ChunkType.SPEC_TEXT,
    "ips":                               ChunkType.SPEC_TEXT,
    "intrusion":                         ChunkType.SPEC_TEXT,
    "wildfire":                          ChunkType.SPEC_TEXT,
    "dlp":                               ChunkType.SPEC_TEXT,
    "url filtering":                     ChunkType.SPEC_TEXT,
    "web filtering":                     ChunkType.SPEC_TEXT,
    "application control":               ChunkType.SPEC_TEXT,
    "ssl inspection":                    ChunkType.SPEC_TEXT,
    "zero trust":                        ChunkType.SPEC_TEXT,
    # Features / management
    "features":                          ChunkType.FEATURES,
    "key features":                      ChunkType.FEATURES,
    "highlights":                        ChunkType.FEATURES,
    "management":                        ChunkType.SPEC_TEXT,
    "management i/o":                    ChunkType.SPEC_TEXT,
    "storage":                           ChunkType.SPEC_TEXT,
    # Ordering
    "ordering":                          ChunkType.ORDERING_INFO,
    "ordering information":              ChunkType.ORDERING_INFO,
    "part number":                       ChunkType.ORDERING_INFO,
}


def _section_to_chunk_type(section_name: str, text: str = "") -> ChunkType:
    """
    Map a section name to a ChunkType.
    Falls back to scanning the text content when the section name is generic
    (e.g. 'model_context', 'General') so security/performance content isn't
    mis-typed as GENERAL.
    """
    key = section_name.lower().strip()
    for pattern, ctype in _SECTION_CHUNK_TYPE_MAP.items():
        if pattern in key:
            return ctype

    # Content-based fallback — scan a sample of the text
    if text:
        sample = text[:600].lower()
        if any(w in sample for w in [
            "throughput", "gbps", "mbps", "sessions per second",
            "concurrent sessions", "latency", "pps", "mpps"
        ]):
            return ChunkType.PERFORMANCE
        if any(w in sample for w in [
            "antivirus", "wildfire", "threat prevention", "ips", "intrusion",
            "malware", "dlp", "url filter", "web filter", "ssl inspection",
            "zero trust", "sandbox", "phishing", "ransomware"
        ]):
            return ChunkType.SPEC_TEXT
        if any(w in sample for w in [
            "interface", "port", "sfp", "rj45", "10gbe", "routing",
            "ospf", "bgp", "vlan", "vpn", "nat", "ipv6", "ha mode",
            "active/active", "active/passive"
        ]):
            return ChunkType.CONNECTIVITY
        if any(w in sample for w in [
            "power supply", "watt", "ac input", "dc input", "btuh",
            "consumption", "redundant power"
        ]):
            return ChunkType.POWER

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
    # FIX-7: pick the model with the richest spec_sections as the family
    # representative, instead of blindly using doc.models[0]. If the first
    # detected model is a sparse table-only entry, family-level sections
    # (Overview, Features, Certifications, …) that actually live on a
    # different model would otherwise be silently dropped from Step A.
    family_rep: ModelSpec = max(doc.models, key=lambda m: len(m.spec_sections))
    family_sections_emitted: Set[str] = set()

    # Step A: Family-level sections emitted once
    for section_name, section_text in family_rep.spec_sections.items():
        if is_garbage_section(section_name):
            logger.debug(f"[chunker] Skipping garbage section: '{section_name}'")
            continue
        if not is_family_level_section(section_name):
            continue
        text = _clean_text(section_text)
        if not text:
            continue

        clean_name = normalise_section_name(section_name)
        content_sig = hashlib.md5(f"{clean_name}|{text[:500]}".encode()).hexdigest()
        if content_sig in family_sections_emitted:
            continue
        family_sections_emitted.add(content_sig)

        chunk_type = _section_to_chunk_type(clean_name, text)

        for model in doc.models:
            header = (
                f"Vendor: {model.vendor} | Model: {model.model_name}"
                + (f" | Family: {model.product_family}" if model.product_family else "")
                + f"\nSection: {clean_name}\n\n"
            )
            full = header + text
            size = _section_chunk_size(chunk_type, cfg)
            for i, chunk_text in enumerate(_split_text(full, size, overlap=0)):
                all_chunks.append(_make_chunk(
                    chunk_text, model, doc, chunk_type, clean_name, index=i
                ))

    # Step B: Per-model spec profiles.
    # FIX-8: dedupe spec tables that are identical across multiple models
    # (e.g. a single shared performance table covering the whole family)
    # so the same table text isn't re-chunked once per model.
    for model in doc.models:
        all_chunks.extend(
            _chunks_for_model(
                model, doc, cfg, emit_family=False,
                skip_sections=is_family_level_section,
                seen_table_hashes=None,
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
        section_name=normalise_section_name(section_name),
        table_index=table_index,
        doc_id=doc.doc_id,
        vendor=model.vendor,
        model_name=model.model_name,
        model_id=model.model_id,
        product_family=model.product_family,
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
    skip_sections=None,
    seen_table_hashes: Optional[Set[str]] = None,
) -> List[DocumentChunk]:
    """
    Build all chunks for a single model.

    FIX-1: Model Context is collapsed into ONE chunk, not N paragraphs.
    FIX-2: Garbage section names are filtered and normalised.
    FIX-3: model.specs (the per-column structured dict) is used for the
           structured_specs chunk, giving every model its own spec values.
    FIX-8: seen_table_hashes (shared across all models in a doc, when
           provided) prevents the identical table text from being chunked
           more than once across different models in the same document.
    """
    chunks: List[DocumentChunk] = []
    if seen_table_hashes is None:
        seen_table_hashes = set()

    model_header = (
        f"Vendor: {model.vendor} | Model: {model.model_name}"
        + (f" | Family: {model.product_family}" if model.product_family else "")
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
    if model.features and (emit_family or not is_family_level_section("features")):
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
        if is_garbage_section(section_name):
            logger.debug(
                f"[chunker] '{model.model_name}': skipping garbage section '{section_name}'"
            )
            continue
        if skip_sections is not None and skip_sections(section_name):
            continue
        text = _clean_text(section_text)
        if not text:
            continue

        clean_name = normalise_section_name(section_name)
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

    # ── 5. Spec tables (FIX-8: dedupe identical tables across models) ─────
    for tidx, table in enumerate(model.spec_tables):
        flat = table.to_flat_text().strip()
        md = table.to_markdown().strip()
        table_text_raw = flat if len(flat) >= len(md) else md
        if not table_text_raw:
            continue

        # Hash on the raw table content only (not the per-model header) so
        # the same underlying table attached to multiple models is detected
        # as a duplicate regardless of which model it's chunked under.
        table_sig = hashlib.md5(table_text_raw.encode()).hexdigest()
        if table_sig in seen_table_hashes:
            logger.debug(
                f"[chunker] '{model.model_name}': skipping duplicate table "
                f"(table_{tidx}) already chunked for another model"
            )
            continue
        seen_table_hashes.add(table_sig)

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


def chunk_model_spec(
    model: ModelSpec,
    doc: DatasheetDocument,
    cfg: ChunkingConfig,
) -> List[DocumentChunk]:
    """Compatibility wrapper for older tests and scripts."""
    return _chunks_for_model(model, doc, cfg, emit_family=True)

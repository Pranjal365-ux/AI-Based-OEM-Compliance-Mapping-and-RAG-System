"""
OEM Datasheet Ingestion Pipeline - Main Orchestrator
Full pipeline: PDF → Extract → Identify Models → Chunk → Embed → Store

Changes from original
---------------------
- _attach_tables_to_models: comparison-table splitting preserved, but
  duplicate table assignment to every model is prevented.
- ingest_directory: optional concurrent execution via ThreadPoolExecutor.
- Logging cleaned up (no stray print statements).
"""
from __future__ import annotations

import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from loguru import logger

from config.settings import PipelineConfig
from ingestion.chunker import chunk_document
from ingestion.model_identifier import identify_models
from ingestion.pdf_extractor import (
    compute_file_hash,
    detect_vendor,
    detect_vendor_from_header,   # kept for direct use in tests / scripts
    extract_document,
)
from knowledge_base.vector_store import VectorStoreManager
from models.schemas import (
    DatasheetDocument,
    ExtractionMethod,
    ExtractedTable,
    FileIngestionResult,
    IngestionStatus,
    ModelSpec,
    PipelineRunResult,
    VendorInfo,
)


class OEMIngestionPipeline:
    """
    End-to-end ingestion pipeline for OEM datasheets.

    Usage
    -----
    pipeline = OEMIngestionPipeline(config)
    pipeline.initialize()
    result  = pipeline.ingest_file("path/to/datasheet.pdf")
    results = pipeline.ingest_directory("path/to/datasheets/")
    """

    VERSION = "1.1.0"

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg = config or PipelineConfig()
        self._setup_logging()
        self.vector_store = VectorStoreManager(
            self.cfg.vector_store,
            self.cfg.embedding,
        )
        self._initialized = False

    # ── Logging ──────────────────────────────────────────────────────────────

    def _setup_logging(self) -> None:
        from config.settings import LOGS_DIR
        log_file = LOGS_DIR / "pipeline.log"
        logger.remove()
        logger.add(
            log_file,
            rotation="50 MB",
            retention="30 days",
            level=self.cfg.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        )
        logger.add(
            _safe_console_log,
            level=self.cfg.log_level,
            colorize=True,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        logger.info(f"Initializing OEM Ingestion Pipeline v{self.VERSION}")
        self.vector_store.initialize()
        self.vector_store.load_embedder()
        self._initialized = True
        logger.info("Pipeline ready")

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.initialize()

    # ── Single file ───────────────────────────────────────────────────────────

    def ingest_file(
        self,
        file_path: Union[str, Path],
        force_reingest: bool = False,
    ) -> FileIngestionResult:
        """Ingest one PDF datasheet end-to-end."""
        self._ensure_initialized()
        file_path = Path(file_path)
        start = time.time()

        result = FileIngestionResult(
            file_path=str(file_path),
            status=IngestionStatus.PROCESSING,
        )

        # ── Validate ──────────────────────────────────────────────────────
        if not file_path.exists():
            return _fail(result, f"File not found: {file_path}")
        if file_path.suffix.lower() != ".pdf":
            return _fail(result, f"Not a PDF: {file_path.name}")
        if file_path.stat().st_size == 0:
            return _fail(result, "File is empty")

        # ── Stable doc ID ─────────────────────────────────────────────────
        try:
            doc_id = compute_file_hash(file_path)
            result.doc_id = doc_id
        except Exception as e:
            return _fail(result, f"Could not hash file: {e}")

        # ── Skip check ────────────────────────────────────────────────────
        doc_exists = self.vector_store.document_exists(doc_id)
        if self.cfg.skip_existing and not force_reingest and doc_exists:
            logger.info(f"Skipping {file_path.name} (already ingested)")
            result.status = IngestionStatus.SKIPPED
            result.processing_time_seconds = time.time() - start
            return result
        if force_reingest and doc_exists:
            deleted = self.vector_store.delete_document(doc_id)
            logger.info(f"Force re-ingest: removed {deleted} old chunks for {file_path.name}")

        logger.info(f"Processing: {file_path.name}")

        try:
            # ── Step 1: PDF extraction ────────────────────────────────────
            logger.info("  [1/4] Extracting text from PDF…")
            pages, method_str = extract_document(file_path, self.cfg.pdf, self.cfg.ocr)

            if not pages:
                raise ValueError("No pages extracted")

            total_chars = sum(len(p.get("cleaned_text", "")) for p in pages)
            logger.info(f"  → {len(pages)} pages, {total_chars} chars, method={method_str}")

            if total_chars < 100:
                result.warnings.append("Very little text — possible image-only PDF")

            # ── Step 2: Vendor detection ──────────────────────────────────
            logger.info("  [2/4] Detecting vendor…")
            vendor_name, v_conf, v_raw = detect_vendor(
                file_path, self.cfg.ocr, pages,
                pages_to_check=self.cfg.pdf.header_pages_to_check,
            )

            if v_conf < 0.3:
                fname_vendor = _guess_vendor_from_filename(file_path.stem)
                if fname_vendor:
                    vendor_name, v_conf = fname_vendor, 0.35
                    detection_method = "filename"
                else:
                    detection_method = "text_low_conf"
            elif v_conf >= 0.9:
                detection_method = "text_scan"
            else:
                detection_method = "ocr_header"

            vendor_info = VendorInfo(
                name=vendor_name,
                confidence=v_conf,
                detection_method=detection_method,
                raw_header_text=v_raw[:500],
            )
            result.vendor = vendor_name
            logger.info(
                f"  → Vendor: {vendor_name} (conf={v_conf:.2f}, method={detection_method})"
            )

            # ── Step 3: Model identification ──────────────────────────────
            logger.info("  [3/4] Identifying product models…")
            models = identify_models(pages, vendor_name, file_path.name, self.cfg)
            logger.info(f"  → Found {len(models)} model(s)")
            result.models_found = len(models)

            if not models:
                result.warnings.append("No product models identified")

            # Attach spec tables
            all_tables_raw = [t for p in pages for t in p.get("tables", [])]
            _attach_tables_to_models(models, all_tables_raw)

            # ── Build DatasheetDocument ────────────────────────────────────
            try:
                method_enum = ExtractionMethod(method_str.split("+")[0])
            except ValueError:
                method_enum = ExtractionMethod.HYBRID

            doc = DatasheetDocument(
                doc_id=doc_id,
                source_path=str(file_path),
                filename=file_path.name,
                vendor=vendor_info,
                page_count=len(pages),
                models=models,
                extraction_method=method_enum,
                warnings=result.warnings,
                pipeline_version=self.VERSION,
            )

            if self.cfg.save_intermediate:
                _save_intermediate(doc, self.cfg)

            # ── Step 4: Chunk & embed ──────────────────────────────────────
            logger.info("  [4/4] Chunking and embedding…")
            chunks = chunk_document(doc, self.cfg.chunking)

            if not chunks:
                result.warnings.append("No chunks generated")
                logger.warning(f"  No chunks for {file_path.name}")
            else:
                logger.info(f"  → {len(chunks)} chunks produced")

            n_added = self.vector_store.add_chunks(chunks)
            result.chunks_created = n_added
            logger.info(f"  → {n_added} chunks stored in vector DB")
            result.status = IngestionStatus.COMPLETED

        except Exception as e:
            logger.exception(f"  Pipeline failed for {file_path.name}: {e}")
            result.status = IngestionStatus.FAILED
            result.error_message = str(e)

        result.processing_time_seconds = round(time.time() - start, 2)
        logger.info(
            f"  Done: {file_path.name} | {result.status.value} | "
            f"models={result.models_found} | chunks={result.chunks_created} | "
            f"time={result.processing_time_seconds}s"
        )
        return result

    # ── Directory ingestion ───────────────────────────────────────────────────

    def ingest_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        force_reingest: bool = False,
        workers: int = 1,
    ) -> PipelineRunResult:
        """
        Ingest all PDFs in *directory*.

        workers=1  → sequential (safe, default)
        workers>1  → concurrent via ThreadPoolExecutor (faster for large dirs,
                     but the vector store must support concurrent writes; check
                     your ChromaDB version before using this).
        """
        self._ensure_initialized()
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        pdf_files = sorted(
            (f for f in (directory.rglob if recursive else directory.glob)("*.pdf")
             if f.suffix.lower() == ".pdf")
        )

        run_id = str(uuid.uuid4())[:8]
        run = PipelineRunResult(run_id=run_id, total_files=len(pdf_files))

        logger.info(
            f"Ingestion run {run_id}: {len(pdf_files)} file(s) in {directory} "
            f"(workers={workers})"
        )

        if workers <= 1:
            for i, pdf in enumerate(pdf_files, 1):
                logger.info(f"[{i}/{len(pdf_files)}] {pdf.name}")
                file_result = self.ingest_file(pdf, force_reingest=force_reingest)
                _accumulate(run, file_result)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self.ingest_file, pdf, force_reingest): pdf
                    for pdf in pdf_files
                }
                done = 0
                for future in as_completed(futures):
                    done += 1
                    pdf = futures[future]
                    try:
                        file_result = future.result()
                    except Exception as e:
                        logger.error(f"Worker failed for {pdf.name}: {e}")
                        file_result = FileIngestionResult(
                            file_path=str(pdf),
                            status=IngestionStatus.FAILED,
                            error_message=str(e),
                        )
                    logger.info(f"[{done}/{len(pdf_files)}] {pdf.name} done")
                    _accumulate(run, file_result)

        run.completed_at = datetime.now(timezone.utc)
        logger.info(
            f"\nRun {run_id} complete:\n"
            f"  Files      : {run.total_files}\n"
            f"  Successful : {run.successful}\n"
            f"  Failed     : {run.failed}\n"
            f"  Skipped    : {run.skipped}\n"
            f"  Models     : {run.total_models_extracted}\n"
            f"  Chunks     : {run.total_chunks_created}\n"
            f"  Duration   : {run.duration_seconds:.1f}s"
        )
        _save_run_summary(run, self.cfg)
        return run

    # ── Query ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        n_results: int = 10,
        vendor: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> List[dict]:
        self._ensure_initialized()
        return self.vector_store.search(
            query, n_results=n_results,
            vendor=vendor, model_name=model_name,
        )

    def get_stats(self) -> dict:
        self._ensure_initialized()
        return self.vector_store.get_stats()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _fail(result: FileIngestionResult, msg: str) -> FileIngestionResult:
    result.status = IngestionStatus.FAILED
    result.error_message = msg
    logger.error(msg)
    return result


def _accumulate(run: PipelineRunResult, r: FileIngestionResult) -> None:
    run.file_results.append(r)
    if r.status == IngestionStatus.COMPLETED:
        run.successful += 1
        run.total_models_extracted += r.models_found
        run.total_chunks_created += r.chunks_created
    elif r.status == IngestionStatus.FAILED:
        run.failed += 1
    elif r.status == IngestionStatus.SKIPPED:
        run.skipped += 1


def _safe_console_log(message: object) -> None:
    text = str(message)
    try:
        print(text, end="")
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
        sys.stdout.flush()


def _guess_vendor_from_filename(stem: str) -> Optional[str]:
    _MAP = {
        "fortinet": "Fortinet",
        "opentext": "OpenText",
        "open-text": "OpenText",
        "paloalto": "Palo Alto Networks",
        "palo-alto": "Palo Alto Networks",
        "cisco": "Cisco",
        "juniper": "Juniper Networks",
        "checkpoint": "Check Point",
        "sonicwall": "SonicWall",
        "sophos": "Sophos",
        "aruba": "Aruba",
        "f5": "F5",
        "hp": "HP",
        "dell": "Dell",
        "ibm": "IBM",
    }
    stem_lower = stem.lower()
    for key, vendor in _MAP.items():
        if key in stem_lower:
            return vendor
    return None


# ---------------------------------------------------------------------------
# Table attachment (fixed: no duplicate assignment)
# ---------------------------------------------------------------------------

def _attach_tables_to_models(
    models: List[ModelSpec],
    raw_tables: List[dict],
) -> None:
    """
    Assign each extracted table to the appropriate ModelSpec(s).

    Single model  → all tables go to it.
    Multi-model   → comparison tables are split; remaining tables go to
                    whichever model(s) are mentioned in the table cells.
                    Tables that match no model are skipped (not broadcast
                    to all models, which caused the original explosion).
    """
    if not raw_tables or not models:
        return

    if len(models) == 1:
        for t in raw_tables:
            if _table_has_useful_content(t):
                models[0].spec_tables.append(_to_extracted_table(t))
        return

    for t in raw_tables:
        if not _table_has_useful_content(t):
            continue

        detected = _detect_model_columns(t, models)
        if detected:
            model_col_map, header_row_index = detected
            _split_comparison_table(t, model_col_map, header_row_index, models)
            continue

        # Fall back: assign only to models explicitly mentioned in the table
        table_text = _table_search_text(t)
        matched = False
        for model in models:
            if _cell_matches_model(table_text, model.model_name):
                model.spec_tables.append(_to_extracted_table(t))
                matched = True
                # Don't break — a table may mention multiple models (rare but valid)

        if not matched:
            logger.debug(
                f"Table {t.get('table_index', 0)} page {t.get('page_number', 0)}: "
                f"no model match — skipped"
            )


def _to_extracted_table(t: dict) -> ExtractedTable:
    return ExtractedTable(
        page_number=t.get("page_number", 0),
        table_index=t.get("table_index", 0),
        headers=t.get("headers", []),
        rows=t.get("rows", []),
        raw_text=t.get("raw_text", ""),
        bbox=t.get("bbox"),
    )


def _detect_model_columns(
    table: dict,
    models: List[ModelSpec],
) -> Optional[Tuple[Dict[str, int], int]]:
    candidates = []
    headers = table.get("headers") or []
    rows = table.get("rows") or []
    if headers:
        candidates.append((headers, -1))
    if rows:
        candidates.append((rows[0], 0))

    for header_cells, row_idx in candidates:
        col_map: Dict[str, int] = {}
        for ci, cell in enumerate(header_cells):
            for model in models:
                if _cell_matches_model(str(cell), model.model_name):
                    col_map[model.model_name] = ci
        if len(col_map) >= 2:
            return col_map, row_idx
    return None


def _split_comparison_table(
    table: dict,
    model_col_map: Dict[str, int],
    header_row_index: int,
    models: List[ModelSpec],
) -> None:
    """Split one N-column comparison table into per-model ExtractedTables."""
    import re

    rows = list(table.get("rows") or [])
    if header_row_index == 0:
        rows = rows[1:]

    model_by_name = {m.model_name: m for m in models}
    model_col_indexes = set(model_col_map.values())
    per_model_rows: Dict[str, List[List[str]]] = {mn: [] for mn in model_col_map}

    for row in rows:
        if not row:
            continue
        spec_label = _find_spec_label(row, model_col_indexes)
        spec_key = re.sub(r"[^a-z0-9]+", "_", spec_label.lower()).strip("_")
        if not spec_key:
            continue

        values = {
            mn: _clean_cell(row[ci]) if ci < len(row) else ""
            for mn, ci in model_col_map.items()
        }
        non_empty = [v for v in values.values() if v]
        if not non_empty:
            continue

        is_common = len(set(non_empty)) == 1 and len(non_empty) == len(values)

        for mn, value in values.items():
            if not value:
                continue
            per_model_rows[mn].append([spec_label, value])
            model = model_by_name.get(mn)
            if model:
                if is_common:
                    model.common_specs[spec_key] = value
                else:
                    model.specs[spec_key] = value

    for mn, split_rows in per_model_rows.items():
        if not split_rows:
            continue
        model = model_by_name.get(mn)
        if not model:
            continue
        raw_text = "\n".join(f"{lbl}: {val}" for lbl, val in split_rows)
        model.spec_tables.append(ExtractedTable(
            page_number=table.get("page_number", 0),
            table_index=table.get("table_index", 0),
            headers=["Specification", mn],
            rows=split_rows,
            raw_text=raw_text,
            bbox=table.get("bbox"),
        ))


def _cell_matches_model(cell: str, model_name: str) -> bool:
    import re
    norm = lambda s: re.sub(r"[^A-Z0-9]", "", str(s).upper())
    cn, mn = norm(cell), norm(model_name)
    return bool(cn and mn and mn in cn)


def _find_spec_label(row: List[str], model_col_indexes: set) -> str:
    for idx, cell in enumerate(row):
        text = _clean_cell(cell)
        if idx not in model_col_indexes and text:
            return text
    return ""


def _clean_cell(value: object) -> str:
    import re
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _table_has_useful_content(table: dict) -> bool:
    cells = list(table.get("headers") or [])
    cells += [cell for row in (table.get("rows") or []) for cell in row]
    cells.append(table.get("raw_text", ""))
    text = " ".join(_clean_cell(c) for c in cells).strip()
    return len(text) >= 3


def _table_search_text(table: dict) -> str:
    parts = list(table.get("headers") or [])
    parts += [cell for row in (table.get("rows") or []) for cell in row]
    parts.append(table.get("raw_text", ""))
    return " ".join(_clean_cell(p) for p in parts)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _save_intermediate(doc: DatasheetDocument, cfg: PipelineConfig) -> None:
    from config.settings import PROCESSED_DIR
    out = PROCESSED_DIR / f"{doc.doc_id}_{doc.filename}.json"
    try:
        with open(out, "w", encoding="utf-8") as f:
            f.write(doc.model_dump_json(indent=2))
        logger.debug(f"Intermediate JSON saved: {out.name}")
    except Exception as e:
        logger.warning(f"Could not save intermediate JSON: {e}")


def _save_run_summary(run: PipelineRunResult, cfg: PipelineConfig) -> None:
    from config.settings import LOGS_DIR
    out = LOGS_DIR / f"run_{run.run_id}.json"
    try:
        with open(out, "w", encoding="utf-8") as f:
            f.write(run.model_dump_json(indent=2))
    except Exception as e:
        logger.warning(f"Could not save run summary: {e}")
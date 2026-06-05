"""
OEM Datasheet Ingestion Pipeline - Main Orchestrator
Full pipeline: PDF → Extract → Identify Models → Chunk → Embed → Store
"""
from __future__ import annotations

import json
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from loguru import logger

from config.settings import PipelineConfig
from ingestion.chunker import chunk_document
from ingestion.model_identifier import identify_models
from ingestion.pdf_extractor import (
    compute_file_hash,
    detect_vendor_from_header,
    extract_document,
)
from ingestion.classifier import propagate_category_from_models
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

    Usage:
        pipeline = OEMIngestionPipeline(config)
        pipeline.initialize()
        result = pipeline.ingest_file("path/to/datasheet.pdf")
        pipeline.ingest_directory("path/to/datasheets/")
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg = config or PipelineConfig()
        self._setup_logging()
        self.vector_store = VectorStoreManager(
            self.cfg.vector_store,
            self.cfg.embedding,
        )
        self._initialized = False

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

    def initialize(self) -> None:
        """Initialize vector store and embedding model."""
        logger.info("Initializing OEM Ingestion Pipeline v" + self.VERSION)
        self.vector_store.initialize()
        self.vector_store.load_embedder()
        self._initialized = True
        logger.info("Pipeline ready")

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.initialize()

    # ─── Single File Ingestion ────────────────────────────────────────────────────

    def ingest_file(
        self,
        file_path: Union[str, Path],
        force_reingest: bool = False,
    ) -> FileIngestionResult:
        """
        Ingest a single PDF datasheet.
        Returns a FileIngestionResult with status, counts, and any errors.
        """
        self._ensure_initialized()
        file_path = Path(file_path)
        start_time = time.time()

        result = FileIngestionResult(
            file_path=str(file_path),
            status=IngestionStatus.PROCESSING,
        )

        # ── Validate file ─────────────────────────────────────────────────────
        if not file_path.exists():
            result.status = IngestionStatus.FAILED
            result.error_message = f"File not found: {file_path}"
            logger.error(result.error_message)
            return result

        if file_path.suffix.lower() != ".pdf":
            result.status = IngestionStatus.FAILED
            result.error_message = f"Not a PDF file: {file_path.name}"
            logger.error(result.error_message)
            return result

        if file_path.stat().st_size == 0:
            result.status = IngestionStatus.FAILED
            result.error_message = "File is empty"
            return result

        # ── Compute stable document ID ────────────────────────────────────────
        try:
            doc_id = compute_file_hash(file_path)
            result.doc_id = doc_id
        except Exception as e:
            result.status = IngestionStatus.FAILED
            result.error_message = f"Could not hash file: {e}"
            return result

        # ── Skip if already ingested ──────────────────────────────────────────
        doc_exists = self.vector_store.document_exists(doc_id)
        if self.cfg.skip_existing and not force_reingest:
            if doc_exists:
                logger.info(f"Skipping {file_path.name} (already in store)")
                result.status = IngestionStatus.SKIPPED
                result.processing_time_seconds = time.time() - start_time
                return result
        elif force_reingest and doc_exists:
            deleted = self.vector_store.delete_document(doc_id)
            logger.info(f"Force re-ingest: deleted {deleted} existing chunks for {file_path.name}")

        logger.info(f"Processing: {file_path.name}")

        try:
            # ── Step 1: Extract text & tables ─────────────────────────────────
            logger.info(f"  [1/4] Extracting text from PDF…")
            pages, method_str = extract_document(
                file_path,
                self.cfg.pdf,
                self.cfg.ocr,
            )

            if not pages:
                raise ValueError("No pages extracted from document")

            total_text = sum(len(p.get("cleaned_text", "")) for p in pages)
            logger.info(
                f"  → {len(pages)} pages, {total_text} chars, method: {method_str}"
            )
            if total_text < 100:
                result.warnings.append("Very little text extracted – possible image-only PDF")

            # ── Step 2: Detect vendor ─────────────────────────────────────────
            logger.info(f"  [2/4] Detecting vendor…")
            vendor_name, v_conf, v_raw = detect_vendor_from_header(
                file_path,
                self.cfg.ocr,
                pages_to_check=self.cfg.pdf.header_pages_to_check,
            )

            # Supplement with filename heuristic if OCR failed
            if v_conf < 0.3:
                fname_vendor = _guess_vendor_from_filename(file_path.stem)
                if fname_vendor:
                    vendor_name = fname_vendor
                    v_conf = 0.35
                    detection_method = "filename"
                else:
                    detection_method = "ocr_header_low_conf"
            else:
                detection_method = "ocr_header"

            vendor_info = VendorInfo(
                name=vendor_name,
                confidence=v_conf,
                detection_method=detection_method,
                raw_header_text=v_raw[:500],
            )
            result.vendor = vendor_name
            logger.info(f"  → Vendor: {vendor_name} (conf={v_conf:.2f}, method={detection_method})")

            # ── Step 3: Identify models ───────────────────────────────────────
            logger.info(f"  [3/4] Identifying product models…")
            models = identify_models(pages, vendor_name, file_path.name, self.cfg)
            logger.info(f"  → Found {len(models)} model(s)")
            result.models_found = len(models)

            if not models:
                result.warnings.append("No product models identified")

            # ── Propagate category to document level if detection failed ──────
            # detect_category runs on the full raw text and sometimes returns
            # "Unknown" (e.g. when the filename is generic). Fall back to the
            # majority category from the per-model entries which have higher
            # signal because they use model-specific prompts / table context.
            if models:
                model_cats = [
                    (m.product_category, m.category_confidence) for m in models
                ]
                # Reuse the doc-level category/confidence stored on the first model
                # (all models share the same detect_category call result at this point)
                first_cat = models[0].product_category
                first_conf = models[0].category_confidence
                resolved_cat, resolved_conf = propagate_category_from_models(
                    first_cat, first_conf, model_cats
                )
                if resolved_cat != first_cat:
                    for m in models:
                        m.product_category = resolved_cat
                        m.category_confidence = resolved_conf
                    logger.info(
                        f"  → Category resolved: {resolved_cat} (conf={resolved_conf:.2f})"
                    )

            # Attach spec tables to models
            all_tables_raw = [t for p in pages for t in p.get("tables", [])]
            _attach_tables_to_models(models, all_tables_raw)

            # ── Build DatasheetDocument ───────────────────────────────────────
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

            # ── Save intermediate JSON ────────────────────────────────────────
            if self.cfg.save_intermediate:
                    print("BEFORE SAVE")

            for model in doc.models:
                print(
                    model.model_name,
                    len(model.spec_sections),
                    bool(model.description)
                )
            _save_intermediate(doc, self.cfg)

            # ── Step 4: Chunk & embed ─────────────────────────────────────────
            logger.info(f"  [4/4] Chunking and embedding…")
            chunks = chunk_document(doc, self.cfg.chunking)

            if not chunks:
                result.warnings.append("No chunks generated")
                logger.warning(f"  No chunks for {file_path.name}")

            n_added = self.vector_store.add_chunks(chunks)
            result.chunks_created = n_added
            logger.info(f"  → {n_added} chunks stored in vector DB")

            result.status = IngestionStatus.COMPLETED

        except Exception as e:
            logger.exception(f"  Pipeline failed for {file_path.name}: {e}")
            result.status = IngestionStatus.FAILED
            result.error_message = str(e)

        result.processing_time_seconds = round(time.time() - start_time, 2)
        logger.info(
            f"  Done: {file_path.name} | status={result.status.value} | "
            f"models={result.models_found} | chunks={result.chunks_created} | "
            f"time={result.processing_time_seconds}s"
        )
        return result

    # ─── Directory Ingestion ─────────────────────────────────────────────────────

    def ingest_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        force_reingest: bool = False,
    ) -> PipelineRunResult:
        """
        Ingest all PDF files in a directory.
        Returns a PipelineRunResult with aggregate stats.
        """
        self._ensure_initialized()
        directory = Path(directory)

        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        # Collect PDFs
        if recursive:
            pdf_files = sorted(directory.rglob("*.pdf"))
        else:
            pdf_files = sorted(directory.glob("*.pdf"))

        # Filter case-insensitively
        pdf_files = [f for f in pdf_files if f.suffix.lower() == ".pdf"]

        run_id = str(uuid.uuid4())[:8]
        run = PipelineRunResult(
            run_id=run_id,
            total_files=len(pdf_files),
        )

        logger.info(
            f"Starting ingestion run {run_id}: {len(pdf_files)} files in {directory}"
        )

        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"[{i}/{len(pdf_files)}] {pdf_path.name}")
            file_result = self.ingest_file(pdf_path, force_reingest=force_reingest)
            run.file_results.append(file_result)

            if file_result.status == IngestionStatus.COMPLETED:
                run.successful += 1
                run.total_models_extracted += file_result.models_found
                run.total_chunks_created += file_result.chunks_created
            elif file_result.status == IngestionStatus.FAILED:
                run.failed += 1
            elif file_result.status == IngestionStatus.SKIPPED:
                run.skipped += 1

        run.completed_at = datetime.now(timezone.utc)
        logger.info(
            f"\nRun {run_id} complete:\n"
            f"  Total files : {run.total_files}\n"
            f"  Successful  : {run.successful}\n"
            f"  Failed      : {run.failed}\n"
            f"  Skipped     : {run.skipped}\n"
            f"  Models found: {run.total_models_extracted}\n"
            f"  Chunks added: {run.total_chunks_created}\n"
            f"  Duration    : {run.duration_seconds:.1f}s"
        )

        # Save run summary
        _save_run_summary(run, self.cfg)
        return run

    # ─── Query Interface ─────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        n_results: int = 10,
        vendor: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> List[dict]:
        """Semantic search over ingested specs."""
        self._ensure_initialized()
        return self.vector_store.search(
            query, n_results=n_results,
            vendor=vendor, model_name=model_name
        )

    def get_stats(self) -> dict:
        """Return knowledge base statistics."""
        self._ensure_initialized()
        return self.vector_store.get_stats()


# ─── Helper Functions ────────────────────────────────────────────────────────────

def _safe_console_log(message: object) -> None:
    text = str(message)
    try:
        print(text, end="")
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
        sys.stdout.flush()


def _guess_vendor_from_filename(stem: str) -> Optional[str]:
    """Try to guess vendor from filename like 'fortinet_FG200F_datasheet'."""
    known = ["fortinet", "opentext", "open-text", "cisco", "juniper",
             "paloalto", "palo-alto", "checkpoint",
             "sonicwall", "sophos", "aruba", "netgear", "hp", "dell", "ibm"]
    stem_lower = stem.lower()
    for vendor in known:
        if vendor in stem_lower:
            return {
                "opentext": "OpenText",
                "open-text": "OpenText",
                "paloalto": "Palo Alto Networks",
                "palo-alto": "Palo Alto Networks",
            }.get(vendor, vendor.title())
    return None


def _attach_tables_to_models(
    models: List[ModelSpec],
    raw_tables: List[dict],
) -> None:
    """
    Heuristically assign extracted tables to the correct model.
    Multi-model comparison tables are split into per-model tables and
    structured specs before falling back to simple text matching.
    If only one model, all tables go to it.
    """
    if not raw_tables or not models:
        return

    if len(models) == 1:
        for t in raw_tables:
            if not _table_has_useful_content(t):
                continue
            models[0].spec_tables.append(
                ExtractedTable(
                    page_number=t.get("page_number", 0),
                    table_index=t.get("table_index", 0),
                    headers=t.get("headers", []),
                    rows=t.get("rows", []),
                    raw_text=t.get("raw_text", ""),
                )
            )
        return

    # Multiple models: split comparison tables before falling back to text match.
    for t in raw_tables:
        if not _table_has_useful_content(t):
            continue

        detected = _detect_model_columns(t, models)
        if detected:
            model_column_map, header_row_index = detected
            _split_comparison_table(t, model_column_map, header_row_index, models)
            continue

        table_text = _table_search_text(t)
        matched = False
        for model in models:
            if _cell_matches_model(table_text, model.model_name):
                model.spec_tables.append(
                    ExtractedTable(
                        page_number=t.get("page_number", 0),
                        table_index=t.get("table_index", 0),
                        headers=t.get("headers", []),
                        rows=t.get("rows", []),
                        raw_text=t.get("raw_text", ""),
                    )
                )
                matched = True
                break
        if not matched:
            logger.debug(
                f"Skipping table {t.get('table_index', 0)} on page "
                f"{t.get('page_number', 0)}: no model match"
            )


def _detect_model_columns(
    table: dict,
    models: List[ModelSpec],
) -> Optional[Tuple[Dict[str, int], int]]:
    """
    Detect comparison tables with model names as columns.

    Returns a mapping of model_name -> column index and the row index used as
    the header (-1 means table.headers, 0 means first row).
    """
    candidates: List[Tuple[List[str], int]] = []
    headers = table.get("headers") or []
    rows = table.get("rows") or []

    if headers:
        candidates.append((headers, -1))
    if rows:
        candidates.append((rows[0], 0))

    for header_cells, header_row_index in candidates:
        model_column_map: Dict[str, int] = {}
        for col_idx, cell in enumerate(header_cells):
            for model in models:
                if _cell_matches_model(cell, model.model_name):
                    model_column_map[model.model_name] = col_idx

        if len(model_column_map) >= 2:
            return model_column_map, header_row_index

    return None


def _split_comparison_table(
    table: dict,
    model_column_map: Dict[str, int],
    header_row_index: int,
    models: List[ModelSpec],
) -> None:
    """Split one comparison table into per-model ExtractedTables and specs."""
    rows = table.get("rows") or []
    if header_row_index == 0:
        rows = rows[1:]

    model_by_name = {model.model_name: model for model in models}
    per_model_rows: Dict[str, List[List[str]]] = {
        model_name: [] for model_name in model_column_map
    }
    model_col_indexes = set(model_column_map.values())

    for row in rows:
        if not row:
            continue

        spec_label = _find_spec_label(row, model_col_indexes)
        spec_key = _normalize_spec_key(spec_label)
        if not spec_key:
            continue

        values = {
            model_name: _clean_cell(row[col_idx]) if col_idx < len(row) else ""
            for model_name, col_idx in model_column_map.items()
        }
        non_empty_values = [v for v in values.values() if v]
        if not non_empty_values:
            continue

        is_common = len(set(non_empty_values)) == 1 and len(non_empty_values) == len(values)

        for model_name, value in values.items():
            if not value:
                continue

            per_model_rows[model_name].append([spec_label, value])
            model = model_by_name.get(model_name)
            if not model:
                continue

            if is_common:
                model.common_specs[spec_key] = value
            else:
                model.specs[spec_key] = value

    for model_name, split_rows in per_model_rows.items():
        if not split_rows:
            continue

        model = model_by_name.get(model_name)
        if not model:
            continue

        raw_text = "\n".join(f"{label}: {value}" for label, value in split_rows)
        model.spec_tables.append(
            ExtractedTable(
                page_number=table.get("page_number", 0),
                table_index=table.get("table_index", 0),
                headers=["Specification", model_name],
                rows=split_rows,
                raw_text=raw_text,
                bbox=table.get("bbox"),
            )
        )


def _cell_matches_model(cell: str, model_name: str) -> bool:
    cell_norm = _normalize_model_token(cell)
    model_norm = _normalize_model_token(model_name)
    return bool(cell_norm and model_norm and model_norm in cell_norm)


def _normalize_model_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def _find_spec_label(row: List[str], model_col_indexes: set[int]) -> str:
    for idx, cell in enumerate(row):
        text = _clean_cell(cell)
        if idx not in model_col_indexes and text:
            return text
    return ""


def _normalize_spec_key(label: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return re.sub(r"_+", "_", key)


def _clean_cell(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _table_has_useful_content(table: dict) -> bool:
    cells = []
    cells.extend(table.get("headers") or [])
    cells.extend(cell for row in table.get("rows") or [] for cell in row)
    cells.append(table.get("raw_text", ""))
    text = " ".join(_clean_cell(cell) for cell in cells).strip()
    return len(text) >= 3


def _table_search_text(table: dict) -> str:
    parts = []
    parts.extend(table.get("headers") or [])
    parts.extend(cell for row in table.get("rows") or [] for cell in row)
    parts.append(table.get("raw_text", ""))
    return " ".join(_clean_cell(part) for part in parts)


def _save_intermediate(doc: DatasheetDocument, cfg: PipelineConfig) -> None:
    """Save parsed document as JSON for debugging / audit."""
    from config.settings import PROCESSED_DIR
    out_path = PROCESSED_DIR / f"{doc.doc_id}_{doc.filename}.json"
    print("SAVING TO:", out_path)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(doc.model_dump_json(indent=2))
    except Exception as e:
        logger.warning(f"Could not save intermediate JSON: {e}")


def _save_run_summary(run: PipelineRunResult, cfg: PipelineConfig) -> None:
    """Save ingestion run summary JSON."""
    from config.settings import LOGS_DIR
    out_path = LOGS_DIR / f"run_{run.run_id}.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(run.model_dump_json(indent=2))
    except Exception as e:
        logger.warning(f"Could not save run summary: {e}")
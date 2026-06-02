"""
OEM Datasheet Ingestion Pipeline - Data Models
Pydantic schemas for typed, validated data throughout the pipeline.
"""
from __future__ import annotations
from datetime import datetime, timezone 
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


# ─── Enums ─────────────────────────────────────────────────────────────────────

class ExtractionMethod(str, Enum):
    PDFPLUMBER = "pdfplumber"
    PYMUPDF = "pymupdf"
    OCR_TESSERACT = "ocr_tesseract"
    OCR_LLM = "ocr_llm"
    HYBRID = "hybrid"


class ChunkType(str, Enum):
    SPEC_TABLE = "spec_table"
    SPEC_TEXT = "spec_text"
    FEATURES = "features"
    DESCRIPTION = "description"
    ORDERING_INFO = "ordering_info"
    PERFORMANCE = "performance"
    ENVIRONMENTAL = "environmental"
    POWER = "power"
    CONNECTIVITY = "connectivity"
    DIMENSIONS = "dimensions"
    CERTIFICATIONS = "certifications"
    GENERAL = "general"


class IngestionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ─── Page-level Models ──────────────────────────────────────────────────────────

class TableCell(BaseModel):
    row: int
    col: int
    text: str
    is_header: bool = False


class ExtractedTable(BaseModel):
    page_number: int
    table_index: int                # Index of table on that page
    headers: List[str] = []
    rows: List[List[str]] = []
    raw_text: str = ""              # Fallback flat text representation
    bbox: Optional[List[float]] = None  # [x0, y0, x1, y1]

    def to_markdown(self) -> str:
        """Render table as markdown for embedding."""
        if not self.rows:
            return self.raw_text
        lines = []
        if self.headers:
            lines.append("| " + " | ".join(str(h) for h in self.headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(self.headers)) + " |")
            for row in self.rows:
                lines.append("| " + " | ".join(str(c) for c in row) + " |")
        else:
            for row in self.rows:
                lines.append("| " + " | ".join(str(c) for c in row) + " |")
        return "\n".join(lines)

    def to_flat_text(self) -> str:
        """Render table as key:value pairs for embedding."""
        parts = []
        if self.headers and self.rows:
            for row in self.rows:
                if (
                    len(self.headers) == 2
                    and len(row) >= 2
                    and self.headers[0].strip().lower() in {"parameter", "specification", "spec", "feature", "name"}
                    and self.headers[1].strip().lower() in {"value", "description", "setting"}
                    and row[0].strip()
                    and row[1].strip()
                ):
                    parts.append(f"{row[0].strip()}: {row[1].strip()}")
                    continue
                for h, v in zip(self.headers, row):
                    if h.strip() and v.strip():
                        parts.append(f"{h.strip()}: {v.strip()}")
        else:
            return self.raw_text
        return "\n".join(parts)


class PageContent(BaseModel):
    page_number: int                # 1-indexed
    raw_text: str = ""
    cleaned_text: str = ""
    tables: List[ExtractedTable] = []
    extraction_method: ExtractionMethod = ExtractionMethod.PDFPLUMBER
    ocr_confidence: Optional[float] = None  # Average Tesseract confidence
    is_scanned: bool = False
    has_text: bool = False


# ─── Vendor / Document-level Models ────────────────────────────────────────────

class VendorInfo(BaseModel):
    name: str = "Unknown"
    confidence: float = 0.0         # 0-1
    detection_method: str = ""      # "ocr_header", "text_search", "llm", "filename"
    raw_header_text: str = ""       # Raw OCR text from the top portion of page 1


class ModelSpec(BaseModel):
    """A single product model with its complete specification."""
    model_id: str                   # Unique within this datasheet
    model_name: str                 # Human-readable model name / part number
    vendor: str
    product_family: Optional[str] = None   # e.g. "FortiGate 200F Series"
    product_category: str = "Unkown" # e.g. "Next-Generation Firewall"
    category_confidence: float = 0.0
    description: str = ""

    # Raw spec data (before chunking)
    spec_sections: Dict[str, str] = {}    # section_name → text
    spec_tables: List[ExtractedTable] = []
    features: List[str] = []

    # Structured per-model specs (from comparison table splitting)
    specs: Dict[str, str] = {}            # e.g. {"firewall_throughput": "4 Gbps"}
    common_specs: Dict[str, str] = {}     # specs identical across all models in family

    # Source tracking
    source_pages: List[int] = []          # Which pages this model was found on
    source_file: str = ""

    # Extraction metadata
    extraction_confidence: float = 1.0   # How confident we are this is a real model
    identified_by: str = ""              # "llm", "regex", "section_header"


class DatasheetDocument(BaseModel):
    """Represents one fully-parsed OEM datasheet file."""
    doc_id: str                          # SHA256 of file content
    source_path: str
    filename: str
    vendor: VendorInfo
    page_count: int
    product_category: str = "Unknown"
    category_confidence: float = 0.0
    models: List[ModelSpec] = []         # One or more product models
    global_description: str = ""        # Text that applies to the whole doc
    extraction_method: ExtractionMethod = ExtractionMethod.PDFPLUMBER

    # Audit
    processed_at: datetime = Field(default_factory=lambda:datetime.now(timezone.utc))
    pipeline_version: str = "1.0.0"
    warnings: List[str] = []
    errors: List[str] = []

    @validator("models")
    def at_least_one_model(cls, v, values):
        # We don't hard-fail, just warn during processing
        return v


# ─── Vector-store Chunk Models ─────────────────────────────────────────────────

class DocumentChunk(BaseModel):
    """
    A single chunk that will be embedded and stored in ChromaDB.
    The metadata dict is what gets stored alongside the vector.
    """
    chunk_id: str                   # Globally unique ID
    text: str                       # Text to embed

    # Metadata (stored in ChromaDB, queryable)
    doc_id: str
    vendor: str
    model_name: str
    model_id: str
    product_family: Optional[str] = None
    product_category: Optional[str] = None
    chunk_type: ChunkType = ChunkType.GENERAL
    section_name: str = ""
    source_file: str = ""
    source_pages: List[int] = []
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    table_index: Optional[int] = None
    extraction_method: str = ""
    pipeline_version: str = "1.0.0"
    created_at: str = ""
    def to_chroma_metadata(self) -> Dict[str, Any]:
        """Flatten for ChromaDB (no nested objects, lists as comma-sep strings)."""
        return {
            "doc_id": self.doc_id,
            "vendor": self.vendor,
            "model_name": self.model_name,
            "model_id": self.model_id,
            "product_family": self.product_family or "",
            "product_category": self.product_category or "",
            "chunk_type": self.chunk_type.value,
            "section_name": self.section_name,
            "source_file": self.source_file,
            "source_pages": ",".join(map(str, self.source_pages)),
            "page_start": self.page_start or 0,
            "page_end": self.page_end or 0,
            "table_index": self.table_index or -1,
            "extraction_method": self.extraction_method,
            "pipeline_version": self.pipeline_version,
            "created_at": self.created_at,
        }


# ─── Pipeline Run Modelsu ────────────────────────────────────────────────────────

class FileIngestionResult(BaseModel):
    file_path: str
    doc_id: str = ""
    status: IngestionStatus = IngestionStatus.PENDING
    vendor: str = ""
    models_found: int = 0
    chunks_created: int = 0
    processing_time_seconds: float = 0.0
    warnings: List[str] = []
    error_message: str = ""


class PipelineRunResult(BaseModel):
    run_id: str
    started_at: datetime = Field(default_factory=lambda:datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    total_files: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    total_models_extracted: int = 0
    total_chunks_created: int = 0
    file_results: List[FileIngestionResult] = []

    @property
    def duration_seconds(self) -> float:
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0

class Requirement(BaseModel):
    requirement_id: str

    category: str

    requirement: str

    source_text: str

    mandatory: bool = True

    operator: Optional[str] = None

    value: Optional[str] = None

    unit: Optional[str] = None

    section: Optional[str] = None
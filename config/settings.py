"""
OEM Datasheet Ingestion Pipeline - Configuration Settings
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
from dotenv import load_dotenv
load_dotenv()

# ─── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
LOGS_DIR = BASE_DIR / "logs"

for d in [RAW_DIR, PROCESSED_DIR, VECTOR_STORE_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

@dataclass
class LLMConfig:
    provider: str = "local"
    model: str = "qwen3:8b"  # Set to the model name running on the local server
    base_url: str = "http://192.168.2.123:11434/v1"  # Defaulting to standard v1 API path (Ollama/LMStudio/vLLM)
    api_key: str = field(
        default_factory=lambda: os.getenv("LLM_API_KEY", "local")
    )

@dataclass
class OCRConfig:
    """Tesseract OCR settings."""
    lang: str = "eng"
    dpi: int = 300
    # PSM 3 = Fully automatic page segmentation, but no OSD
    # PSM 6 = Assume a single uniform block of text
    psm_default: int = 3
    psm_header: int = 6       # For vendor logo / header region
    oem: int = 3               # LSTM + Legacy (best accuracy)
    confidence_threshold: float = 30.0  # Min OCR confidence to accept text
    header_crop_fraction: float = 0.2  # Top 20% of page for vendor detection


@dataclass
class PDFConfig:
    """PDF extraction settings."""
    min_text_chars_per_page: int = 50   # Below this → assume scanned page
    table_extraction_strategy: str = "lines_strict"
    image_extraction_dpi: int = 150
    header_pages_to_check: int = 2      # Check first N pages for vendor info
    max_pages: Optional[int] = None     # None = process all pages


@dataclass
class ChunkingConfig:
    """Text chunking / splitting settings."""
    # Model-level spec chunks
    spec_chunk_size: int = 1200
    spec_chunk_overlap: int = 200

    # Table chunks — split large tables into rows/groups
    table_chunk_size: int = 1500
    table_chunk_overlap: int = 150

    # General description chunks
    general_chunk_size: int = 1000
    general_chunk_overlap: int = 150

    # Hard limit: if a single spec block exceeds this, split it
    max_single_chunk: int = 2000


@dataclass
class EmbeddingConfig:
    """Embedding model settings."""
    # Can also use: "all-MiniLM-L6-v2" (faster, less accurate)
    # or "BAAI/bge-large-en-v1.5" (more accurate, heavier)
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    device: str = "cpu"
    batch_size: int = 32
    normalize_embeddings: bool = True


@dataclass
class VectorStoreConfig:
    """ChromaDB vector store settings."""
    collection_name: str = "oem_datasheets"
    distance_metric: str = "cosine"
    persist_directory: str = str(VECTOR_STORE_DIR)


@dataclass
class ModelIdentificationConfig:
    """Heuristics for identifying models within a datasheet."""
    # Keywords that precede model names/numbers in datasheets
    model_header_keywords: List[str] = field(default_factory=lambda: [
        "model", "part number", "part no", "p/n", "sku", "ordering code",
        "product code", "device", "variant", "type", "series",
        "model number", "item", "catalog number", "cat. no",
    ])

    # Regex patterns for model number formats (common in tech/OEM)
    model_number_patterns: List[str] = field(default_factory=lambda: [
        r'\b[A-Z]{1,5}[-_]?\d{3,8}[A-Z0-9]{0,6}\b',   # e.g. PA-3050, FGT-200F
        r'\b[A-Z]{2,8}\d{2,6}[A-Z]?\b',                  # e.g. ASA5505, PA3050
        r'\b\d{3,6}[A-Z]{1,4}\d{0,4}\b',                 # e.g. 2960X, 3850
        r'\b[A-Z]{1,4}-\d{1,4}[A-Z]?-[A-Z0-9]{1,8}\b', # e.g. C-200-K9
    ])

    # Section titles that indicate the start of a new model specification block
    model_section_triggers: List[str] = field(default_factory=lambda: [
        "specifications", "technical specifications", "spec sheet",
        "product specifications", "features", "ordering information",
        "performance", "hardware specifications", "system specifications",
    ])

    # Minimum occurrences of model number pattern to consider it a real model
    min_model_occurrences: int = 1


@dataclass
class PipelineConfig:
    """Master pipeline configuration."""
    ocr: OCRConfig = field(default_factory=OCRConfig)
    pdf: PDFConfig = field(default_factory=PDFConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    model_id: ModelIdentificationConfig = field(default_factory=ModelIdentificationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    # Processing behaviour
    skip_existing: bool = True          # Skip PDFs already in vector DB
    save_intermediate: bool = True      # Save extracted JSON to PROCESSED_DIR
    parallel_workers: int = 2
    log_level: str = "INFO"

    # LLM config for advanced model identification
    use_llm_for_model_id: bool = True
    llm_model: str = "qwen3:8b"
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "local"))
    llm_base_url: str = "http://192.168.2.123:11434/v1"


# Singleton config instance
DEFAULT_CONFIG = PipelineConfig()

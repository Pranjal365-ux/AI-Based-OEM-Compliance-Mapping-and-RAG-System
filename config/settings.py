"""
OEM Datasheet Ingestion Pipeline - Configuration Settings
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

# ── Base paths ────────────────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).parent.parent
DATA_DIR          = BASE_DIR / "data"
requirements_json = BASE_DIR / "requirements"
RAW_DIR           = DATA_DIR / "raw"
PROCESSED_DIR     = DATA_DIR / "processed"
VECTOR_STORE_DIR  = DATA_DIR / "vector_store"
LOGS_DIR          = BASE_DIR / "logs"
RFP_DIR           = DATA_DIR / "requirements"   # JSON + per-RFP Chroma collections

for _d in (RAW_DIR, PROCESSED_DIR, VECTOR_STORE_DIR, LOGS_DIR, RFP_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class LLMConfig:
    provider: str = "local"
    # ── Reasoning model — used for compliance report generation where
    #    chain-of-thought genuinely helps answer quality.
    model: str = "qwen2.5:7b"
    base_url: str = "http://100.98.219.69:11434/v1"
    api_key: str = field(
        default_factory=lambda: os.getenv("LLM_API_KEY", "local")
    )

    # ── Fast extraction model — used for RFP requirement extraction where
    #    we need pure JSON output with no reasoning overhead.
    #    Any non-reasoning instruct model served by the same Ollama instance
    #    works here. Recommended options (pull whichever you have):
    #      • qwen2.5:7b        — fastest, very accurate JSON output
    #      • llama3.1:8b       — good alternative
    #      • mistral:7b        — lightweight option
    #    Set to None to fall back to `model` above (with /no_think suppression).
    extraction_model: Optional[str] = "qwen2.5:7b"


@dataclass
class OCRConfig:
    lang: str = "eng"
    dpi: int = 300
    psm_default: int = 3
    psm_header: int = 6
    oem: int = 3
    confidence_threshold: float = 30.0
    header_crop_fraction: float = 0.2


@dataclass
class PDFConfig:
    min_text_chars_per_page: int = 50
    table_extraction_strategy: str = "lines_strict"
    image_extraction_dpi: int = 150
    header_pages_to_check: int = 2
    max_pages: Optional[int] = None


@dataclass
class ChunkingConfig:
    """
    Chunk size tuning.

    Key insight: the original 300-600 char budgets were far too small for
    spec data.  A single spec table row ("Firewall throughput (appmix): 5 Gbps")
    is ~45 chars; fitting useful context requires 600-1200 chars minimum.

    These values produce roughly 15-40 chunks per typical 7-15 page datasheet
    and 3-8 chunks per model in a multi-model document.
    """
    # Target size for dense spec / table content (key:value pairs)
    spec_chunk_size: int = 1200
    spec_chunk_overlap: int = 0       # spec rows are self-contained

    # Target size for table content
    table_chunk_size: int = 1200
    table_chunk_overlap: int = 0

    # Target size for prose descriptions / feature lists
    general_chunk_size: int = 800
    general_chunk_overlap: int = 80   # small overlap for prose continuity

    # Hard limit: never exceed this (safety valve)
    max_single_chunk: int = 1600


@dataclass
class EmbeddingConfig:
    base_url: str = "http://100.98.219.69:11434"
    model_name: str = "bge-m3"
    batch_size: int = 32
    timeout_seconds: int = 300


@dataclass
class VectorStoreConfig:
    collection_name: str = "oem_datasheets"
    distance_metric: str = "cosine"
    persist_directory: str = str(VECTOR_STORE_DIR)


@dataclass
class RFPConfig:
    """Settings for RFP requirement extraction and storage."""
    # Where to persist everything related to a single RFP run
    output_dir: str = str(RFP_DIR)

    # The Chroma collection that holds embedded RFP requirements.
    # Each RFP gets its own sub-collection keyed by a sanitised filename
    # so multiple RFPs can coexist without clobbering each other.
    chroma_collection_prefix: str = "rfp_requirements"
    distance_metric: str = "cosine"

    # Extraction chunking
    chunk_size_chars: int = 2500

    # Token budget for the extraction LLM call.
    # With thinking suppressed, 3000 tokens comfortably covers one chunk's
    # worth of JSON requirement objects (typically 15-35 items × ~80 tokens
    # each = ~1200-2800 tokens).  We keep a safety margin above that.
    extraction_max_tokens: int = 3000

    # Parallel LLM workers for extraction (see note in rfp_extractor.py)
    max_workers: int = 2


@dataclass
class ModelIdentificationConfig:
    model_header_keywords: List[str] = field(default_factory=lambda: [
        "model", "part number", "part no", "p/n", "sku", "ordering code",
        "product code", "device", "variant", "type", "series",
        "model number", "item", "catalog number", "cat. no",
    ])

    model_number_patterns: List[str] = field(default_factory=lambda: [
        r'\b[A-Z]{1,5}[-_]?\d{3,8}[A-Z0-9]{0,6}\b',
        r'\b[A-Z]{2,8}\d{2,6}[A-Z]?\b',
        r'\b\d{3,6}[A-Z]{1,4}\d{0,4}\b',
        r'\b[A-Z]{1,4}-\d{1,4}[A-Z]?-[A-Z0-9]{1,8}\b',
    ])

    model_section_triggers: List[str] = field(default_factory=lambda: [
        "specifications", "technical specifications", "spec sheet",
        "product specifications", "features", "ordering information",
        "performance", "hardware specifications", "system specifications",
    ])

    min_model_occurrences: int = 1

    component_model_prefixes: List[str] = field(default_factory=lambda: [
        "FIM-", "FPM-", "SPM-", "FMC-", "FPC-", "FAP-",
    ])


@dataclass
class PipelineConfig:
    ocr:         OCRConfig         = field(default_factory=OCRConfig)
    pdf:         PDFConfig         = field(default_factory=PDFConfig)
    chunking:    ChunkingConfig    = field(default_factory=ChunkingConfig)
    embedding:   EmbeddingConfig   = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    rfp:         RFPConfig         = field(default_factory=RFPConfig)
    model_id:    ModelIdentificationConfig = field(
        default_factory=ModelIdentificationConfig
    )
    llm:         LLMConfig         = field(default_factory=LLMConfig)

    skip_existing: bool = True
    save_intermediate: bool = True
    parallel_workers: int = 8
    log_level: str = "INFO"

    use_llm_for_model_id: bool = True
    llm_model: str = "qwen2.5:7b"
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("LLM_API_KEY", "local")
    )
    llm_base_url: str = "http://100.98.219.69:11434/v1"


DEFAULT_CONFIG = PipelineConfig()

"""
Compliance Engine – Data Models
================================
All Pydantic v2 models used by the retrieval, matcher, ranker,
and report generator modules.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════════════════════════════

class ComplianceStatus(str, Enum):
    FULL           = "Full Match"
    PARTIAL        = "Partial Match"
    NO             = "No Match"             # Evidence contradicts or shows inadequacy
    NOT_FOUND      = "Not Found"            # No evidence in datasheet at all
    UNSUPPORTED    = "Unsupported"          # Product explicitly doesn't support this
    NOT_APPLICABLE = "Not Applicable"       # Requirement doesn't apply (procurement/commercial)


# ══════════════════════════════════════════════════════════════════════════════
# EVIDENCE  (Phase 3 – Semantic Search & Evidence Mapping)
# ══════════════════════════════════════════════════════════════════════════════

class EvidenceChunk(BaseModel):
    """
    A single OEM KB chunk that supports (or partially supports) a requirement.

    Carries source document name and page number so the report can cite
    the exact datasheet page — satisfying the Phase 5 evidence mapping spec.
    """
    chunk_id:       str
    text:           str
    score:          float           # cosine similarity 0–1
    vendor:         str
    model_name:     str
    product_family: str = ""
    chunk_type:     str = ""
    source_file:    str = ""        # absolute or relative path to OEM datasheet
    page_start:     int = 0         # page number within source_file (0 = unknown)


# ══════════════════════════════════════════════════════════════════════════════
# PER-REQUIREMENT RESULT  (Phase 4 – Compliance Mapping + Phase 5 – Evidence)
# ══════════════════════════════════════════════════════════════════════════════

class RequirementResult(BaseModel):
    """
    Compliance result for one extracted RFP requirement against one product.

    Fields
    ------
    status        : Full / Partial / No Match  (LLM or heuristic verdict)
    confidence    : 0–1 float, LLM-assigned or heuristic-derived
    justification : one-sentence explanation citing specific evidence
    gap           : what is missing (empty if Full Match)
    evidence      : top matching OEM KB chunks for audit trail
    """
    requirement_id: str
    requirement:    str
    category:       str
    mandatory:      bool

    # Numeric threshold extracted from the RFP requirement (if any)
    operator:       Optional[str] = None   # e.g. ">=" / "<="
    value:          Optional[str] = None   # e.g. "10"
    unit:           Optional[str] = None   # e.g. "Gbps"

    # Top evidence chunks retrieved from the KB (stored for the report)
    evidence:       List[EvidenceChunk] = []

    # LLM / heuristic verdict
    status:         ComplianceStatus = ComplianceStatus.NO
    confidence:     float = 0.0
    justification:  str   = ""
    gap:            str   = ""   # empty string = no gap (Full Match)


# ══════════════════════════════════════════════════════════════════════════════
# PER-PRODUCT RESULT  (Phase 4 – Compliance Mapping)
# ══════════════════════════════════════════════════════════════════════════════

class ProductScore(BaseModel):
    """
    Aggregated weighted compliance score for one OEM product/model.

    Scoring weights
    ---------------
    Mandatory requirements are weighted 2× optional ones.
    rank_key = 0.7 × mandatory_score + 0.3 × overall_score
      → products that miss mandatory reqs are pushed down even if they
        score well on optional ones.
    """
    vendor:         str
    model_name:     str
    product_family: str = ""

    # Requirement counts
    total_requirements: int = 0
    mandatory_count:    int = 0
    full_matches:       int = 0
    partial_matches:    int = 0
    no_matches:         int = 0

    # Weighted percentage scores (0–100)
    overall_score:   float = 0.0   # all requirements combined
    mandatory_score: float = 0.0   # mandatory requirements only
    optional_score:  float = 0.0   # optional requirements only

    # Detailed per-requirement results
    requirement_results: List[RequirementResult] = []

    # Mandatory requirements that are No/Partial (for Gap Analysis section)
    key_gaps: List[str] = []

    # Datasheet filenames that contributed evidence
    source_files: List[str] = []

    @property
    def rank_key(self) -> float:
        """Primary sort key for ranking products."""
        return 0.7 * self.mandatory_score + 0.3 * self.overall_score


# ══════════════════════════════════════════════════════════════════════════════
# FULL REPORT  (Phase 6 – Reporting)
# ══════════════════════════════════════════════════════════════════════════════

class ComplianceReport(BaseModel):
    """
    Top-level report returned by ComplianceEngine.run().

    Serialised to JSON and Markdown by compliance.reporter.
    """
    report_id:          str
    generated_at:       datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    rfp_source:         str = ""
    page_range:         Dict[str, int] = {}
    total_requirements: int = 0
    mandatory_count:    int = 0
    optional_count:     int = 0

    # Top-N ranked products (sorted by rank_key desc)
    top_products: List[ProductScore] = []

    # Pipeline metadata
    kb_chunks_searched: int = 0
    llm_model_used:     str = ""
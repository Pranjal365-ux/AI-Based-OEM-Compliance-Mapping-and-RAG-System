"""
Compliance Engine – Retrieval Module  (Phase 3: Semantic Search & Retrieval)
=============================================================================
Fetches the most relevant OEM KB chunks for each extracted requirement
using the existing VectorStoreManager (bge-m3 via Ollama, ChromaDB).

Design
------
- One semantic search per requirement against the OEM KB collection.
- Results are grouped by (vendor, model_name) so the ranker can see
  which product has the most supporting evidence before any LLM runs.
- search_for_requirement() adds a technical-context prefix to the query,
  improving embedding relevance for spec-style text.
- Chunks below MIN_SCORE are discarded as irrelevant.

CPU performance notes
---------------------
- bge-m3 embeddings are run by the VectorStoreManager; no GPU needed.
- ChromaDB HNSW index keeps searches fast even with large collections.
- TOP_K_PER_REQUIREMENT=8 gives the LLM enough context without bloating
  prompts and slowing down generation on the CPU-only host.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Tuple

from models.schemas import Requirement
from compliance.schemas import EvidenceChunk

logger = logging.getLogger(__name__)

# How many KB chunks to retrieve per requirement.
# Raised from 8 → 12: spec_text and spec_table chunks (model-specific numbers)
# often rank lower than general prose for feature-style requirements.
# More chunks ensures the numeric precheck and LLM both see actual spec values.
TOP_K_PER_REQUIREMENT = 12

# Minimum cosine similarity to keep a chunk as evidence.
# Chunks below this are almost certainly irrelevant.
MIN_SCORE = 0.30


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_evidence(
    requirements: List[Requirement],
    vector_store,                       # VectorStoreManager instance
    top_k:      int   = TOP_K_PER_REQUIREMENT,
    min_score:  float = MIN_SCORE,
) -> Dict[str, List[EvidenceChunk]]:
    """
    For each requirement perform a semantic search against the OEM KB
    and return the top matching chunks.

    Returns
    -------
    dict mapping requirement_id → List[EvidenceChunk]  (sorted by score desc)

    The returned chunks carry vendor/model metadata so the ranker can
    immediately identify which products appear most frequently.
    """
    evidence_map: Dict[str, List[EvidenceChunk]] = {}

    for req in requirements:
        query = _build_query(req)
        try:
            raw_results = vector_store.search_for_requirement(
                requirement_text=query,
                n_results=top_k,
            )
        except Exception as exc:
            logger.warning(f"Search failed for {req.requirement_id}: {exc}")
            evidence_map[req.requirement_id] = []
            continue

        chunks: List[EvidenceChunk] = []
        for r in raw_results:
            score = r.get("score", 0.0)
            if score < min_score:
                continue
            # page_start is stored as a top-level integer field in ChromaDB
            # metadata (key="page_start", int_value column).  VectorStoreManager
            # surfaces it as a top-level key in the result dict, NOT nested
            # inside a "metadata" sub-dict.  Falling back to the nested path
            # always returned 0, stripping page citations from every report.
            raw_page = r.get("page_start") or r.get("metadata", {}).get("page_start", 0)
            try:
                page_start = int(raw_page) if raw_page is not None else 0
            except (ValueError, TypeError):
                page_start = 0
            chunks.append(EvidenceChunk(
                chunk_id       = r.get("id", ""),
                text           = r.get("text", ""),
                score          = round(score, 4),
                vendor         = r.get("vendor", ""),
                model_name     = r.get("model_name", ""),
                product_family = r.get("product_family", ""),
                chunk_type     = r.get("chunk_type", ""),
                source_file    = r.get("source_file", ""),
                page_start     = page_start,
            ))

        # VectorStoreManager already returns results sorted by score desc
        evidence_map[req.requirement_id] = chunks

        if chunks:
            logger.debug(
                f"  {req.requirement_id}: {len(chunks)} chunks "
                f"(top score={chunks[0].score:.3f})"
            )
        else:
            logger.debug(f"  {req.requirement_id}: no evidence found")

    return evidence_map


def build_candidate_products(
    evidence_map: Dict[str, List[EvidenceChunk]],
) -> List[Tuple[str, str, str]]:
    """
    Scan all retrieved evidence chunks and return a ranked list of
    (vendor, model_name, product_family) tuples ordered by how many
    distinct requirements they appeared in.

    Purpose
    -------
    The ranker uses this list to decide which products warrant a full
    LLM compliance check, avoiding wasted calls on products with almost
    no evidence.  Products that appear in more requirements are ranked
    higher because they are more likely to be relevant.
    """
    appearance_count: Dict[Tuple[str, str, str], int] = defaultdict(int)

    for chunks in evidence_map.values():
        # Count each product once per requirement (not once per chunk)
        seen_this_req: set[Tuple[str, str, str]] = set()
        for chunk in chunks:
            if not chunk.vendor:
                continue
            key = (chunk.vendor, chunk.model_name, chunk.product_family)
            if key not in seen_this_req:
                appearance_count[key] += 1
                seen_this_req.add(key)

    # Sort descending by appearance count
    ranked = sorted(appearance_count.items(), key=lambda x: -x[1])
    return [product for product, _ in ranked]


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_query(req: Requirement) -> str:
    """
    Construct a search query from a requirement object.

    For numeric requirements the threshold is appended so the embedding
    captures both the concept AND the scale (e.g. "throughput >= 10 Gbps").
    This improves retrieval of spec sections that mention specific numbers.
    """
    base = req.requirement.strip()
    if req.value and req.unit and req.value not in ("true", ""):
        return f"{base} {req.operator or '>='} {req.value} {req.unit}"
    return base
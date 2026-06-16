"""
Compliance Engine – Ranker Module  (Phase 4: Compliance Mapping Engine)
========================================================================
Orchestrates the matcher across all candidate products and returns
the top-N ranked ProductScore objects.

Score formula
-------------
  Full Match    = 1.0 × weight
  Partial Match = 0.5 × weight
  No Match      = 0.0 × weight

  mandatory weight = 2.0   (penalises mandatory gaps more heavily)
  optional  weight = 1.0

  product_score = sum(weighted_earned) / sum(weighted_possible) × 100

Rank key: 70 % mandatory score + 30 % overall score
  → Products that miss mandatory requirements are pushed down even if
    they score well on optional ones.

CPU performance notes
---------------------
- Requirements for a single product are evaluated in parallel using
  ThreadPoolExecutor(max_workers=EVAL_WORKERS).  EVAL_WORKERS=2 matches
  OLLAMA_NUM_PARALLEL on a CPU-only workstation; raising it beyond the
  Ollama parallel limit causes queuing without speed gain.
- Products are evaluated sequentially to avoid saturating the Ollama queue.
- Progress is printed live per product so the operator can monitor runtime.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from models.schemas import Requirement
from compliance.schemas import (
    ComplianceStatus,
    EvidenceChunk,
    ProductScore,
    RequirementResult,
)
from compliance import matcher

logger = logging.getLogger(__name__)

MANDATORY_WEIGHT = 2.0
OPTIONAL_WEIGHT  = 1.0
FULL_SCORE       = 1.0
PARTIAL_SCORE    = 0.5
NO_SCORE         = 0.0

# Match OLLAMA_NUM_PARALLEL on your workstation.
# 2 = safe default for a single CPU-only Ollama instance.
EVAL_WORKERS = 1


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def rank_products(
    requirements:  List[Requirement],
    evidence_map:  Dict[str, List[EvidenceChunk]],
    candidates:    List[Tuple[str, str, str]],
    top_n:         int = 3,
) -> List[ProductScore]:
    """
    Run compliance evaluation for every candidate product and return
    the top_n ranked ProductScore objects (sorted by rank_key desc).

    Parameters
    ----------
    requirements  : full list of extracted RFP requirements
    evidence_map  : requirement_id → list of EvidenceChunk (from retrieval)
    candidates    : (vendor, model_name, product_family) sorted by evidence count
    top_n         : how many top products to return
    """
    if not candidates:
        logger.warning("No candidate products found in the knowledge base.")
        return []

    scored: List[ProductScore] = []
    total = len(candidates)

    for idx, (vendor, model_name, product_family) in enumerate(candidates, 1):
        if not vendor or not model_name:
            continue

        t0 = time.time()
        print(
            f"\n  [{idx}/{total}] Evaluating: {vendor} – {model_name}",
            flush=True,
        )

        product_score = _evaluate_product(
            vendor, model_name, product_family,
            requirements, evidence_map,
        )
        elapsed = time.time() - t0
        scored.append(product_score)

        print(
            f"    → Score: {product_score.overall_score:.1f}% overall | "
            f"{product_score.mandatory_score:.1f}% mandatory | "
            f"✅{product_score.full_matches} "
            f"⚠️{product_score.partial_matches} "
            f"❌{product_score.no_matches} "
            f"({elapsed:.0f}s)",
            flush=True,
        )

    # Sort by rank_key (70% mandatory + 30% overall) descending
    scored.sort(key=lambda p: p.rank_key, reverse=True)
    top = scored[:top_n]

    logger.info(
        f"Top {len(top)} products: "
        + ", ".join(
            f"{p.vendor} {p.model_name} ({p.overall_score:.1f}%)"
            for p in top
        )
    )
    return top


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL — SINGLE-PRODUCT EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate_product(
    vendor:         str,
    model_name:     str,
    product_family: str,
    requirements:   List[Requirement],
    evidence_map:   Dict[str, List[EvidenceChunk]],
) -> ProductScore:
    """
    Evaluate all requirements for one product, parallelising per-requirement
    LLM calls across EVAL_WORKERS threads.
    """
    product = ProductScore(
        vendor=vendor,
        model_name=model_name,
        product_family=product_family,
    )

    # Collect source files up front (no LLM involved)
    source_files: set[str] = set()
    for req in requirements:
        for chunk in evidence_map.get(req.requirement_id, []):
            if chunk.vendor == vendor and chunk.model_name == model_name:
                if chunk.source_file:
                    source_files.add(Path(chunk.source_file).name)

    # ── Parallel requirement evaluation ───────────────────────────────────────
    req_results: Dict[str, RequirementResult] = {}

    with ThreadPoolExecutor(max_workers=EVAL_WORKERS) as pool:
        future_to_req = {
            pool.submit(
                matcher.evaluate_requirement,
                req,
                evidence_map.get(req.requirement_id, []),
                vendor,
                model_name,
            ): req
            for req in requirements
        }

        done = 0
        for future in as_completed(future_to_req):
            req  = future_to_req[future]
            done += 1
            try:
                result = future.result()
            except Exception as exc:
                logger.warning(
                    f"    Matcher failed for {req.requirement_id}: {exc} — defaulting No Match"
                )
                result = RequirementResult(
                    requirement_id = req.requirement_id,
                    requirement    = req.requirement,
                    category       = req.category,
                    mandatory      = req.mandatory,
                    status         = ComplianceStatus.NO,
                    confidence     = 0.0,
                    justification  = f"Evaluation error: {exc}",
                    gap            = req.requirement,
                )
            req_results[req.requirement_id] = result

            # Live progress indicator every 10 requirements
            if done % 10 == 0 or done == len(requirements):
                print(f"    {done}/{len(requirements)} requirements evaluated…", flush=True)

    # ── Aggregate scores in original requirement order ─────────────────────────
    mandatory_earned   = 0.0
    mandatory_possible = 0.0
    optional_earned    = 0.0
    optional_possible  = 0.0

    for req in requirements:
        req_result = req_results.get(req.requirement_id)
        if not req_result:
            continue

        product.requirement_results.append(req_result)
        product.total_requirements += 1

        weight = MANDATORY_WEIGHT if req.mandatory else OPTIONAL_WEIGHT
        earned = _status_to_score(req_result.status) * weight

        if req.mandatory:
            product.mandatory_count  += 1
            mandatory_possible       += weight
            mandatory_earned         += earned
        else:
            optional_possible += weight
            optional_earned   += earned

        if req_result.status == ComplianceStatus.FULL:
            product.full_matches += 1
        elif req_result.status == ComplianceStatus.PARTIAL:
            product.partial_matches += 1
        else:
            product.no_matches += 1
            if req.mandatory:
                product.key_gaps.append(req.requirement)

    product.overall_score   = _pct(mandatory_earned + optional_earned,
                                   mandatory_possible + optional_possible)
    product.mandatory_score = _pct(mandatory_earned,  mandatory_possible)
    product.optional_score  = _pct(optional_earned,   optional_possible)
    product.source_files    = sorted(source_files)
    return product


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _status_to_score(status: ComplianceStatus) -> float:
    if status == ComplianceStatus.FULL:    return FULL_SCORE
    if status == ComplianceStatus.PARTIAL: return PARTIAL_SCORE
    return NO_SCORE


def _pct(earned: float, possible: float) -> float:
    if possible == 0:
        return 0.0
    return round((earned / possible) * 100, 1)  
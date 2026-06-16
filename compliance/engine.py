"""
Compliance Engine – Main Orchestrator
=======================================
Implements Phases 3–6 of the OEM Compliance Mapping System:

  Phase 3 – Semantic Search & Retrieval
  Phase 4 – Compliance Mapping Engine
  Phase 5 – Evidence Mapping
  Phase 6 – Reporting

Usage
-----
    from compliance.engine import ComplianceEngine
    from knowledge_base.vector_store import VectorStoreManager
    from config.settings import DEFAULT_CONFIG

    vs = VectorStoreManager(DEFAULT_CONFIG.vector_store, DEFAULT_CONFIG.embedding)
    vs.initialize()
    vs.load_embedder()

    engine = ComplianceEngine(vector_store=vs, top_n=3)

    # Option A – from already-extracted Requirement objects
    from models.schemas import Requirement
    reqs = [...]
    report = engine.run(requirements=reqs, rfp_source="my_rfp.pdf")

    # Option B – from a requirements JSON file written by rfp_extractor
    report = engine.run_from_json("data/requirements/my_rfp_pp1-10.json")

    print(f"Top product: {report.top_products[0].vendor} {report.top_products[0].model_name}")
    print(f"Score: {report.top_products[0].overall_score:.1f}%")
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from config.settings import DEFAULT_CONFIG
from models.schemas import Requirement
from knowledge_base.vector_store import VectorStoreManager

from compliance import retrieval, ranker, reporter
from compliance.schemas import ComplianceReport

logger = logging.getLogger(__name__)


class ComplianceEngine:
    """
    Orchestrates the full compliance pipeline:
      retrieval → candidate identification → ranking → reporting.

    Parameters
    ----------
    vector_store : VectorStoreManager
        Already-initialized OEM knowledge base (ChromaDB + bge-m3).
    top_n : int
        Number of top products to include in the report (default 3).
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        top_n: int = 3,
    ):
        self.vs    = vector_store
        self.top_n = top_n

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC ENTRY POINTS
    # ══════════════════════════════════════════════════════════════════════════

    def run(
        self,
        requirements: List[Requirement],
        rfp_source:   str = "",
        page_range:   Optional[dict] = None,
    ) -> ComplianceReport:
        """
        Run the full compliance pipeline on a list of Requirement objects.

        Pipeline steps
        --------------
        1. [Phase 3] Retrieve top KB chunks for every requirement
           (semantic search via VectorStoreManager / bge-m3 / ChromaDB).
        2. [Phase 3] Identify candidate products from retrieved evidence.
        3. [Phase 4+5] For each candidate, run LLM compliance check per
           requirement with evidence-backed verdicts.
        4. [Phase 4] Rank products by weighted mandatory + overall score.
        5. [Phase 6] Generate JSON + Markdown compliance report.

        Parameters
        ----------
        requirements : list of Requirement objects extracted from the RFP
        rfp_source   : display name / path of the source RFP (for the report)
        page_range   : {'start': int, 'end': int} of the analysed RFP section
        """
        if not requirements:
            logger.warning("No requirements provided — nothing to evaluate.")
            return ComplianceReport(
                report_id="empty",
                rfp_source=rfp_source,
            )

        mandatory_count = sum(1 for r in requirements if r.mandatory)
        logger.info(
            f"\n{'='*60}\n"
            f"COMPLIANCE ENGINE\n"
            f"  Requirements : {len(requirements)} "
            f"({mandatory_count} mandatory)\n"
            f"  Top-N        : {self.top_n}\n"
            f"{'='*60}"
        )

        # ── Step 1: Retrieve evidence (Phase 3) ───────────────────────────────
        logger.info("\n[1/4] Retrieving evidence from OEM knowledge base…")
        evidence_map = retrieval.retrieve_evidence(requirements, self.vs)
        total_chunks = sum(len(v) for v in evidence_map.values())
        logger.info(f"  → {total_chunks} evidence chunks retrieved")

        # ── Step 2: Identify candidate products ───────────────────────────────
        logger.info("\n[2/4] Identifying candidate products…")
        candidates = retrieval.build_candidate_products(evidence_map)
        logger.info(f"  → {len(candidates)} candidate product(s) found")
        for i, (v, m, f) in enumerate(candidates[:10], 1):
            logger.info(f"     {i}. {v} – {m}" + (f" ({f})" if f else ""))

        if not candidates:
            logger.warning("Knowledge base returned no matching products.")
            return ComplianceReport(
                report_id="no_candidates",
                rfp_source=rfp_source,
                page_range=page_range or {},
                total_requirements=len(requirements),
                mandatory_count=mandatory_count,
                optional_count=len(requirements) - mandatory_count,
            )

        # ── Step 3: Rank products (Phases 4 + 5) ──────────────────────────────
        logger.info("\n[3/4] Evaluating compliance for top candidates…")
        # Cap candidates to top_n × 3 to keep runtime manageable.
        # Candidates are pre-sorted by evidence frequency, so the most
        # relevant products are always evaluated first.
        max_candidates = min(len(candidates), self.top_n * 3)
        top_products = ranker.rank_products(
            requirements = requirements,
            evidence_map = evidence_map,
            candidates   = candidates[:max_candidates],
            top_n        = self.top_n,
        )

        # ── Step 4: Generate report (Phase 6) ─────────────────────────────────
        logger.info("\n[4/4] Generating compliance report…")
        report = reporter.generate_report(
            requirements = requirements,
            top_products = top_products,
            rfp_source   = rfp_source,
            page_range   = page_range,
            kb_chunks    = total_chunks,
        )

        logger.info(
            f"\n{'='*60}\n"
            f"REPORT COMPLETE  (ID: {report.report_id})\n"
            f"{'='*60}"
        )
        return report

    def run_from_json(
        self,
        json_path: str,
        top_n: Optional[int] = None,
    ) -> ComplianceReport:
        """
        Load requirements from a JSON file written by rfp_extractor.run()
        and run the full compliance pipeline.

        Expected JSON format
        --------------------
        {
          "source_file": "path/to/rfp.pdf",
          "page_range": {"start": 1, "end": 10},
          "requirements": [{"requirement_id": ..., ...}, ...]
        }
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Requirements JSON not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        reqs = [Requirement(**r) for r in data.get("requirements", [])]
        if not reqs:
            raise ValueError(f"No requirements found in {path.name}")

        logger.info(f"Loaded {len(reqs)} requirements from {path.name}")

        if top_n:
            self.top_n = top_n

        return self.run(
            requirements = reqs,
            rfp_source   = data.get("source_file", str(path)),
            page_range   = data.get("page_range"),
        )
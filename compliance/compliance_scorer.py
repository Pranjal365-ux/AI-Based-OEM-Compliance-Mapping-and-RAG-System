# rfp/compliance_scorer.py
"""
Compliance Scorer & Gap Report Generator
=========================================
Matches normalized RFP capability statements against an OEM knowledge-base
(ChromaDB) to produce per-capability compliance verdicts and a gap report.

Architecture
------------
For each RFP capability statement:
  1. Embed the capability statement (BGE-M3 via Ollama, consistent with OEM KB)
  2. Vector search OEM KB → top-5 most relevant chunks
  3. LLM scores compliance: {compliant, confidence, evidence, gap}
  4. Results aggregated into a structured gap report

CPU constraint: all LLM scoring calls are SERIAL (no ThreadPoolExecutor).
A Qwen 8b call for a short compliance verdict takes ~8–15 s on CPU. With
~50 capabilities × N vendors, this is a batch job, not interactive.

Inputs
------
  requirements_json  — output of RFPRequirementExtractor.run()
                       (data/requirements.json by default)
  vendor_kb_path     — path to the OEM KB ChromaDB
                       (separate from the RFP requirements ChromaDB)
  vendor_name        — human-readable vendor name for the report

Output
------
  gap_report.json    — per-capability compliance results + summary
  gap_report.md      — human-readable markdown gap report
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from config.settings import DEFAULT_CONFIG
from services.embedding_service import ChromaBGEM3EmbeddingFunction, EmbeddingService

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

COMPLIANCE_MAX_TOKENS = 200   # short: one JSON object, one sentence per field
TOP_K_CHUNKS          = 5     # OEM KB chunks retrieved per capability
TOP_K_FOR_SCORING     = 3     # chunks passed to LLM (top 3 of 5)

# Parses "Page 12" (produced by RFPRequirementExtractor for LLM-derived
# requirements) and "Chunk-2 (pp.3-5)" (the fallback `section` label used
# when no page number was returned by the LLM).
_PAGE_RE       = re.compile(r"Page\s+(\d+)", re.IGNORECASE)
_PAGE_RANGE_RE = re.compile(r"pp\.(\d+)-(\d+)", re.IGNORECASE)


# ══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ComplianceResult:
    capability_id:        str
    capability_statement: str
    category:             str
    topic:                str
    mandatory:            bool
    pages:                List[int]

    normalized_by:        str              = ""
    source_text:          str              = ""
    value:                str              = ""
    unit:                 Optional[str]    = None
    operator:             str              = ""

    compliant:            Optional[bool]   = None   # None = undetermined
    confidence:           float            = 0.0
    evidence:             Optional[str]    = None
    gap:                  Optional[str]    = None
    oem_chunks_used:      List[str]        = field(default_factory=list)
    scoring_error:        Optional[str]    = None

    @property
    def status(self) -> str:
        if self.compliant is None or self.scoring_error:
            return "UNDETERMINED"
        if self.compliant and self.confidence >= 0.7:
            return "COMPLIANT"
        if self.compliant and self.confidence >= 0.4:
            return "LIKELY_COMPLIANT"
        if not self.compliant and self.confidence >= 0.7:
            return "NON_COMPLIANT"
        return "PARTIAL"

    def to_dict(self) -> dict:
        return {
            "capability_id":        self.capability_id,
            "capability_statement": self.capability_statement,
            "category":             self.category,
            "topic":                self.topic,
            "mandatory":            self.mandatory,
            "pages":                self.pages,
            "source_text":          self.source_text,
            "value":                self.value,
            "unit":                 self.unit,
            "operator":             self.operator,
            "status":               self.status,
            "compliant":            self.compliant,
            "confidence":           self.confidence,
            "evidence":             self.evidence,
            "gap":                  self.gap,
            "oem_chunks_used":      self.oem_chunks_used,
            "scoring_error":        self.scoring_error,
        }


@dataclass
class CategoryReport:
    category:       str
    vendor:         str
    results:        List[ComplianceResult] = field(default_factory=list)

    @property
    def total(self)           -> int: return len(self.results)
    @property
    def mandatory_count(self) -> int: return sum(1 for r in self.results if r.mandatory)
    @property
    def compliant_count(self) -> int:
        return sum(1 for r in self.results if r.status in ("COMPLIANT", "LIKELY_COMPLIANT"))
    @property
    def non_compliant_mandatory(self) -> List[ComplianceResult]:
        return [r for r in self.results
                if r.mandatory and r.status in ("NON_COMPLIANT", "UNDETERMINED")]
    @property
    def compliance_rate(self) -> float:
        return self.compliant_count / self.total if self.total else 0.0
    @property
    def mandatory_compliance_rate(self) -> float:
        mc = self.mandatory_count
        if not mc:
            return 1.0
        passed = sum(1 for r in self.results
                     if r.mandatory and r.status in ("COMPLIANT", "LIKELY_COMPLIANT"))
        return passed / mc


# ══════════════════════════════════════════════════════════════════════════════
# COMPLIANCE SCORER
# ══════════════════════════════════════════════════════════════════════════════

class ComplianceScorer:
    """
    Scores RFP capabilities against an OEM knowledge-base.

    Usage
    -----
        scorer = ComplianceScorer(
            llm            = llm,
            vendor_kb_path = "data/oem_chroma_db",
            vendor_name    = "Radware",
        )
        report = scorer.score_requirements_file(
            requirements_json = "data/requirements.json",
            output_dir        = "data/reports/radware",
        )
    """

    _THINK_RE   = re.compile(r"<think>.*?</think>", re.DOTALL)
    _FENCE_RE   = re.compile(r"```(?:json)?\s*([\s\S]*?)```")
    _JSON_OBJ   = re.compile(r"\{[\s\S]*\}")

    def __init__(
        self,
        llm,
        vendor_kb_path:     str,
        vendor_name:        str,
        kb_collection_name: str = DEFAULT_CONFIG.paths.oem_kb_collection,
        top_k:              int = TOP_K_CHUNKS,
    ):
        self._llm              = llm
        self._vendor_kb_path   = vendor_kb_path
        self._vendor_name      = vendor_name
        self._kb_collection    = kb_collection_name
        self._top_k            = top_k
        self._collection       = None   # lazy-loaded
        self._embedder         = EmbeddingService()

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    def score_requirements_file(
        self,
        requirements_json: str,
        output_dir:        str = "data/reports",
    ) -> dict:
        """
        Main entry point. Loads requirements JSON, scores every capability,
        writes gap_report.json and gap_report.md, returns the report dict.
        """
        with open(requirements_json, "r", encoding="utf-8") as f:
            rfp_data = json.load(f)

        source_file = rfp_data.get("source_file", "unknown")
        products    = rfp_data.get("products", [])

        print(f"\n{'='*60}")
        print(f"COMPLIANCE SCORING")
        print(f"  RFP file   : {source_file}")
        print(f"  Vendor     : {self._vendor_name}")
        print(f"  Categories : {len(products)}")
        total_reqs = sum(p.get("requirement_count", len(p.get("requirements", [])))
                         for p in products)
        print(f"  Requirements to score: {total_reqs}")
        print(f"{'='*60}")

        category_reports: List[CategoryReport] = []
        pipeline_start = time.monotonic()

        for product in products:
            cat  = product["category"]
            reqs = product.get("requirements", [])
            if not reqs:
                continue

            print(f"\n--- {cat} ({len(reqs)} requirements) ---")
            cat_start = time.monotonic()
            cat_report = CategoryReport(category=cat, vendor=self._vendor_name)

            for i, req in enumerate(reqs):
                cap_id    = req.get("requirement_id") or f"{cat}-{i + 1:04d}"
                statement = req.get("requirement", "")
                pages     = self._extract_pages_from_section(req.get("section", ""))

                print(f"  [{i+1}/{len(reqs)}] {cap_id}: {statement[:70]}…")

                result = self._score_capability(
                    capability_id        = cap_id,
                    capability_statement = statement,
                    category             = cat,
                    topic                = req.get("category", ""),
                    mandatory            = bool(req.get("mandatory", True)),
                    pages                = pages,
                    source_text          = req.get("source_text", ""),
                    operator             = req.get("operator", ""),
                    value                = req.get("value", ""),
                    unit                 = req.get("unit"),
                )
                cat_report.results.append(result)

                status_sym = {"COMPLIANT": "✓", "LIKELY_COMPLIANT": "~",
                               "NON_COMPLIANT": "✗", "PARTIAL": "?",
                               "UNDETERMINED": "?"}.get(result.status, "?")
                print(f"         → {status_sym} {result.status} (conf: {result.confidence:.2f})")

            category_reports.append(cat_report)
            self._print_category_summary(cat_report)
            cat_elapsed = time.monotonic() - cat_start
            print(f"    Time for {cat}: {cat_elapsed:.1f}s")

        # Build full report
        report = self._build_report(
            rfp_source       = source_file,
            category_reports = category_reports,
        )

        # Write outputs
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "gap_report.json")
        md_path   = os.path.join(output_dir, "gap_report.md")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nWrote {json_path}")

        md = self._render_markdown(report, category_reports)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Wrote {md_path}")

        total_elapsed = time.monotonic() - pipeline_start
        print(f"\n=== COMPLIANCE SCORING DONE in {total_elapsed:.1f}s "
              f"({total_elapsed / 60:.1f} min) ===")

        return report

    # ──────────────────────────────────────────────────────────────────────────
    # CAPABILITY SCORING
    # ──────────────────────────────────────────────────────────────────────────

    def _score_capability(
        self,
        capability_id:        str,
        capability_statement: str,
        category:             str,
        topic:                str,
        mandatory:            bool,
        pages:                List[int],
        source_text:          str = "",
        operator:             str = "",
        value:                str = "",
        unit:                 Optional[str] = None,
        normalized_by:        str = "",
    ) -> ComplianceResult:
        result = ComplianceResult(
            capability_id        = capability_id,
            capability_statement = capability_statement,
            category             = category,
            topic                = topic,
            mandatory            = mandatory,
            pages                = pages,
            normalized_by        = normalized_by,
            source_text          = source_text,
            operator             = operator,
            value                = value,
            unit                 = unit,
        )

        # Step 1: retrieve relevant OEM KB chunks
        try:
            oem_chunks = self._retrieve_oem_chunks(capability_statement, category)
        except Exception as exc:
            result.scoring_error = f"OEM retrieval failed: {exc}"
            logger.warning(f"OEM retrieval failed for {capability_id}: {exc}")
            return result

        if not oem_chunks:
            result.scoring_error = "No OEM KB chunks found"
            result.gap           = "No vendor documentation found for this capability area"
            result.compliant     = False
            result.confidence    = 0.9
            return result

        result.oem_chunks_used = [c[:200] for c in oem_chunks[:TOP_K_FOR_SCORING]]

        # Step 2: LLM compliance scoring
        try:
            verdict = self._llm_score(
                capability   = capability_statement,
                oem_chunks   = oem_chunks[:TOP_K_FOR_SCORING],
                vendor       = self._vendor_name,
                category     = category,
                operator     = operator,
                value        = value,
                unit         = unit,
            )
            result.compliant   = verdict.get("compliant")
            result.confidence  = float(verdict.get("confidence", 0.5))
            result.evidence    = verdict.get("evidence")
            result.gap         = verdict.get("gap")
        except Exception as exc:
            result.scoring_error = f"LLM scoring failed: {exc}"
            logger.warning(f"LLM scoring failed for {capability_id}: {exc}")

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # OEM KB RETRIEVAL
    # ──────────────────────────────────────────────────────────────────────────

    def _retrieve_oem_chunks(
        self, capability_statement: str, category: str
    ) -> List[str]:
        """
        Vector search the OEM KB ChromaDB for chunks relevant to the capability.
        Optionally filters by category metadata if available.
        """
        collection = self._get_collection()
        if collection is None:
            raise RuntimeError("OEM KB collection not available")

        embedding = self._embed_text(capability_statement)
        if embedding is None:
            raise RuntimeError("Could not embed capability statement")

        # Try category-filtered search first; fall back to global if too few results
        try:
            results = collection.query(
                query_embeddings = [embedding],
                n_results        = self._top_k,
                where            = {"category": category},
            )
            docs = results.get("documents", [[]])[0]
            if len(docs) < 2:
                # Not enough filtered results — try without filter
                results = collection.query(
                    query_embeddings = [embedding],
                    n_results        = self._top_k,
                )
                docs = results.get("documents", [[]])[0]
        except Exception:
            results = collection.query(
                query_embeddings = [embedding],
                n_results        = self._top_k,
            )
            docs = results.get("documents", [[]])[0]

        return [d for d in docs if d]

    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self._vendor_kb_path)
            embed_fn = self._build_embedding_function()
            self._collection = client.get_collection(
                name             = self._kb_collection,
                embedding_function = embed_fn,
            )
            return self._collection
        except Exception as exc:
            logger.error(f"Could not open OEM KB at {self._vendor_kb_path}: {exc}")
            return None

    def _embed_text(self, text: str) -> Optional[List[float]]:
        """Embed using BGE-M3 via Ollama. Returns embedding vector or None."""
        try:
            import requests
            resp = requests.post(
                OLLAMA_EMBED_URL,
                json    = {"model": OLLAMA_EMBED_MODEL, "prompt": text},
                timeout = 30,
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
        except Exception as exc:
            logger.warning(f"Ollama embed failed: {exc}")
            return None

    def _build_embedding_function(self):
        """Return the same embedding function used when building the KB."""
        try:
            import requests
            from chromadb import EmbeddingFunction, Documents, Embeddings

            resp = requests.post(
                OLLAMA_EMBED_URL,
                json={"model": OLLAMA_EMBED_MODEL, "prompt": "test"},
                timeout=5,
            )
            if resp.status_code == 200:
                class OllamaBGEM3(EmbeddingFunction):
                    def __call__(self, input: Documents) -> Embeddings:
                        import requests as _req
                        return [
                            _req.post(OLLAMA_EMBED_URL,
                                      json={"model": OLLAMA_EMBED_MODEL, "prompt": d},
                                      timeout=30).json()["embedding"]
                            for d in input
                        ]
                return OllamaBGEM3()
        except Exception:
            pass

        from chromadb.utils import embedding_functions
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # LLM COMPLIANCE SCORING  (one call per capability — short output)
    # ──────────────────────────────────────────────────────────────────────────

    def _llm_score(
        self,
        capability:  str,
        oem_chunks:  List[str],
        vendor:      str,
        category:    str,
    ) -> dict:
        chunks_text = "\n\n---\n\n".join(
            f"[Evidence {i+1}]\n{chunk[:600]}"
            for i, chunk in enumerate(oem_chunks)
        )

        prompt = (
            f"You are evaluating whether {vendor}'s {category} product meets an RFP requirement.\n\n"
            f"RFP REQUIREMENT:\n{capability}\n\n"
            f"VENDOR DOCUMENTATION:\n{chunks_text}\n\n"
            "Evaluate compliance strictly. Respond ONLY with this JSON object:\n"
            "{\n"
            '  "compliant": true or false,\n'
            '  "confidence": 0.0 to 1.0,\n'
            '  "evidence": "one sentence from vendor docs that supports compliance, or null if non-compliant",\n'
            '  "gap": "one sentence describing what is missing or unproven, or null if fully compliant"\n'
            "}\n"
            "No explanation, no markdown, no preamble."
        )

        try:
            response = self._llm.generate(prompt, max_tokens=COMPLIANCE_MAX_TOKENS)
        except Exception as exc:
            raise RuntimeError(f"LLM call failed: {exc}")

        return self._parse_verdict(response)

    def _parse_verdict(self, response: str) -> dict:
        """Parse LLM compliance verdict, with fallback extraction."""
        cleaned = self._THINK_RE.sub("", response).strip()

        # Try fence-wrapped JSON first
        fence_match = self._FENCE_RE.search(cleaned)
        if fence_match:
            try:
                return json.loads(fence_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try raw JSON object
        obj_match = self._JSON_OBJ.search(cleaned)
        if obj_match:
            try:
                return json.loads(obj_match.group(0))
            except json.JSONDecodeError:
                pass

        # Last resort: scan for key patterns
        verdict: dict = {}
        if re.search(r'"compliant"\s*:\s*true', cleaned, re.IGNORECASE):
            verdict["compliant"]  = True
            verdict["confidence"] = 0.6
        elif re.search(r'"compliant"\s*:\s*false', cleaned, re.IGNORECASE):
            verdict["compliant"]  = False
            verdict["confidence"] = 0.6

        conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', cleaned)
        if conf_match:
            try:
                verdict["confidence"] = float(conf_match.group(1))
            except ValueError:
                pass

        return verdict

    # ──────────────────────────────────────────────────────────────────────────
    # REPORT BUILDING
    # ──────────────────────────────────────────────────────────────────────────

    def _build_report(
        self,
        rfp_source:       str,
        category_reports: List[CategoryReport],
    ) -> dict:
        all_results = [r for cr in category_reports for r in cr.results]
        mandatory   = [r for r in all_results if r.mandatory]

        return {
            "meta": {
                "rfp_source":   rfp_source,
                "vendor":       self._vendor_name,
                "generated_at": datetime.utcnow().isoformat() + "Z",
            },
            "summary": {
                "total_capabilities":       len(all_results),
                "mandatory_capabilities":   len(mandatory),
                "compliant":                sum(1 for r in all_results
                                               if r.status in ("COMPLIANT", "LIKELY_COMPLIANT")),
                "non_compliant":            sum(1 for r in all_results
                                               if r.status == "NON_COMPLIANT"),
                "partial":                  sum(1 for r in all_results
                                               if r.status == "PARTIAL"),
                "undetermined":             sum(1 for r in all_results
                                               if r.status == "UNDETERMINED"),
                "mandatory_compliance_pct": round(
                    sum(1 for r in mandatory
                        if r.status in ("COMPLIANT", "LIKELY_COMPLIANT"))
                    / len(mandatory) * 100 if mandatory else 100, 1
                ),
                "overall_compliance_pct":   round(
                    sum(1 for r in all_results
                        if r.status in ("COMPLIANT", "LIKELY_COMPLIANT"))
                    / len(all_results) * 100 if all_results else 0, 1
                ),
            },
            "categories": [
                {
                    "category":                  cr.category,
                    "capability_count":          cr.total,
                    "mandatory_count":           cr.mandatory_count,
                    "compliance_rate_pct":       round(cr.compliance_rate * 100, 1),
                    "mandatory_compliance_pct":  round(cr.mandatory_compliance_rate * 100, 1),
                    "results":                   [r.to_dict() for r in cr.results],
                }
                for cr in category_reports
            ],
        }

    # ──────────────────────────────────────────────────────────────────────────
    # MARKDOWN REPORT
    # ──────────────────────────────────────────────────────────────────────────

    def _render_markdown(
        self,
        report:           dict,
        category_reports: List[CategoryReport],
    ) -> str:
        meta    = report["meta"]
        summary = report["summary"]
        lines   = []

        lines.append(f"# Compliance Gap Report")
        lines.append(f"")
        lines.append(f"**Vendor:** {meta['vendor']}  ")
        lines.append(f"**RFP:** {meta['rfp_source']}  ")
        lines.append(f"**Generated:** {meta['generated_at']}  ")
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")
        lines.append(f"## Executive Summary")
        lines.append(f"")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total capabilities evaluated | {summary['total_capabilities']} |")
        lines.append(f"| Mandatory capabilities | {summary['mandatory_capabilities']} |")
        lines.append(f"| Overall compliance | {summary['overall_compliance_pct']}% |")
        lines.append(f"| Mandatory compliance | {summary['mandatory_compliance_pct']}% |")
        lines.append(f"| Compliant / Likely | {summary['compliant']} |")
        lines.append(f"| Non-compliant | {summary['non_compliant']} |")
        lines.append(f"| Partial / Undetermined | {summary['partial'] + summary['undetermined']} |")
        lines.append(f"")

        # Category summary table
        lines.append(f"## Category Summary")
        lines.append(f"")
        lines.append(f"| Category | Capabilities | Mandatory | Overall % | Mandatory % |")
        lines.append(f"|----------|-------------|-----------|-----------|-------------|")
        for cr in category_reports:
            lines.append(
                f"| {cr.category} | {cr.total} | {cr.mandatory_count} "
                f"| {cr.compliance_rate*100:.0f}% | {cr.mandatory_compliance_rate*100:.0f}% |"
            )
        lines.append(f"")

        # Critical gaps section (mandatory non-compliant)
        all_mandatory_gaps = [
            (cr.category, r)
            for cr in category_reports
            for r in cr.non_compliant_mandatory
        ]
        if all_mandatory_gaps:
            lines.append(f"## ⚠ Critical Gaps (Mandatory Requirements Not Met)")
            lines.append(f"")
            for cat, r in all_mandatory_gaps:
                lines.append(f"### {cat} — {r.capability_id}")
                lines.append(f"")
                lines.append(f"**Requirement:** {r.capability_statement}")
                lines.append(f"")
                if r.pages:
                    lines.append(f"**RFP pages:** {', '.join(str(p) for p in r.pages)}")
                    lines.append(f"")
                if r.gap:
                    lines.append(f"**Gap:** {r.gap}")
                    lines.append(f"")
                lines.append(f"---")
                lines.append(f"")

        # Full results per category
        lines.append(f"## Detailed Results")
        lines.append(f"")
        for cr in category_reports:
            lines.append(f"### {cr.category}")
            lines.append(f"")
            lines.append(f"Compliance: {cr.compliance_rate*100:.0f}% overall, "
                          f"{cr.mandatory_compliance_rate*100:.0f}% mandatory")
            lines.append(f"")

            status_icons = {
                "COMPLIANT":        "✅",
                "LIKELY_COMPLIANT": "🟡",
                "NON_COMPLIANT":    "❌",
                "PARTIAL":          "⚠️",
                "UNDETERMINED":     "❓",
            }

            for r in cr.results:
                icon  = status_icons.get(r.status, "❓")
                mand  = " **(M)**" if r.mandatory else ""
                pages = f" — pp. {', '.join(str(p) for p in r.pages)}" if r.pages else ""
                lines.append(f"**{r.capability_id}**{mand} {icon} `{r.status}`{pages}")
                lines.append(f"")
                lines.append(f"*{r.capability_statement}*")
                lines.append(f"")
                if r.evidence:
                    lines.append(f"Evidence: {r.evidence}")
                    lines.append(f"")
                if r.gap:
                    lines.append(f"Gap: {r.gap}")
                    lines.append(f"")

            lines.append(f"")

        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _print_category_summary(self, cr: CategoryReport) -> None:
        print(f"\n  {cr.category} summary:")
        print(f"    Capabilities  : {cr.total}")
        print(f"    Compliant     : {cr.compliant_count} ({cr.compliance_rate*100:.0f}%)")
        print(f"    Mandatory gaps: {len(cr.non_compliant_mandatory)}")
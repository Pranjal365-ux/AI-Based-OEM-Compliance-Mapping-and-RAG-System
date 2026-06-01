"""
OEM Datasheet Ingestion Pipeline - Python API
Clean public interface for use by other modules (e.g. RFP scanner, compliance report generator).

Example:
    from oem_pipeline.api import OEMKnowledgeBase

    kb = OEMKnowledgeBase()
    kb.ingest("datasheets/")

    results = kb.search_for_requirement("NGFW with 20Gbps threat prevention throughput")
    for r in results:
        print(r.vendor, r.model_name, r.score)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Union

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import PipelineConfig
from ingestion.pipeline import OEMIngestionPipeline
from models.schemas import (
    FileIngestionResult,
    PipelineRunResult,
)


class SearchResult:
    """A single search result from the knowledge base."""

    def __init__(self, raw: dict):
        self._raw = raw

    @property
    def vendor(self) -> str:
        return self._raw.get("vendor", "")

    @property
    def model_name(self) -> str:
        return self._raw.get("model_name", "")

    @property
    def product_family(self) -> str:
        return self._raw.get("product_family", "")

    @property
    def product_category(self) -> str:
        return self._raw.get("product_category", "")

    @property
    def chunk_type(self) -> str:
        return self._raw.get("chunk_type", "")

    @property
    def score(self) -> float:
        return self._raw.get("score", 0.0)

    @property
    def text(self) -> str:
        return self._raw.get("text", "")

    @property
    def source_file(self) -> str:
        return self._raw.get("source_file", "")

    @property
    def metadata(self) -> dict:
        return self._raw.get("metadata", {})

    def __repr__(self) -> str:
        return (
            f"SearchResult(vendor={self.vendor!r}, model={self.model_name!r}, "
            f"type={self.chunk_type!r}, score={self.score:.3f})"
        )


class OEMKnowledgeBase:
    """
    High-level API for the OEM datasheet knowledge base.
    Suitable for integration with the RFP scanner and compliance reporter.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        groq_api_key: Optional[str] = None,
    ):
        from config.settings import DEFAULT_CONFIG
        self._cfg = config or DEFAULT_CONFIG
        if groq_api_key:
            self._cfg.groq_api_key = groq_api_key
        elif not self._cfg.groq_api_key:
            self._cfg.groq_api_key = os.getenv("GROQ_API_KEY", "")

        self._pipeline = OEMIngestionPipeline(self._cfg)
        self._ready = False

    def _ensure_ready(self) -> None:
        if not self._ready:
            self._pipeline.initialize()
            self._ready = True

    # ─── Ingestion ────────────────────────────────────────────────────────────────

    def ingest_file(
        self,
        path: Union[str, Path],
        force: bool = False,
    ) -> FileIngestionResult:
        """Ingest a single PDF datasheet into the knowledge base."""
        self._ensure_ready()
        return self._pipeline.ingest_file(path, force_reingest=force)

    def ingest(
        self,
        path: Union[str, Path],
        recursive: bool = True,
        force: bool = False,
    ) -> PipelineRunResult:
        """
        Ingest a directory of PDF datasheets (or a single file).
        Returns aggregate statistics.
        """
        self._ensure_ready()
        path = Path(path)
        if path.is_file():
            result = self._pipeline.ingest_file(path, force_reingest=force)
            run = PipelineRunResult(run_id="single", total_files=1)
            run.file_results.append(result)
            return run
        return self._pipeline.ingest_directory(path, recursive=recursive, force_reingest=force)

    # ─── Search ───────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        n: int = 10,
        vendor: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search the knowledge base with a natural language query.

        Args:
            query: Natural language query (e.g. "firewall with SD-WAN and 5Gbps throughput")
            n: Number of results to return
            vendor: Optional vendor filter
            model: Optional model name filter

        Returns:
            List of SearchResult objects, sorted by relevance score (descending)
        """
        self._ensure_ready()
        raw = self._pipeline.search(query, n_results=n, vendor=vendor, model_name=model)
        return [SearchResult(r) for r in raw]

    def search_for_requirement(
        self,
        requirement: str,
        n: int = 15,
    ) -> List[SearchResult]:
        """
        Specialised search optimised for RFP requirement matching.
        Returns the most relevant product spec chunks that match the requirement.
        """
        self._ensure_ready()
        raw = self._pipeline.vector_store.search_for_requirement(requirement, n_results=n)
        return [SearchResult(r) for r in raw]

    def get_model_specs(
        self,
        model_name: str,
        vendor: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Retrieve all stored specification chunks for a specific model.
        """
        self._ensure_ready()
        raw = self._pipeline.vector_store.search(
            query=f"specifications for {model_name}",
            n_results=50,
            vendor=vendor,
            model_name=model_name,
        )
        return [SearchResult(r) for r in raw]

    def get_all_vendors(self) -> List[str]:
        """Return list of all vendors in the knowledge base."""
        stats = self.stats()
        return [v for v, _ in stats.get("vendors", [])]

    # ─── Stats & Admin ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return knowledge base statistics."""
        self._ensure_ready()
        return self._pipeline.get_stats()

    def list_documents(self) -> list:
        """List all ingested documents."""
        self._ensure_ready()
        return self._pipeline.vector_store.list_documents()

    def delete_document(self, doc_id: str) -> int:
        """Remove all chunks for a document. Returns number of chunks deleted."""
        self._ensure_ready()
        return self._pipeline.vector_store.delete_document(doc_id)

    def close(self) -> None:
        """Release resources."""
        self._pipeline.vector_store.close()
        self._ready = False

    def __enter__(self):
        self._ensure_ready()
        return self

    def __exit__(self, *_):
        self.close()

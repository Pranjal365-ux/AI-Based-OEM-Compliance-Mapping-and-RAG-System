"""
OEM Datasheet Ingestion Pipeline - Vector Store Manager
Manages ChromaDB collections for OEM spec embeddings.
Supports:
  - Upsert / add chunks
  - Semantic search
  - Filter by vendor / model / chunk_type
  - Document deletion
  - Stats and inventory
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger
from sentence_transformers import SentenceTransformer

from config.settings import EmbeddingConfig, VectorStoreConfig
from models.schemas import ChunkType, DocumentChunk


class VectorStoreManager:
    """
    Wraps ChromaDB with a high-level API for the OEM pipeline.
    Uses SentenceTransformers for local embeddings (no external API needed).
    """

    def __init__(
        self,
        vs_cfg: VectorStoreConfig,
        emb_cfg: EmbeddingConfig,
    ):
        self.vs_cfg = vs_cfg
        self.emb_cfg = emb_cfg
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None
        self._embedder: Optional[SentenceTransformer] = None

    # ─── Lifecycle ───────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Connect to (or create) the persistent ChromaDB collection."""
        persist_dir = self.vs_cfg.persist_directory
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self._collection = self._client.get_or_create_collection(
            name=self.vs_cfg.collection_name,
            metadata={"hnsw:space": self.vs_cfg.distance_metric},
        )
        logger.info(
            f"Vector store initialized: '{self.vs_cfg.collection_name}' "
            f"at {persist_dir} "
            f"({self._collection.count()} existing chunks)"
        )

    def load_embedder(self) -> None:
        """Load the sentence transformer model into memory."""
        logger.info(f"Loading embedding model: {self.emb_cfg.model_name}")
        self._embedder = SentenceTransformer(
            self.emb_cfg.model_name,
            device=self.emb_cfg.device,
        )
        logger.info("Embedding model loaded")

    def close(self) -> None:
        """Gracefully release resources."""
        self._embedder = None
        self._collection = None
        self._client = None

    # ─── Embedding ────────────────────────────────────────────────────────────────

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of strings, returning float vectors."""
        if self._embedder is None:
            self.load_embedder()
        embeddings = self._embedder.encode(
            texts,
            batch_size=self.emb_cfg.batch_size,
            normalize_embeddings=self.emb_cfg.normalize_embeddings,
            show_progress_bar=len(texts) > 50,
        )
        return embeddings.tolist()

    # ─── CRUD ─────────────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Upsert chunks into ChromaDB.
        Returns number of chunks successfully added.
        """
        if not chunks:
            return 0

        if self._collection is None:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")

        # Check for existing chunk IDs (skip already-stored)
        existing_ids = self._get_existing_ids([c.chunk_id for c in chunks])
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

        if not new_chunks:
            logger.info("All chunks already exist in vector store – skipping")
            return 0

        # Batch processing
        batch_size = 100
        added = 0
        for start in range(0, len(new_chunks), batch_size):
            batch = new_chunks[start:start + batch_size]
            texts = [c.text for c in batch]
            ids = [c.chunk_id for c in batch]
            metadatas = [c.to_chroma_metadata() for c in batch]

            try:
                embeddings = self.embed_texts(texts)
                self._collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                )
                added += len(batch)
            except Exception as e:
                logger.error(f"Failed to add batch {start//batch_size}: {e}")
                continue

        logger.info(f"Added {added} new chunks to vector store")
        return added

    def _get_existing_ids(self, ids: List[str]) -> set:
        """Return set of IDs that already exist in the collection."""
        try:
            result = self._collection.get(ids=ids, include=[])
            return set(result.get("ids", []))
        except Exception:
            return set()

    def delete_document(self, doc_id: str) -> int:
        """Remove all chunks for a given doc_id. Returns count deleted."""
        try:
            results = self._collection.get(
                where={"doc_id": doc_id},
                include=[]
            )
            ids_to_delete = results.get("ids", [])
            if ids_to_delete:
                self._collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} chunks for doc {doc_id}")
            return len(ids_to_delete)
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return 0

    def document_exists(self, doc_id: str) -> bool:
        """Check if any chunks for this doc_id are already in the store."""
        try:
            results = self._collection.get(
                where={"doc_id": doc_id},
                limit=1,
                include=[]
            )
            return len(results.get("ids", [])) > 0
        except Exception:
            return False

    # ─── Search ───────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        n_results: int = 10,
        vendor: Optional[str] = None,
        model_name: Optional[str] = None,
        chunk_type: Optional[ChunkType] = None,
        product_category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over all stored chunks.
        Optional metadata filters narrow results.

        Returns list of dicts with keys:
            id, text, score (0-1), vendor, model_name, chunk_type, metadata
        """
        where: Dict[str, Any] = {}
        if vendor:
            where["vendor"] = {"$eq": vendor}
        if model_name:
            where["model_name"] = {"$eq": model_name}
        if chunk_type:
            where["chunk_type"] = {"$eq": chunk_type.value}
        if product_category:
            where["product_category"] = {"$eq": product_category}

        query_embedding = self.embed_texts([query])[0]

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, max(1, self._collection.count())),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        try:
            results = self._collection.query(**kwargs)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

        output = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            score = 1.0 - dist   # cosine distance → similarity
            output.append({
                "id": cid,
                "text": doc,
                "score": round(score, 4),
                "vendor": meta.get("vendor", ""),
                "model_name": meta.get("model_name", ""),
                "chunk_type": meta.get("chunk_type", ""),
                "product_family": meta.get("product_family", ""),
                "product_category": meta.get("product_category", ""),
                "source_file": meta.get("source_file", ""),
                "metadata": meta,
            })

        return sorted(output, key=lambda x: -x["score"])

    def search_for_requirement(
        self,
        requirement_text: str,
        n_results: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        Specialised search for RFP requirement matching.
        Returns ranked list of product specs that may satisfy the requirement.
        """
        # Enrich query with technical context
        enriched_query = f"Technical specification requirement: {requirement_text}"
        return self.search(enriched_query, n_results=n_results)

    # ─── Inventory / Stats ────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return counts and inventory information."""
        total = self._collection.count()
        if total == 0:
            return {"total_chunks": 0, "vendors": [], "models": []}

        # Sample to get vendor/model breakdown
        # ChromaDB doesn't support GROUP BY, so we peek at metadata
        try:
            sample = self._collection.get(
                limit=min(total, 5000),
                include=["metadatas"]
            )
            metas = sample.get("metadatas", [])

            vendors: Dict[str, int] = {}
            models: Dict[str, int] = {}
            chunk_types: Dict[str, int] = {}

            for m in metas:
                v = m.get("vendor", "Unknown")
                mn = m.get("model_name", "Unknown")
                ct = m.get("chunk_type", "unknown")
                vendors[v] = vendors.get(v, 0) + 1
                models[mn] = models.get(mn, 0) + 1
                chunk_types[ct] = chunk_types.get(ct, 0) + 1

            return {
                "total_chunks": total,
                "vendor_count": len(vendors),
                "model_count": len(models),
                "vendors": sorted(vendors.items(), key=lambda x: -x[1]),
                "chunk_type_distribution": chunk_types,
            }
        except Exception as e:
            logger.error(f"Stats query failed: {e}")
            return {"total_chunks": total}

    def list_documents(self) -> List[Dict[str, str]]:
        """Return one entry per unique doc_id."""
        try:
            sample = self._collection.get(
                limit=10000,
                include=["metadatas"]
            )
            seen: Dict[str, Dict] = {}
            for m in sample.get("metadatas", []):
                did = m.get("doc_id", "")
                if did and did not in seen:
                    seen[did] = {
                        "doc_id": did,
                        "vendor": m.get("vendor", ""),
                        "source_file": m.get("source_file", ""),
                    }
            return list(seen.values())
        except Exception as e:
            logger.error(f"list_documents failed: {e}")
            return []

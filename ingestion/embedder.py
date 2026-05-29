
# =============================================================================
# embedder.py — Embedding + ChromaDB storage (BGE-base-en-v1.5)
# =============================================================================

import logging
import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL, DB_PATH, COLLECTION

logger = logging.getLogger("ingestion")

_model = None
_client = None
_collection = None

BATCH_SIZE = 32


# =============================================================================
# MODEL
# =============================================================================

def _get_model() -> SentenceTransformer:

    global _model

    if _model is None:

        logger.info(
            f"  [embedder] Loading {EMBED_MODEL} ..."
        )

        _model = SentenceTransformer(
            EMBED_MODEL
        )

        logger.info(
            f"  [embedder] Model loaded."
        )

    return _model


# =============================================================================
# COLLECTION
# =============================================================================

def _get_collection() -> chromadb.Collection:

    global _client
    global _collection

    if _collection is None:

        _client = chromadb.PersistentClient(
            path=DB_PATH
        )

        _collection = _client.get_or_create_collection(

            name=COLLECTION,

            metadata={

                "hnsw:space":"cosine"
            }
        )

        logger.info(

            f"  [embedder] Collection "
            f"'{COLLECTION}' "
            f"({_collection.count()} existing vectors)"
        )

    return _collection


# =============================================================================
# STORE
# =============================================================================

def embed_and_store(chunks:list)->int:

    if not chunks:

        logger.warning(
            "  [embedder] No chunks to embed."
        )

        return 0


    model=_get_model()

    collection=_get_collection()

    total=0


    for start in range(

        0,
        len(chunks),
        BATCH_SIZE
    ):

        batch=chunks[
            start:
            start+BATCH_SIZE
        ]

        texts=[

            c["text"]

            for c in batch
        ]

        metadatas=[

            c["metadata"]

            for c in batch
        ]

        ids=[

            c["metadata"]["chunk_id"]

            for c in batch
        ]


        embeddings=model.encode(

            texts,

            batch_size=BATCH_SIZE,

            show_progress_bar=False,

            normalize_embeddings=True
        ).tolist()


        collection.upsert(

            ids=ids,

            documents=texts,

            embeddings=embeddings,

            metadatas=metadatas
        )

        total+=len(batch)


    logger.info(

        f"  [embedder] "
        f"{total} vectors stored. "
        f"Collection total: "
        f"{collection.count()}"
    )

    return total


# =============================================================================
# STATS
# =============================================================================

def get_collection_stats()->dict:

    collection=_get_collection()

    total=collection.count()

    if total==0:

        return{

            "total":0,
            "by_category":{},
            "by_vendor":{},
            "by_chunk_type":{},
            "by_model_combo":{}
        }


    results=collection.get(
        include=["metadatas"]
    )


    cat_counts={}
    vendor_counts={}
    type_counts={}
    combo_counts={}


    for meta in results["metadatas"]:
        category = meta.get("category", "UNKNOWN")
        vendor = meta.get("vendor", "UNKNOWN")
        chunk_type = meta.get("chunk_type", "UNKNOWN")
        product_family = meta.get("product_family", "")

        if product_family and product_family != "UNKNOWN":
            combo = f"{vendor} {product_family}"
        else:
            combo = f"{vendor} {category}"

        cat_counts[category] = cat_counts.get(category, 0) + 1
        vendor_counts[vendor] = vendor_counts.get(vendor, 0) + 1
        type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        combo_counts[combo] = combo_counts.get(combo, 0) + 1

    return {
        "total": total,
        "by_category": cat_counts,
        "by_vendor": vendor_counts,
        "by_chunk_type": type_counts,
        "by_model_combo": combo_counts
    }

#!/usr/bin/env python3
# =============================================================================
# ingest.py — Main ingestion orchestrator
# =============================================================================

import os
import sys
import json
import logging
import argparse
import time
import shutil
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config         import DOCS_PATH, LOG_PATH, DB_PATH, COLLECTION
from extractor      import extract_pages
from cleaner        import clean_document
from classifier     import detect_category
from meta_extractor import extract_doc_meta
from chunker        import chunk_pages
from embedder       import embed_and_store, get_collection_stats


# ── Logging ─────────────────────────────────────────

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH,mode="a"),
    ]
)

logger=logging.getLogger("ingestion")


# ── Single file ─────────────────────────────────────

def ingest_file(pdf_path:Path)->dict:

    start=time.time()

    result={

        "file":pdf_path.name,
        "status":"failed",
        "category":None,
        "vendor":None,
        "models":[],
        "pages":0,
        "chunks":0,
        "vectors":0,
        "ocr_pages":0,
        "error":None
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {pdf_path.name}")

    try:

        # Extract
        raw_pages = extract_pages(str(pdf_path))
        if not raw_pages:
            raise ValueError("No text extracted.")

        result["pages"] = len(raw_pages)
        result["ocr_pages"] = sum(1 for p in raw_pages if p.get("ocr_used"))

        # Clean
        pages = clean_document(raw_pages)

        if not pages:

            raise ValueError(
                "All pages empty."
            )


        # Category

        full_text="\n".join(

            p["text"]

            for p in pages
        )

        category,conf=detect_category(

            pdf_path.name,
            full_text
        )

        result["category"]=category


        # FIXED HERE

        # Step 4 — Extract vendor + models from content (NO filename parsing)
        doc_identity = extract_doc_meta(raw_pages, pdf_path.name)
        result["vendor"] = doc_identity["vendor"]
        result["models"] = doc_identity["models"]

        doc_meta = {
            "vendor":         doc_identity["vendor"],
            "product_family": doc_identity["product_family"],
            "models":         doc_identity["models"],
            "category":       category,
            "doc_name":       pdf_path.name,
        }


        chunks=chunk_pages(

            pages,
            doc_meta
        )

        if not chunks:

            raise ValueError(
                "No chunks produced."
            )

        result["chunks"]=len(
            chunks
        )


        result["vectors"]=embed_and_store(
            chunks
        )

        result["status"]="success"


    except Exception as e:

        result["error"]=str(e)

        logger.error(
            f"FAILED: {e}",
            exc_info=True
        )


    elapsed=round(
        time.time()-start,
        1
    )

    logger.info(

        f"{result['status'].upper()} | "
        f"category={result['category']} | "
        f"vendor='{result['vendor']}' | "
        f"models={result['models']} | "
        f"pages={result['pages']} | "
        f"chunks={result['chunks']} | "
        f"vectors={result['vectors']} | "
        f"{elapsed}s"

    )

    return result


# ── Batch ───────────────────────────────────────────

def ingest_all(
    docs_path=DOCS_PATH
):

    pdf_files=sorted(
        Path(
            docs_path
        ).glob(
            "*.pdf"
        )
    )

    results=[

        ingest_file(p)

        for p in pdf_files
    ]

    print_stats()

    return results


# ── Stats ───────────────────────────────────────────

def print_stats():

    logger.info(
        f"\n{'='*60}"
    )

    logger.info(
        "KNOWLEDGE BASE STATS"
    )

    stats=get_collection_stats()

    logger.info(
        f"  Total vectors : {stats['total']}"
    )

    logger.info(
        "  By category:"
    )

    for k,v in sorted(
        stats["by_category"].items()
    ):

        logger.info(
            f"    {k:<15}: {v}"
        )


    logger.info(
        "  By vendor:"
    )

    for k,v in sorted(
        stats["by_vendor"].items()
    ):

        logger.info(
            f"    {k:<30}: {v}"
        )


    logger.info(
        "  By family:"
    )

    for k,v in sorted(
        stats["by_model_combo"].items()
    ):

        logger.info(
            f"    {k:<45}: {v}"
        )


    logger.info(
        "  By chunk type:"
    )

    for k,v in sorted(
        stats["by_chunk_type"].items()
    ):

        logger.info(
            f"    {k:<15}: {v}"
        )


# ── Reset ───────────────────────────────────────────

def reset_db():

    if Path(
        DB_PATH
    ).exists():

        shutil.rmtree(
            DB_PATH
        )

        logger.info(
            f"DB wiped: {DB_PATH}"
        )


# ── CLI ─────────────────────────────────────────────

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    parser.add_argument(
        "--reset",
        action="store_true"
    )

    args=parser.parse_args()

    if args.reset:

        reset_db()

    ingest_all()


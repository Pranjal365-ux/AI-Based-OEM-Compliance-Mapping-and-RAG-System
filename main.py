#!/usr/bin/env python3
"""
OEM Datasheet Ingestion Pipeline - CLI Entry Point

Usage examples:
  # Ingest a single PDF
  python main.py ingest --file datasheets/fortigate_200f.pdf

  # Ingest all PDFs in a directory
  python main.py ingest --dir datasheets/

  # Force re-ingest (ignore skip-existing)
  python main.py ingest --dir datasheets/ --force

  # Search the knowledge base
  python main.py search --query "Next-gen firewall with 10Gbps throughput"

  # Show knowledge base statistics
  python main.py stats

  # Delete a document from the knowledge base
  python main.py delete --doc-id abc123def456
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass


# ─── Path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OEM Datasheet Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── ingest ─────────────────────────────────────────────────────────────────
    ingest_p = sub.add_parser("ingest", help="Ingest PDF datasheets")
    ingest_group = ingest_p.add_mutually_exclusive_group(required=True)
    ingest_group.add_argument("--file", type=Path, help="Single PDF to ingest")
    ingest_group.add_argument("--dir", type=Path, help="Directory of PDFs to ingest")
    ingest_p.add_argument(
        "--force", action="store_true", default=False,
        help="Re-ingest even if document already exists in vector store"
    )
    ingest_p.add_argument(
        "--no-recursive", action="store_true", default=False,
        help="Do not search subdirectories (only with --dir)"
    )
    ingest_p.add_argument(
        "--no-llm", action="store_true", default=False,
        help="Disable LLM-based model identification (faster, less accurate)"
    )
    ingest_p.add_argument(
        "--output-json", type=Path, default=None,
        help="Write ingestion results to a JSON file"
    )

    # ── search ─────────────────────────────────────────────────────────────────
    search_p = sub.add_parser("search", help="Search the knowledge base")
    search_p.add_argument("--query", required=True, help="Natural language query")
    search_p.add_argument("--n", type=int, default=10, help="Number of results")
    search_p.add_argument("--vendor", type=str, default=None, help="Filter by vendor")
    search_p.add_argument("--model", type=str, default=None, help="Filter by model name")
    search_p.add_argument("--json", action="store_true", help="Output raw JSON")

    # ── stats ──────────────────────────────────────────────────────────────────
    sub.add_parser("stats", help="Show knowledge base statistics")

    # ── delete ─────────────────────────────────────────────────────────────────
    delete_p = sub.add_parser("delete", help="Remove a document from the knowledge base")
    delete_p.add_argument("--doc-id", required=True, help="Document ID to delete")

    # ── list-docs ──────────────────────────────────────────────────────────────
    sub.add_parser("list-docs", help="List all ingested documents")

    return parser


def cmd_ingest(args, pipeline) -> None:
    if args.no_llm:
        pipeline.cfg.use_llm_for_model_id = False

    if args.file:
        result = pipeline.ingest_file(args.file, force_reingest=args.force)
        print(f"\n{'─'*60}")
        print(f"Status   : {result.status.value}")
        print(f"Vendor   : {result.vendor}")
        print(f"Models   : {result.models_found}")
        print(f"Chunks   : {result.chunks_created}")
        print(f"Time     : {result.processing_time_seconds}s")
        if result.warnings:
            print(f"Warnings : {'; '.join(result.warnings)}")
        if result.error_message:
            print(f"Error    : {result.error_message}")
        if args.output_json:
            Path(args.output_json).write_text(result.model_dump_json(indent=2))
    else:
        run = pipeline.ingest_directory(
            args.dir,
            recursive=not args.no_recursive,
            force_reingest=args.force,
        )
        print(f"\n{'─'*60}")
        print(f"Run ID   : {run.run_id}")
        print(f"Total    : {run.total_files}")
        print(f"Success  : {run.successful}")
        print(f"Failed   : {run.failed}")
        print(f"Skipped  : {run.skipped}")
        print(f"Models   : {run.total_models_extracted}")
        print(f"Chunks   : {run.total_chunks_created}")
        print(f"Duration : {run.duration_seconds:.1f}s")

        if run.failed > 0:
            print("\nFailed files:")
            for r in run.file_results:
                if r.status.value == "failed":
                    print(f"  - {r.file_path}: {r.error_message}")

        if args.output_json:
            Path(args.output_json).write_text(run.model_dump_json(indent=2))


def cmd_search(args, pipeline) -> None:
    results = pipeline.search(
        args.query,
        n_results=args.n,
        vendor=args.vendor,
        model_name=args.model,
    )

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print(f"\nQuery: '{args.query}'")
    print(f"Found {len(results)} results\n")
    print("─" * 70)

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Score: {r['score']:.3f}")
        print(f"    Vendor : {r['vendor']}")
        print(f"    Model  : {r['model_name']}")
        if r.get("product_family"):
            print(f"    Family : {r['product_family']}")
        print(f"    Type   : {r['chunk_type']}")
        print(f"    File   : {Path(r['source_file']).name}")
        preview = r["text"][:300].replace("\n", " ")
        print(f"    Text   : {preview}…")


def cmd_stats(pipeline) -> None:
    stats = pipeline.get_stats()
    print("\n── Knowledge Base Statistics ──")
    print(f"Total chunks  : {stats.get('total_chunks', 0)}")
    print(f"Unique vendors: {stats.get('vendor_count', 0)}")
    print(f"Unique models : {stats.get('model_count', 0)}")

    if stats.get("vendors"):
        print("\nTop vendors:")
        for name, count in stats["vendors"][:10]:
            print(f"  {name:<30} {count} chunks")

    if stats.get("chunk_type_distribution"):
        print("\nChunk type distribution:")
        for ct, count in sorted(stats["chunk_type_distribution"].items()):
            print(f"  {ct:<25} {count}")


def cmd_delete(args, pipeline) -> None:
    n = pipeline.vector_store.delete_document(args.doc_id)
    if n > 0:
        print(f"Deleted {n} chunks for document {args.doc_id}")
    else:
        print(f"No chunks found for document {args.doc_id}")


def cmd_list_docs(pipeline) -> None:
    docs = pipeline.vector_store.list_documents()
    if not docs:
        print("No documents ingested yet.")
        return
    print(f"\n{len(docs)} document(s) in knowledge base:\n")
    print(f"{'Doc ID':<18} {'Vendor':<20} {'Source File'}")
    print("─" * 70)
    for d in docs:
        print(f"{d['doc_id']:<18} {d['vendor']:<20} {Path(d['source_file']).name}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Import here so path is set
    from config.settings import DEFAULT_CONFIG
    from ingestion.pipeline import OEMIngestionPipeline

    config = DEFAULT_CONFIG

    # Apply API key from env
    api_key = os.getenv("GROQ_API_KEY", "")
    if api_key:
        config.groq_api_key = api_key

    pipeline = OEMIngestionPipeline(config)

    try:
        if args.command == "ingest":
            pipeline.initialize()
            cmd_ingest(args, pipeline)
        elif args.command == "search":
            pipeline.initialize()
            cmd_search(args, pipeline)
        elif args.command == "stats":
            pipeline.initialize()
            cmd_stats(pipeline)
        elif args.command == "delete":
            pipeline.initialize()
            cmd_delete(args, pipeline)
        elif args.command == "list-docs":
            pipeline.initialize()
            cmd_list_docs(pipeline)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 1
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1
    finally:
        pipeline.vector_store.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

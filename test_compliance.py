"""
End-to-end compliance pipeline test / CLI runner.

Usage
-----
  # From a requirements JSON file (output of rfp_extractor.run()):
  python test_compliance.py --json data/requirements/datacenter_pp1-5.json

  # From a live RFP PDF (extracts then evaluates in one go):
  python test_compliance.py --pdf path/to/rfp.pdf --start 1 --end 10

  # Control how many top products to report (default 3):
  python test_compliance.py --json ... --top 5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import DEFAULT_CONFIG
from knowledge_base.vector_store import VectorStoreManager
from compliance.engine import ComplianceEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="OEM Compliance Engine")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--json", type=str, help="Path to requirements JSON from rfp_extractor")
    src.add_argument("--pdf",  type=str, help="Path to RFP PDF (extract + evaluate in one go)")
    parser.add_argument("--start", type=int, default=1,  help="Start page (with --pdf)")
    parser.add_argument("--end",   type=int, default=10, help="End page (with --pdf)")
    parser.add_argument("--top",   type=int, default=3,  help="Number of top products to report")
    args = parser.parse_args()

    # ── Initialise knowledge base ─────────────────────────────────────────────
    print("\nInitialising OEM knowledge base…")
    vs = VectorStoreManager(DEFAULT_CONFIG.vector_store, DEFAULT_CONFIG.embedding)
    vs.initialize()
    vs.load_embedder()

    stats = vs.get_stats()
    print(
        f"  KB: {stats.get('total_chunks', 0)} chunks | "
        f"{stats.get('vendor_count', 0)} vendors | "
        f"{stats.get('model_count', 0)} models"
    )

    engine = ComplianceEngine(vector_store=vs, top_n=args.top)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    if args.json:
        print(f"\nLoading requirements from: {args.json}")
        report = engine.run_from_json(args.json, top_n=args.top)

    else:
        print(f"\nExtracting requirements from PDF: {args.pdf}")
        print(f"  Pages: {args.start}–{args.end}")

        from rfp.rfp_extractor import RFPRequirementExtractor
        extractor = RFPRequirementExtractor()
        result = extractor.run(
            pdf_path   = args.pdf,
            start_page = args.start,
            end_page   = args.end,
            embed      = True,
        )
        print(f"  Extracted {result['requirement_count']} requirements")

        from models.schemas import Requirement
        reqs = [Requirement(**r) for r in result["requirements"]]
        report = engine.run(
            requirements = reqs,
            rfp_source   = args.pdf,
            page_range   = result["page_range"],
        )

    # ── Print summary to console ──────────────────────────────────────────────
    _print_summary(report)

    # ── Show output file paths ────────────────────────────────────────────────
    from compliance.reporter import REPORTS_DIR
    report_files = sorted(REPORTS_DIR.glob(f"compliance_{report.report_id}.*"))
    print(f"{'─'*65}")
    print("Reports saved:")
    for f in report_files:
        print(f"  → {f}")


def _print_summary(report) -> None:
    """Print a human-readable compliance summary to stdout."""
    from compliance.schemas import ComplianceStatus

    print(f"\n{'='*65}")
    print(f"COMPLIANCE REPORT  [{report.report_id}]")
    print(f"{'='*65}")
    print(
        f"Requirements: {report.total_requirements} "
        f"({report.mandatory_count} mandatory / {report.optional_count} optional)"
    )
    print()

    if not report.top_products:
        print("❌ No matching products found in the knowledge base.")
        print("   Ensure OEM datasheets have been ingested before running the engine.")
        return

    medals = ["🥇", "🥈", "🥉"]
    for i, p in enumerate(report.top_products, 1):
        print(
            f"  {medals[i-1] if i <= 3 else f'#{i}'} "
            f"{p.vendor} – {p.model_name}"
        )
        if p.product_family:
            print(f"     Family   : {p.product_family}")
        print(f"     Overall  : {_bar(p.overall_score)}  {p.overall_score:.1f}%")
        print(f"     Mandatory: {_bar(p.mandatory_score)}  {p.mandatory_score:.1f}%")
        print(f"     ✅ {p.full_matches}  ⚠️ {p.partial_matches}  ❌ {p.no_matches}")
        if p.key_gaps:
            shown = p.key_gaps[:3]
            more  = len(p.key_gaps) - 3
            print(
                f"     Gaps     : {'; '.join(shown)}"
                + (f" (+{more} more)" if more > 0 else "")
            )
        print()


def _bar(score: float, width: int = 20) -> str:
    filled = round(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


if __name__ == "__main__":
    main()
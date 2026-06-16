"""
Compliance Engine – Report Generator  (Phase 6: Reporting)
============================================================
Takes the ranked ProductScore list and produces:
  1. A structured JSON report  → data/compliance_reports/<id>.json
  2. A human-readable Markdown report  → data/compliance_reports/<id>.md

Markdown report structure
--------------------------
  • Executive summary with ranked comparison table
  • Per-product sections:
      – Compliance matrix (all requirements)
      – Gap analysis (mandatory failures only)
      – Evidence references (source doc + page + snippet)
  • Footer
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from config.settings import DEFAULT_CONFIG
from compliance.schemas import (
    ComplianceReport,
    ComplianceStatus,
    ProductScore,
)
from models.schemas import Requirement

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(DEFAULT_CONFIG.rfp.output_dir).parent / "compliance_reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Max evidence entries shown per product in the report
MAX_EVIDENCE_ENTRIES = 10


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(
    requirements:  List[Requirement],
    top_products:  List[ProductScore],
    rfp_source:    str = "",
    page_range:    Optional[dict] = None,
    kb_chunks:     int = 0,
) -> ComplianceReport:
    """Build, persist, and return a ComplianceReport."""
    report = ComplianceReport(
        report_id           = uuid.uuid4().hex[:12],
        rfp_source          = rfp_source,
        page_range          = page_range or {},
        total_requirements  = len(requirements),
        mandatory_count     = sum(1 for r in requirements if r.mandatory),
        optional_count      = sum(1 for r in requirements if not r.mandatory),
        top_products        = top_products,
        kb_chunks_searched  = kb_chunks,
        llm_model_used      = DEFAULT_CONFIG.llm.model,
    )

    _save_json(report)
    _save_markdown(report)
    return report


# ══════════════════════════════════════════════════════════════════════════════
# JSON OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def _save_json(report: ComplianceReport) -> Path:
    out = REPORTS_DIR / f"compliance_{report.report_id}.json"
    with open(out, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2, default=str))
    logger.info(f"✓ JSON report → {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def _save_markdown(report: ComplianceReport) -> Path:
    out = REPORTS_DIR / f"compliance_{report.report_id}.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(_build_markdown(report))
    logger.info(f"✓ Markdown report → {out}")
    return out


def _build_markdown(report: ComplianceReport) -> str:
    lines: List[str] = []
    ts = report.generated_at.strftime("%Y-%m-%d %H:%M UTC")

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        "# OEM Compliance Report",
        "",
        f"**Report ID:** `{report.report_id}`  ",
        f"**Generated:** {ts}  ",
        f"**RFP Source:** {Path(report.rfp_source).name if report.rfp_source else 'N/A'}  ",
    ]
    if report.page_range:
        lines.append(
            f"**Page Range:** {report.page_range.get('start')}–{report.page_range.get('end')}  "
        )
    lines += [
        f"**Requirements Analysed:** {report.total_requirements} "
        f"({report.mandatory_count} mandatory / {report.optional_count} optional)  ",
        f"**LLM Model:** {report.llm_model_used}  ",
        "",
    ]

    # ── Executive Summary ─────────────────────────────────────────────────────
    lines += ["## Executive Summary", ""]
    if not report.top_products:
        lines += ["No matching products found in the knowledge base.", ""]
    else:
        winner = report.top_products[0]
        lines += [
            f"The top-ranked product is **{winner.vendor} {winner.model_name}** "
            f"with an overall compliance score of **{winner.overall_score:.1f}%** "
            f"({winner.mandatory_score:.1f}% on mandatory requirements).",
            "",
            "| Rank | Vendor | Model | Overall | Mandatory | Full | Partial | No Match |",
            "|------|--------|-------|---------|-----------|------|---------|----------|",
        ]
        for i, p in enumerate(report.top_products, 1):
            lines.append(
                f"| {i} | {p.vendor} | {p.model_name} | "
                f"{p.overall_score:.1f}% | {p.mandatory_score:.1f}% | "
                f"{p.full_matches} | {p.partial_matches} | {p.no_matches} |"
            )
        lines.append("")

    # ── OEM Recommendation Summary ────────────────────────────────────────────
    if report.top_products:
        lines += ["## OEM Recommendation Summary", ""]
        for i, p in enumerate(report.top_products, 1):
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"#{i}")
            gap_count = len([r for r in p.requirement_results
                             if r.mandatory and r.status != ComplianceStatus.FULL])
            rec = (
                "**Recommended** — strong mandatory coverage."
                if p.mandatory_score >= 80 else
                "**Conditionally recommended** — review mandatory gaps before selection."
                if p.mandatory_score >= 50 else
                "**Not recommended** — significant mandatory gaps."
            )
            lines += [
                f"{medal} **{p.vendor} {p.model_name}** — {rec}  ",
                f"  Overall: {p.overall_score:.1f}% | Mandatory: {p.mandatory_score:.1f}% "
                f"| Mandatory gaps: {gap_count}  ",
                "",
            ]

    # ── Per-product sections ───────────────────────────────────────────────────
    for rank, product in enumerate(report.top_products, 1):
        lines += _product_section(rank, product)

    # ── Footer ────────────────────────────────────────────────────────────────
    lines += [
        "---",
        f"*Generated by AI-Powered OEM Compliance Mapping System | {ts}*",
        "",
    ]
    return "\n".join(lines)


def _product_section(rank: int, product: ProductScore) -> List[str]:
    """Build the full Markdown section for one product."""
    lines: List[str] = []
    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")

    lines += [
        "---",
        "",
        f"## {medal} Rank {rank}: {product.vendor} – {product.model_name}",
        "",
    ]
    if product.product_family:
        lines.append(f"**Product Family:** {product.product_family}  ")
    lines += [
        f"**Overall Score:** {product.overall_score:.1f}%  ",
        f"**Mandatory Score:** {product.mandatory_score:.1f}%  ",
        f"**Optional Score:** {product.optional_score:.1f}%  ",
        "",
        "| | Count |",
        "|---|---|",
        f"| ✅ Full Match | {product.full_matches} |",
        f"| ⚠️ Partial Match | {product.partial_matches} |",
        f"| ❌ No Match | {product.no_matches} |",
        f"| **Total** | **{product.total_requirements}** |",
        "",
    ]
    if product.source_files:
        lines += [
            f"**Source Documents:** {', '.join(product.source_files)}  ",
            "",
        ]

    # ── Compliance Matrix (Requirement Mapping Report) ─────────────────────────
    lines += [
        "### Compliance Matrix",
        "",
        "| ID | Category | M/O | Requirement | Status | Confidence | Gap |",
        "|----|----------|-----|-------------|--------|------------|-----|",
    ]
    for r in product.requirement_results:
        mo      = "M" if r.mandatory else "O"
        status  = _status_icon(r.status)
        conf    = f"{r.confidence:.0%}"
        req_txt = (r.requirement[:80] + "…") if len(r.requirement) > 80 else r.requirement
        gap_txt = ((r.gap[:60] + "…") if r.gap and len(r.gap) > 60 else r.gap or "—")
        lines.append(
            f"| {r.requirement_id} | {r.category} | {mo} | {req_txt} | "
            f"{status} | {conf} | {gap_txt} |"
        )
    lines.append("")

    # ── Gap Analysis (mandatory failures only) ────────────────────────────────
    mandatory_gaps = [
        r for r in product.requirement_results
        if r.mandatory and r.status != ComplianceStatus.FULL
    ]
    if mandatory_gaps:
        lines += ["### Gap Analysis (Mandatory Requirements)", ""]
        for r in mandatory_gaps:
            icon = "⚠️" if r.status == ComplianceStatus.PARTIAL else "❌"
            lines += [
                f"**{icon} {r.requirement_id} – {r.requirement}**  ",
                f"Status: {r.status.value} | Confidence: {r.confidence:.0%}  ",
                f"Justification: {r.justification}  ",
            ]
            if r.gap:
                lines.append(f"Gap: {r.gap}  ")
            lines.append("")
    else:
        lines += [
            "### Gap Analysis",
            "",
            "✅ All mandatory requirements are fully or partially met.",
            "",
        ]

    # ── Evidence References ───────────────────────────────────────────────────
    # Per spec: display supporting text with source document + page reference.
    lines += ["### Evidence References", ""]
    shown = 0
    for r in product.requirement_results:
        if not r.evidence or r.status == ComplianceStatus.NO:
            continue
        top_chunk = r.evidence[0]
        src  = Path(top_chunk.source_file).name if top_chunk.source_file else "KB"
        page = f" p.{top_chunk.page_start}" if top_chunk.page_start else ""
        snippet = top_chunk.text[:250].replace("\n", " ").strip()
        lines += [
            f"**{r.requirement_id}** ({r.requirement[:60]}{'…' if len(r.requirement) > 60 else ''})  ",
            f"Source: `{src}{page}` | Similarity: {top_chunk.score:.2f}  ",
            f"> {snippet}…",
            "",
        ]
        shown += 1
        if shown >= MAX_EVIDENCE_ENTRIES:
            remaining = len([rr for rr in product.requirement_results
                             if rr.evidence and rr.status != ComplianceStatus.NO]) - shown
            if remaining > 0:
                lines += [f"*… and {remaining} more evidence entries not shown.*", ""]
            break

    return lines


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _status_icon(status: ComplianceStatus) -> str:
    return {
        ComplianceStatus.FULL:    "✅ Full",
        ComplianceStatus.PARTIAL: "⚠️ Partial",
        ComplianceStatus.NO:      "❌ No Match",
    }.get(status, status.value)
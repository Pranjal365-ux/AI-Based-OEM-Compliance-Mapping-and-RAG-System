"""
app.py — OEM Compliance Mapping System UI
============================================
Three tabs:
  1. "Upload RFP"        — upload an RFP PDF, pick a page range, extract
                            requirements, run the compliance engine, view
                            the result inline.
  2. "Manage Database"   — see every OEM datasheet currently embedded in
                            the knowledge base (vendor, models, chunk
                            counts), upload new datasheets, remove existing
                            ones.
  3. "Reports"           — browse every compliance report generated so
                            far, view/download the Markdown or JSON.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import DEFAULT_CONFIG, RAW_DIR, RFP_DIR
from knowledge_base.vector_store import VectorStoreManager
from api import OEMKnowledgeBase
from rfp.rfp_extractor import RFPRequirementExtractor
from compliance.engine import ComplianceEngine
from compliance.reporter import REPORTS_DIR
from models.schemas import Requirement

from ui.kb_inventory import (
    get_kb_inventory,
    save_uploaded_pdf,
    save_uploaded_rfp,
    list_report_files,
    delete_report,
)

st.set_page_config(
    page_title="OEM Compliance Mapping System",
    page_icon="",
    layout="wide",
)

RFP_RAW_DIR = Path(DEFAULT_CONFIG.rfp.output_dir).parent / "raw_rfps"


# ──────────────────────────────────────────────────────────────────────────────
# CACHED / SHARED RESOURCES
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Connecting to knowledge base…")
def get_kb() -> OEMKnowledgeBase:
    kb = OEMKnowledgeBase()
    kb._ensure_ready()
    return kb


@st.cache_resource(show_spinner=False)
def get_vector_store(_kb: OEMKnowledgeBase) -> VectorStoreManager:
    return _kb._pipeline.vector_store


def get_extractor() -> RFPRequirementExtractor:
    if "extractor" not in st.session_state:
        st.session_state.extractor = RFPRequirementExtractor()
    return st.session_state.extractor


def get_engine(vs: VectorStoreManager, top_n: int) -> ComplianceEngine:
    return ComplianceEngine(vector_store=vs, top_n=top_n)


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — quick KB stats, always visible
# ──────────────────────────────────────────────────────────────────────────────

def render_sidebar(kb: OEMKnowledgeBase):
    st.sidebar.title(" OEM Compliance System")
    st.sidebar.caption("AI-powered RFP - OEM compliance mapping")
    st.sidebar.divider()

    try:
        stats = kb.stats()
    except Exception:
        stats = {}

    st.sidebar.metric("Chunks in KB", stats.get("total_chunks", 0))
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Vendors", stats.get("vendor_count", 0))
    c2.metric("Models", stats.get("model_count", 0))

    st.sidebar.divider()
    st.sidebar.caption(f"LLM (reasoning): `{DEFAULT_CONFIG.llm.model}`")
    st.sidebar.caption(f"LLM (extraction): `{DEFAULT_CONFIG.llm.extraction_model}`")
    st.sidebar.caption(f"Embedding: `{DEFAULT_CONFIG.embedding.model_name}`")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — UPLOAD RFP → PAGE RANGE → EXTRACT → COMPLIANCE
# ──────────────────────────────────────────────────────────────────────────────

def render_upload_rfp_tab(kb: OEMKnowledgeBase, vs: VectorStoreManager):
    st.header("Upload RFP & Generate Compliance Report")

    # ── Step 1: upload ───────────────────────────────────────────────────────
    st.subheader("1. Upload RFP")
    uploaded = st.file_uploader(
        "Upload an RFP / tender document (PDF)",
        type=["pdf"],
        key="rfp_uploader",
    )

    if uploaded is None:
        st.info("Upload a PDF to begin.")
        return

    # Save once per distinct upload (avoid re-writing on every rerun)
    cache_key = f"rfp_path::{uploaded.name}::{uploaded.size}"
    if cache_key not in st.session_state:
        saved_path = save_uploaded_rfp(uploaded, RFP_RAW_DIR)
        st.session_state[cache_key] = str(saved_path)
        # New file → clear any previously extracted state tied to a
        # different file so the UI doesn't show stale page counts.
        for k in ("rfp_pages", "rfp_pages_path", "extracted_requirements",
                   "extracted_meta", "compliance_report"):
            st.session_state.pop(k, None)
    rfp_path = st.session_state[cache_key]
    st.success(f"Loaded: **{uploaded.name}**")

    # ── Step 2: page count (lazy, only once per file) ────────────────────────
    extractor = get_extractor()
    if st.session_state.get("rfp_pages_path") != rfp_path:
        with st.spinner("Reading PDF page count…"):
            pages = extractor.extract_pages(rfp_path)
        st.session_state.rfp_pages = pages
        st.session_state.rfp_pages_path = rfp_path

    pages = st.session_state.rfp_pages
    total_pages = len(pages)
    st.caption(f"Document has **{total_pages}** page(s).")

    # ── Step 3: page range ────────────────────────────────────────────────────
    st.subheader("2. Select Page Range to Process")
    col1, col2 = st.columns(2)
    start_page = col1.number_input(
        "Start page", min_value=1, max_value=total_pages, value=1, step=1,
    )
    end_page = col2.number_input(
        "End page", min_value=1, max_value=total_pages,
        value=min(10, total_pages), step=1,
    )
    if start_page > end_page:
        st.error("Start page must be ≤ end page.")
        return

    top_n = st.slider("Number of top products in the final report", 1, 10, 3)

    # ── Step 4: run ────────────────────────────────────────────────────────────
    st.subheader("3. Extract Requirements & Generate Report")
    run_clicked = st.button(" Extract & Generate Compliance Report", type="primary")

    if run_clicked:
        progress = st.progress(0, text="Starting…")
        log_box = st.empty()

        try:
            progress.progress(10, text="Extracting requirements from selected pages…")
            t0 = time.time()
            extraction_result = extractor.run(
                pdf_path=rfp_path,
                start_page=int(start_page),
                end_page=int(end_page),
                embed=True,
            )
            elapsed_extract = time.time() - t0
            n_reqs = extraction_result["requirement_count"]
            log_box.info(
                f"✅ Extracted {n_reqs} requirement(s) in {elapsed_extract:.0f}s "
                f"→ saved to `{extraction_result['json_path']}`"
            )

            if n_reqs == 0:
                progress.progress(100, text="No requirements found.")
                st.warning(
                    "No requirements were extracted from this page range. "
                    "Try a different range."
                )
                return

            progress.progress(40, text=f"Evaluating compliance for {n_reqs} requirement(s)…")
            reqs = [Requirement(**r) for r in extraction_result["requirements"]]

            engine = get_engine(vs, top_n=top_n)
            t1 = time.time()
            report = engine.run(
                requirements=reqs,
                rfp_source=rfp_path,
                page_range=extraction_result["page_range"],
            )
            elapsed_compliance = time.time() - t1

            progress.progress(100, text="Done.")
            st.session_state.compliance_report = report
            st.session_state.extracted_meta = {
                "n_reqs": n_reqs,
                "elapsed_extract": elapsed_extract,
                "elapsed_compliance": elapsed_compliance,
            }
            st.success(
                f"✅ Compliance report generated in {elapsed_compliance:.0f}s "
                f"(report ID: `{report.report_id}`)"
            )

        except Exception as exc:
            progress.progress(100, text="Failed.")
            st.error(f"Pipeline failed: {exc}")
            with st.expander("Show traceback"):
                st.code(traceback.format_exc())
            return

    # ── Step 5: show result if available ──────────────────────────────────────
    report = st.session_state.get("compliance_report")
    if report is not None:
        st.divider()
        render_report_summary(report)


def render_report_summary(report):
    st.subheader(f"Compliance Report — `{report.report_id}`")

    c1, c2, c3 = st.columns(3)
    c1.metric("Requirements", report.total_requirements)
    c2.metric("Mandatory", report.mandatory_count)
    c3.metric("Optional", report.optional_count)

    if not report.top_products:
        st.warning("No matching products were found in the knowledge base for this RFP.")
        return

    st.markdown("### Top Products")
    for i, p in enumerate(report.top_products, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"#{i}")
        with st.expander(
            f"{medal} **{p.vendor} — {p.model_name}** "
            f"({p.overall_score:.1f}% overall, {p.mandatory_score:.1f}% mandatory)",
            expanded=(i == 1),
        ):
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Overall Score", f"{p.overall_score:.1f}%")
            cc2.metric("Mandatory Score", f"{p.mandatory_score:.1f}%")
            cc3.metric("Full Matches", p.full_matches)
            cc4.metric("No Match", p.no_matches)

            if p.key_gaps:
                st.markdown("**Key gaps (mandatory):**")
                for g in p.key_gaps[:8]:
                    st.markdown(f"- {g}")
                if len(p.key_gaps) > 8:
                    st.caption(f"... and {len(p.key_gaps) - 8} more")

    # Download links
    json_path = REPORTS_DIR / f"compliance_{report.report_id}.json"
    md_path = REPORTS_DIR / f"compliance_{report.report_id}.md"
    st.divider()
    dl1, dl2 = st.columns(2)
    if md_path.exists():
        dl1.download_button(
            " Download Markdown Report",
            data=md_path.read_text(encoding="utf-8"),
            file_name=md_path.name,
            mime="text/markdown",
            use_container_width=True,
        )
    if json_path.exists():
        dl2.download_button(
            " Download JSON Report",
            data=json_path.read_text(encoding="utf-8"),
            file_name=json_path.name,
            mime="application/json",
            use_container_width=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — MANAGE DATABASE
# ──────────────────────────────────────────────────────────────────────────────

def render_manage_db_tab(kb: OEMKnowledgeBase, vs: VectorStoreManager):
    st.header("Manage OEM Knowledge Base")

    # ── Upload new datasheets ────────────────────────────────────────────────
    st.subheader("Add Datasheets")
    new_files = st.file_uploader(
        "Upload one or more OEM datasheet PDFs to embed into the knowledge base",
        type=["pdf"],
        accept_multiple_files=True,
        key="kb_uploader",
    )
    force_reingest = st.checkbox(
        "Force re-ingest (overwrite if already in the KB)", value=False,
    )

    if new_files:
        if st.button(f" Ingest {len(new_files)} file(s) into KB", type="primary"):
            results_box = st.container()
            progress = st.progress(0)
            for i, uf in enumerate(new_files):
                with results_box:
                    with st.spinner(f"Ingesting {uf.name}…"):
                        try:
                            saved_path = save_uploaded_pdf(uf, RAW_DIR)
                            result = kb.ingest_file(saved_path, force=force_reingest)
                            if result.status.value == "completed":
                                st.success(
                                    f"✅ {uf.name}: {result.models_found} model(s), "
                                    f"{result.chunks_created} chunk(s) "
                                    f"({result.processing_time_seconds:.1f}s)"
                                )
                            elif result.status.value == "skipped":
                                st.info(f"⏭️ {uf.name}: already in KB, skipped.")
                            else:
                                st.error(f"❌ {uf.name}: {result.error_message}")
                        except Exception as exc:
                            st.error(f"❌ {uf.name}: {exc}")
                progress.progress((i + 1) / len(new_files))
            # Force fresh stats/inventory after ingesting
            st.cache_resource.clear()
            st.rerun()

    st.divider()

    # ── Current inventory ────────────────────────────────────────────────────
    st.subheader("Current Knowledge Base Contents")

    if st.button("🔄 Refresh"):
        st.rerun()

    inventory = get_kb_inventory(vs)

    if not inventory:
        st.info("The knowledge base is empty. Upload datasheets above to get started.")
        return

    # Vendor filter
    vendors = sorted({row["vendor"] for row in inventory})
    selected_vendor = st.selectbox("Filter by vendor", ["All"] + vendors)
    filtered = (
        inventory if selected_vendor == "All"
        else [r for r in inventory if r["vendor"] == selected_vendor]
    )

    st.caption(f"{len(filtered)} document(s) shown")

    for row in filtered:
        with st.expander(
            f" **{row['filename']}**  —  {row['vendor']}  "
            f"({len(row['models'])} model(s), {row['chunk_count']} chunks)"
        ):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**Vendor:** {row['vendor']}")
                st.markdown(f"**Models:** {', '.join(row['models']) or '—'}")
                st.markdown(f"**Doc ID:** `{row['doc_id']}`")
                st.markdown(f"**Source path:** `{row['source_file']}`")
                if row["chunk_types"]:
                    type_str = ", ".join(
                        f"{k}: {v}" for k, v in sorted(row["chunk_types"].items())
                    )
                    st.caption(f"Chunk types — {type_str}")
            with c2:
                st.metric("Chunks", row["chunk_count"])
                if st.button(" Remove", key=f"del_{row['doc_id']}"):
                    n = kb.delete_document(row["doc_id"])
                    st.success(f"Deleted {n} chunk(s) for {row['filename']}.")
                    st.cache_resource.clear()
                    time.sleep(0.5)
                    st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — REPORTS
# ──────────────────────────────────────────────────────────────────────────────

def render_reports_tab():
    st.header("Generated Compliance Reports")

    reports = list_report_files(REPORTS_DIR)
    if not reports:
        st.info("No compliance reports have been generated yet. "
                 "Use the **Upload RFP** tab to create one.")
        return

    st.caption(f"{len(reports)} report(s) found")

    for row in reports:
        report_id = row["report_id"]
        mtime_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(row["mtime"]))

        with st.expander(f" `{report_id}`  —  {mtime_str}"):
            c1, c2, c3 = st.columns([2, 2, 1])

            if row["md_path"] and row["md_path"].exists():
                md_text = row["md_path"].read_text(encoding="utf-8")
                with c1:
                    st.download_button(
                        " Markdown",
                        data=md_text,
                        file_name=row["md_path"].name,
                        mime="text/markdown",
                        key=f"dl_md_{report_id}",
                        use_container_width=True,
                    )
            if row["json_path"] and row["json_path"].exists():
                json_text = row["json_path"].read_text(encoding="utf-8")
                with c2:
                    st.download_button(
                        " JSON",
                        data=json_text,
                        file_name=row["json_path"].name,
                        mime="application/json",
                        key=f"dl_json_{report_id}",
                        use_container_width=True,
                    )
            with c3:
                if st.button(" Delete", key=f"del_report_{report_id}"):
                    delete_report(REPORTS_DIR, report_id)
                    st.success("Deleted.")
                    time.sleep(0.3)
                    st.rerun()

            if row["md_path"] and row["md_path"].exists():
                st.markdown("---")
                st.markdown(md_text)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    try:
        kb = get_kb()
        vs = get_vector_store(kb)
    except Exception as exc:
        st.error(f"Failed to initialise knowledge base: {exc}")
        with st.expander("Show traceback"):
            st.code(traceback.format_exc())
        st.stop()

    render_sidebar(kb)

    tab1, tab2, tab3 = st.tabs([
        "📤  Upload RFP",
        "🗄️  Manage Database",
        "📑  Reports",
    ])

    with tab1:
        render_upload_rfp_tab(kb, vs)
    with tab2:
        render_manage_db_tab(kb, vs)
    with tab3:
        render_reports_tab()


if __name__ == "__main__":
    main()
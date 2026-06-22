"""
ui/kb_inventory.py
===================
Read-only helper layer for the UI's "Manage Database" tab.

Does NOT modify any backend module. Builds a richer per-document /
per-model inventory view on top of VectorStoreManager by reading
chunk metadata directly (the same data add_chunks() already wrote).

Also tracks "known source files" (the raw PDFs that were ingested) by
mirroring them into data/raw/ — this lets the UI show original
filenames and offer a delete-from-disk option even though the vector
store itself only keys chunks by doc_id (sha256 of file content).
"""
from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def get_kb_inventory(vector_store) -> List[Dict[str, Any]]:
    """
    Build one row per ingested document, aggregating model names and
    chunk counts from the chunk-level metadata already stored in Chroma.

    Returns a list of dicts:
      {
        "doc_id": str,
        "vendor": str,
        "source_file": str,        # path as stored at ingestion time
        "filename": str,           # basename, for display
        "models": List[str],       # unique model names found in this doc
        "chunk_count": int,
        "chunk_types": Dict[str,int],
      }
    """
    if vector_store._collection is None:
        return []

    try:
        sample = vector_store._collection.get(
            limit=20000,
            include=["metadatas"],
        )
    except Exception:
        return []

    by_doc: Dict[str, Dict[str, Any]] = {}
    for m in sample.get("metadatas", []):
        doc_id = m.get("doc_id", "")
        if not doc_id:
            continue
        entry = by_doc.setdefault(doc_id, {
            "doc_id": doc_id,
            "vendor": m.get("vendor", ""),
            "source_file": m.get("source_file", ""),
            "models": set(),
            "chunk_count": 0,
            "chunk_types": defaultdict(int),
        })
        entry["chunk_count"] += 1
        model_name = m.get("model_name", "")
        if model_name:
            entry["models"].add(model_name)
        ct = m.get("chunk_type", "unknown")
        entry["chunk_types"][ct] += 1
        # Keep the most informative vendor/source_file seen
        if not entry["vendor"] and m.get("vendor"):
            entry["vendor"] = m.get("vendor")
        if not entry["source_file"] and m.get("source_file"):
            entry["source_file"] = m.get("source_file")

    rows = []
    for doc_id, entry in by_doc.items():
        src = entry["source_file"] or ""
        rows.append({
            "doc_id": doc_id,
            "vendor": entry["vendor"] or "Unknown",
            "source_file": src,
            "filename": Path(src).name if src else doc_id[:16],
            "models": sorted(entry["models"]),
            "chunk_count": entry["chunk_count"],
            "chunk_types": dict(entry["chunk_types"]),
        })

    rows.sort(key=lambda r: (r["vendor"], r["filename"]))
    return rows


def save_uploaded_pdf(uploaded_file, raw_dir: Path) -> Path:
    """
    Persist an uploaded PDF (Streamlit UploadedFile) into the pipeline's
    raw/ directory so OEMKnowledgeBase.ingest_file() can read it from a
    real path. Returns the saved path.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / uploaded_file.name
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


def save_uploaded_rfp(uploaded_file, rfp_raw_dir: Path) -> Path:
    """Persist an uploaded RFP PDF for the RFP-upload tab."""
    rfp_raw_dir.mkdir(parents=True, exist_ok=True)
    dest = rfp_raw_dir / uploaded_file.name
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


def list_report_files(reports_dir: Path) -> List[Dict[str, Any]]:
    """
    Pair up compliance_<id>.json / .md files in REPORTS_DIR into one
    row per report_id, newest first.
    """
    if not reports_dir.exists():
        return []

    by_id: Dict[str, Dict[str, Any]] = {}
    for f in reports_dir.glob("compliance_*.*"):
        # filename pattern: compliance_<report_id>.json | .md
        stem = f.stem  # compliance_<report_id>
        report_id = stem[len("compliance_"):]
        entry = by_id.setdefault(report_id, {
            "report_id": report_id,
            "json_path": None,
            "md_path": None,
            "mtime": 0.0,
        })
        if f.suffix == ".json":
            entry["json_path"] = f
        elif f.suffix == ".md":
            entry["md_path"] = f
        entry["mtime"] = max(entry["mtime"], f.stat().st_mtime)

    rows = list(by_id.values())
    rows.sort(key=lambda r: -r["mtime"])
    return rows


def delete_report(reports_dir: Path, report_id: str) -> None:
    for ext in (".json", ".md"):
        f = reports_dir / f"compliance_{report_id}{ext}"
        if f.exists():
            f.unlink()


def list_requirement_jsons(rfp_dir: Path) -> List[Path]:
    """All saved requirement-extraction JSON files, newest first."""
    if not rfp_dir.exists():
        return []
    files = [f for f in rfp_dir.glob("*.json")]
    files.sort(key=lambda f: -f.stat().st_mtime)
    return files
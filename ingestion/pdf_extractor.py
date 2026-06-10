from __future__ import annotations

import hashlib
import io
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
import fitz  # PyMuPDF
import pdfplumber
from loguru import logger

def clean_text(text: str) -> str:
    if not text:
        return ""
    replacements = {
        "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
        "\x00": "", "\xad": "-", "–": "-", "—": "-", "\u2022": "•",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    
    # Fast regex normalization instead of heavy multi-pass loops
    text = re.sub(r'[^\x09\x0A\x20-\x7E•μ]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())

def extract_page_data(page_layout) -> Tuple[str, List[dict]]:
    """Extracts text and structured tables simultaneously in a single pass."""
    text = page_layout.extract_text() or ""
    tables = []
    
    try:
        found_tables = page_layout.find_tables()
        for idx, table_layout in enumerate(found_tables):
            raw_table = table_layout.extract()
            if not raw_table or len(raw_table) < 2:
                continue
            
            headers = [str(h or "").strip() for h in raw_table[0]]
            rows = [[str(cell or "").strip() for cell in row] for row in raw_table[1:]]
            
            tables.append({
                "table_index": idx,
                "headers": headers,
                "rows": rows,
                "raw_text": f"Table Headers: {', '.join(headers)}"
            })
    except Exception as e:
        logger.debug(f"Table parsing skipped on layout pass: {e}")
        
    return text, tables

def extract_document(pdf_path: str | Path, cfg) -> Tuple[List[dict], str]:
    """Fast single-pass document extractor using PyMuPDF as the lightning primary driver."""
    pages_data = []
    path_str = str(pdf_path)
    
    # Run primary text extraction through fitz (orders of magnitude faster than pdfplumber)
    with fitz.open(path_str) as doc:
        has_text = any(len(page.get_text("text").strip()) > 50 for page in doc[:3])
        method = "text" if has_text else "ocr"
        
        if method == "text":
            with pdfplumber.open(path_str) as pdf:
                for idx, page in enumerate(pdf.pages):
                    raw_text, tables = extract_page_data(page)
                    cleaned = clean_text(raw_text)
                    pages_data.append({
                        "page_number": idx + 1,
                        "cleaned_text": cleaned,
                        "tables": tables
                    })
            return pages_data, "text"
            
    # Simple Fallback tracking if completely scanned (OCR routing layer)
    logger.warning(f"No selectable text found in {pdf_path}. Routing to fallback layer.")
    return pages_data, "failed_or_empty"
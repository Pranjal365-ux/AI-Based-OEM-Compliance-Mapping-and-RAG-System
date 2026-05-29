# =============================================================================
# extractor.py — PDF text extraction with table handling + PaddleOCR fallback
# =============================================================================
#
# Strategy per page:
#   1. PyMuPDF text extraction (fast, accurate for digital PDFs)
#   2. PyMuPDF find_tables() → Markdown (handles ruled and borderless tables)
#      Table regions are masked from raw text to prevent double-extraction.
#   3. If text < OCR_CHAR_THRESHOLD chars → render page image → PaddleOCR
# =============================================================================

import re
import logging
import fitz
from config import OCR_CHAR_THRESHOLD, OCR_DPI

logger = logging.getLogger("ingestion")

_paddle_ocr = None

def _get_ocr():
    global _paddle_ocr
    if _paddle_ocr is None:
        try:
            from paddleocr import PaddleOCR
            _paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            logger.info("  [OCR] PaddleOCR loaded.")
        except ImportError:
            logger.warning("  [OCR] PaddleOCR not installed — image pages will be skipped.")
            _paddle_ocr = "unavailable"
    return _paddle_ocr


def _extract_tables_as_markdown(page: fitz.Page) -> list[dict]:
    tables_md = []
    try:
        tab_finder = page.find_tables()
        for table in tab_finder.tables:
            df = table.to_pandas()
            if df.empty:
                continue

            def clean_cell(v):
                if v is None:
                    return ""
                return re.sub(r"\s+", " ", str(v)).strip()

            headers = list(df.columns)
            rows    = df.values.tolist()

            header_line = " | ".join(clean_cell(h) for h in headers)
            sep_line    = " | ".join(["---"] * len(headers))
            data_lines  = [
                " | ".join(clean_cell(c) for c in row)
                for row in rows
            ]
            md = "\n".join([header_line, sep_line] + data_lines)
            tables_md.append({"bbox": table.bbox, "markdown": md})

    except Exception as e:
        logger.debug(f"  [table] Skipped: {e}")

    return tables_md


def _bbox_to_rect(bbox) -> fitz.Rect:
    return bbox if isinstance(bbox, fitz.Rect) else fitz.Rect(bbox)


def _ocr_page(page: fitz.Page) -> str:
    ocr = _get_ocr()
    if ocr == "unavailable":
        return ""
    import numpy as np
    mat = fitz.Matrix(OCR_DPI / 72, OCR_DPI / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    try:
        results = ocr.ocr(img, cls=True)
        if not results or not results[0]:
            return ""
        lines = [
            line[1][0] for line in results[0]
            if line[1][1] > 0.6 and line[1][0].strip()
        ]
        return " ".join(lines)
    except Exception as e:
        logger.warning(f"  [OCR] Failed: {e}")
        return ""


def extract_pages(pdf_path: str) -> list[dict]:
    """
    Extract all pages. Returns list of:
    { page: int, text: str, has_tables: bool, ocr_used: bool }
    """
    doc = fitz.open(pdf_path)
    pages_out = []

    for page_num, page in enumerate(doc):
        pd = {"page": page_num + 1, "text": "", "has_tables": False, "ocr_used": False}

        # Tables first
        tables     = _extract_tables_as_markdown(page)
        t_regions  = [_bbox_to_rect(t["bbox"]) for t in tables]
        t_md_text  = "\n\n".join(t["markdown"] for t in tables)
        if tables:
            pd["has_tables"] = True

        # Raw text — skip blocks that overlap table regions
        if t_regions:
            blocks   = page.get_text("blocks")
            raw_parts = [
                b[4] for b in blocks
                if not any(fitz.Rect(b[:4]).intersects(r) for r in t_regions)
            ]
        else:
            raw_parts = [page.get_text("text")]

        raw_text = "\n".join(raw_parts).strip()

        combined = raw_text
        if t_md_text:
            combined += ("\n\n" if combined else "") + "<!-- TABLE -->\n" + t_md_text

        # OCR fallback
        if len(combined.strip()) < OCR_CHAR_THRESHOLD:
            logger.info(f"  [OCR] Page {page_num+1}: {len(combined)} chars → OCR")
            ocr_text = _ocr_page(page)
            if ocr_text:
                combined        = ocr_text
                pd["ocr_used"]  = True
            else:
                logger.warning(f"  [OCR] Page {page_num+1}: empty, skipping.")

        if combined.strip():
            pd["text"] = combined
            pages_out.append(pd)

    doc.close()
    logger.info(
        f"  Extracted {len(pages_out)} pages | "
        f"OCR: {sum(p['ocr_used'] for p in pages_out)} | "
        f"Tables: {sum(p['has_tables'] for p in pages_out)}"
    )
    return pages_out

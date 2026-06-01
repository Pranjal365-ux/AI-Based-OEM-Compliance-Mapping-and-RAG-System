"""
OEM Datasheet Ingestion Pipeline - PDF Extraction Utilities
Handles text-based PDFs, scanned PDFs (OCR), and mixed documents.
"""
from __future__ import annotations

import hashlib
import io
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
import pdfplumber
from loguru import logger
from PIL import Image

from config.settings import OCRConfig, PDFConfig

try:
    from paddleocr import PaddleOCR

    PADDLE_AVAILABLE = True
    OCR_ENGINE = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="en"
    )
except ImportError:
    PADDLE_AVAILABLE = False
    logger.warning("PaddleOCR not available – OCR fallback disabled")
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available – image preprocessing for OCR limited")


# ─── File Utilities ─────────────────────────────────────────────────────────────

def compute_file_hash(file_path: str | Path) -> str:
    """SHA-256 of file content — used as stable document ID."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]   # 16 hex chars is plenty for an ID


# ─── Page-Level OCR ─────────────────────────────────────────────────────────────

def preprocess_image_for_ocr(pil_image: Image.Image) -> Image.Image:
    """
    Apply image-processing steps that improve paddle accuracy:
    - Convert to grayscale
    - Deskew (if OpenCV available)
    - Binarize with Otsu's threshold (if OpenCV available)
    """
    img = pil_image.convert("L")  # Grayscale

    if CV2_AVAILABLE:
        import numpy as np
        arr = np.array(img)
        # Deskew
        coords = np.column_stack(np.where(arr < 200))
        if len(coords) > 100:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            if abs(angle) < 15:          # Only correct small skew
                h, w = arr.shape
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                arr = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
        # Binarize
        _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = Image.fromarray(arr)

    return img


def ocr_page_image(
    pil_image: Image.Image,
    ocr_cfg: OCRConfig,
    psm: Optional[int] = None
    ) -> Tuple[str, float]:

    if not PADDLE_AVAILABLE:
        return "", 0.0

    try:
        import numpy as np

        img = np.array(pil_image.convert("RGB"))

        result = OCR_ENGINE.predict(img)
        texts = []
        confidences = []

        for page in result:
            if "rec_texts" in page:
                texts.extend(page["rec_texts"])

            if "rec_scores" in page:
                confidences.extend(page["rec_scores"])

        text = "\n".join(texts)

        mean_conf = (
            sum(confidences) / len(confidences)
            if confidences else 0.0
        )

        return text, mean_conf

    except Exception as e:
        import traceback

        logger.error(traceback.format_exc())
        logger.warning(f"PaddleOCR failed: {e}")
        return "", 0.0
    
def rasterize_page(
    pdf_path: str | Path,
    page_index: int,          # 0-indexed
    dpi: int = 300
) -> Optional[Image.Image]:
    """Rasterize a single PDF page to a PIL Image using PyMuPDF."""
    try:
        doc = fitz.open(str(pdf_path))
        page = doc[page_index]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    except Exception as e:
        logger.error(f"Failed to rasterize page {page_index+1}: {e}")
        return None


# ─── Vendor Detection from Header ──────────────────────────────────────────────

def extract_header_image(
    pdf_path: str | Path,
    page_index: int = 0,
    crop_fraction: float = 0.30,
    dpi: int = 300
) -> Optional[Image.Image]:
    """
    Extract the top `crop_fraction` of a page as a PIL image.
    Used to OCR the vendor logo / letterhead area.
    """
    img = rasterize_page(pdf_path, page_index, dpi=dpi)
    if img is None:
        return None
    w, h = img.size
    crop_h = int(h * crop_fraction)
    return img.crop((0, 0, w, crop_h))


def detect_vendor_from_header(
    pdf_path: str | Path,
    ocr_cfg: OCRConfig,
    pages_to_check: int = 1,
) -> Tuple[str, float, str]:
    """
    Attempt to identify the vendor by OCR-ing the top portion of the
    first 1-2 pages of the PDF.

    Returns:
        (vendor_name, confidence, raw_header_text)
    """
    all_header_texts = []

    for page_idx in range(min(pages_to_check, _get_page_count(pdf_path))):
        header_img = extract_header_image(
            pdf_path, page_idx,
            crop_fraction=ocr_cfg.header_crop_fraction,
            dpi=ocr_cfg.dpi
        )
        if header_img is None:
            continue

        # NOTE: Do NOT binarize header images — vendor logos are often colored
        # images that Otsu thresholding destroys.  Send the raw RGB crop to
        # PaddleOCR which handles color/grayscale images natively.
        text, conf = ocr_page_image(header_img, ocr_cfg, psm=ocr_cfg.psm_header)
        if text.strip():
            all_header_texts.append(text.strip())

    if not all_header_texts:
        return "Unknown", 0.0, ""

    combined = "\n".join(all_header_texts)
    vendor, confidence = _parse_vendor_from_text(combined)
    return vendor, confidence, combined


def _get_page_count(pdf_path: str | Path) -> int:
    try:
        doc = fitz.open(str(pdf_path))
        n = len(doc)
        doc.close()
        return n
    except Exception:
        return 0


def _parse_vendor_from_text(header_text: str) -> Tuple[str, float]:
    """
    Heuristic: the vendor name is usually the FIRST non-trivial line
    or a short all-caps word near the top.
    """
    lines = [ln.strip() for ln in header_text.splitlines() if ln.strip()]
    if not lines:
        return "Unknown", 0.0

    known_vendors = {
        # Network & Security
        "fortinet": "Fortinet",
        "cisco": "Cisco",
        "palo alto": "Palo Alto Networks",
        "paloalto": "Palo Alto Networks",
        "palo alto networks": "Palo Alto Networks",
        "juniper": "Juniper Networks",
        "check point": "Check Point",
        "checkpoint": "Check Point",
        "sonicwall": "SonicWall",
        "sophos": "Sophos",
        "barracuda": "Barracuda Networks",
        "watchguard": "WatchGuard",
        "f5": "F5",
        "f5 networks": "F5",
        "radware": "Radware",
        "netscout": "NETSCOUT",
        "netskope": "Netskope",
        "zscaler": "Zscaler",
        "forescout": "Forescout",
        "arista": "Arista Networks",
        "extreme": "Extreme Networks",
        "extreme networks": "Extreme Networks",
        "ubiquiti": "Ubiquiti",
        "netgear": "Netgear",
        "aruba": "Aruba Networks",

        # Endpoint / Email / Security
        "trendmicro": "Trend Micro",
        "trend micro": "Trend Micro",
        "trellix": "Trellix",
        "mcafee": "Trellix",
        "ivanti": "Ivanti",
        "cyberark": "CyberArk",
        "opswat": "OPSWAT",
        "sysdig": "Sysdig",

        # Threat Intelligence / ASM
        "cyble": "Cyble",
        "cloudsek": "CloudSEK",
        "recorded future": "Recorded Future",
        "crowdstrike": "CrowdStrike",
        "microsoft defender": "Microsoft",
        "sentinelone": "SentinelOne",
        "rapid7": "Rapid7",
        "tenable": "Tenable",
        "qualys": "Qualys",
        "varonis": "Varonis",

        # PKI / Encryption / HSM
        "emudhra": "eMudhra",
        "e mudhra": "eMudhra",
        "utimaco": "Utimaco",
        "thales": "Thales",
        "entrust": "Entrust",

        # PAM / Identity
        "arcon": "ARCON",
        "okta": "Okta",
        "ping identity": "Ping Identity",
        "forgerock": "ForgeRock",
        "sailpoint": "SailPoint",

        # DLP / Data Security
        "forcepoint": "Forcepoint",
        "symantec": "Broadcom",
        "broadcom": "Broadcom",
        "digital guardian": "Digital Guardian",
        "proofpoint": "Proofpoint",

        # Data Governance / Discovery
        "data resolve": "Data Resolve",
        "dataresolve": "Data Resolve",

        # Cloud Security
        "wiz": "Wiz",
        "prisma cloud": "Palo Alto Networks",
        "lacework": "Lacework",
        "orca security": "Orca Security",

        # SIEM / Observability / Analytics
        "elastic": "Elastic",
        "elk": "Elastic",
        "splunk": "Splunk",
        "sumologic": "Sumo Logic",
        "datadog": "Datadog",
        "dynatrace": "Dynatrace",
        "new relic": "New Relic",
        "grafana": "Grafana Labs",
        "opentext": "OpenText",
        "open text": "OpenText",

        # CDN / WAF / DNS
        "cloudflare": "Cloudflare",
        "cloud flare": "Cloudflare",
        "akamai": "Akamai",
        "imperva": "Imperva",

        # Infrastructure / Compute
        "hp ": "HP",
        "hewlett packard": "HP",
        "hpe": "HPE",
        "hewlett packard enterprise": "HPE",
        "dell": "Dell Technologies",
        "lenovo": "Lenovo",
        "ibm": "IBM",
        "intel": "Intel",
        "amd": "AMD",
        "nvidia": "NVIDIA",
        "oracle": "Oracle",

        # Storage / Hyperconverged
        "hitachi": "Hitachi Vantara",
        "hitachi vantara": "Hitachi Vantara",
        "netapp": "NetApp",
        "nutanix": "Nutanix",
        "pure storage": "Pure Storage",
        "emc": "Dell EMC",
        "dell emc": "Dell EMC",

        # Virtualization
        "vmware": "VMware",
        "citrix": "Citrix",
        "red hat": "Red Hat",

        # Physical Security
        "axis": "Axis Communications",
        "hikvision": "Hikvision",
        "dahua": "Dahua Technology",
        "bosch": "Bosch Security",
        "honeywell": "Honeywell",

        # Industrial / OT
        "schneider": "Schneider Electric",
        "siemens": "Siemens",
        "rockwell": "Rockwell Automation",
        "abb": "ABB",

        # Backup & Recovery
        "veeam": "Veeam",
        "commvault": "Commvault",
        "rubrik": "Rubrik",
        "cohesity": "Cohesity",

        # Public Cloud
        "aws": "Amazon Web Services",
        "amazon web services": "Amazon Web Services",
        "azure": "Microsoft Azure",
        "microsoft azure": "Microsoft Azure",
        "gcp": "Google Cloud",
        "google cloud": "Google Cloud",
    }

    text_lower = header_text.lower()
    for key, name in sorted(known_vendors.items(), key=lambda item: -len(item[0])):
        if _vendor_key_in_text(key, text_lower):
            return name, 0.9

    # Fallback: take first line that looks like a company name
    for line in lines[:5]:
        clean = re.sub(r'[^A-Za-z0-9\s\-&.,]', '', line).strip()
        words = clean.split()
        # Skip pure-numeric lines (page numbers, "1", etc.) and single-char noise
        if not words or all(w.isdigit() for w in words) or len(clean) <= 2:
            continue
        if 1 <= len(words) <= 6:
            return clean.title(), 0.5

    # Last resort: return first non-numeric, non-trivial line
    for line in lines:
        clean = re.sub(r'[^A-Za-z0-9\s\-&.,]', '', line).strip()
        if clean and not clean.isdigit() and len(clean) > 2:
            return clean[:50].title(), 0.3

    return "Unknown", 0.1


def _vendor_key_in_text(key: str, text_lower: str) -> bool:
    """Match vendor names as terms, so 'intel' does not match 'intelligence'."""
    key = key.strip().lower()
    if not key:
        return False
    pattern = r"(?<![a-z0-9])" + r"\s+".join(
        re.escape(part) for part in key.split()
    ) + r"(?![a-z0-9])"
    return re.search(pattern, text_lower) is not None


# ─── Full Document Text Extraction ─────────────────────────────────────────────

class PageExtractor:
    """
    Extracts text and tables from a single page, with automatic
    OCR fallback for scanned / image-based pages.
    """

    def __init__(self, pdf_cfg: PDFConfig, ocr_cfg: OCRConfig):
        self.pdf_cfg = pdf_cfg
        self.ocr_cfg = ocr_cfg

    def is_page_scanned(self, page_text: str) -> bool:
        """True if the text layer has too little content to be reliable."""
        return len(page_text.strip()) < self.pdf_cfg.min_text_chars_per_page

    def extract_text_pdfplumber(
        self, plumber_page
    ) -> Tuple[str, List]:
        """Extract text + tables from a pdfplumber page object."""
        text = plumber_page.extract_text() or ""
        raw_tables = plumber_page.extract_tables() or []
        return text, raw_tables

    def extract_text_pymupdf(self, fitz_page) -> str:
        """Extract text via PyMuPDF (handles some PDFs better than pdfplumber)."""
        return fitz_page.get_text("text")

    def extract_page(
        self,
        pdf_path: str | Path,
        page_index: int,        # 0-indexed
        plumber_page=None,
        fitz_page=None,
    ) -> dict:
        """
        Main extraction entry point for a single page.
        Returns a dict suitable for constructing PageContent.
        """
        result = {
            "page_number": page_index + 1,
            "raw_text": "",
            "cleaned_text": "",
            "tables": [],
            "extraction_method": "pdfplumber",
            "ocr_confidence": None,
            "is_scanned": False,
            "has_text": False,
        }

        # ── Attempt 1: pdfplumber ─────────────────────────────────────────────
        text = ""
        raw_tables = []
        if plumber_page is not None:
            try:
                text, raw_tables = self.extract_text_pdfplumber(plumber_page)
            except Exception as e:
                logger.warning(f"pdfplumber failed on page {page_index+1}: {e}")

        # ── Attempt 2: PyMuPDF fallback ───────────────────────────────────────
        if self.is_page_scanned(text) and fitz_page is not None:
            try:
                text_fitz = self.extract_text_pymupdf(fitz_page)
                if len(text_fitz) > len(text):
                    text = text_fitz
                    result["extraction_method"] = "pymupdf"
            except Exception as e:
                logger.warning(f"PyMuPDF failed on page {page_index+1}: {e}")

        # ── Attempt 3: OCR fallback ───────────────────────────────────────────
        if self.is_page_scanned(text) and PADDLE_AVAILABLE:
            logger.info(f"Page {page_index+1} appears scanned → applying OCR")
            img = rasterize_page(pdf_path, page_index, dpi=self.ocr_cfg.dpi)
            if img is not None:
                img = preprocess_image_for_ocr(img)
                ocr_text, conf = ocr_page_image(img, self.ocr_cfg)
                if len(ocr_text) > len(text):
                    text = ocr_text
                    result["extraction_method"] = "ocr_paddle"
                    result["ocr_confidence"] = conf
                    result["is_scanned"] = True

        result["raw_text"] = text
        result["cleaned_text"] = clean_text(text)
        result["has_text"] = bool(result["cleaned_text"].strip())
        result["tables"] = self._parse_raw_tables(raw_tables, page_index + 1)
        return result

    def _parse_raw_tables(
        self, raw_tables: List, page_number: int
    ) -> List[dict]:
        """Convert raw pdfplumber table data into structured dicts."""
        parsed = []
        for idx, table in enumerate(raw_tables):
            if not table:
                continue
            # Treat first row as header if it looks like one
            rows = [[str(c or "").strip() for c in row] for row in table]
            if not _table_has_useful_text(rows):
                continue
            headers: List[str] = []
            data_rows = rows

            if rows:
                first_row = rows[0]
                # Heuristic: header if cells are short text, no numbers dominate
                if all(not re.match(r"^\d+(\.\d+)?$", c) for c in first_row if c):
                    headers = first_row
                    data_rows = rows[1:]

            flat_rows = rows if not headers else [headers] + data_rows
            flat_text = "\n".join(
                " | ".join(r) for r in flat_rows if any(c.strip() for c in r)
            )
            parsed.append({
                "page_number": page_number,
                "table_index": idx,
                "headers": headers,
                "rows": data_rows,
                "raw_text": flat_text,
            })
        return parsed


def _table_has_useful_text(rows: List[List[str]]) -> bool:
    text_cells = [cell for row in rows for cell in row if cell.strip()]
    if not text_cells:
        return False
    text = " ".join(text_cells)
    return len(text) >= 3


# ─── Document-Level Extraction ──────────────────────────────────────────────────

def extract_document(
    pdf_path: str | Path,
    pdf_cfg: PDFConfig,
    ocr_cfg: OCRConfig,
) -> Tuple[List[dict], str]:
    """
    Extract all pages from a PDF file.

    Returns:
        (list_of_page_dicts, extraction_method_summary)
    """
    pdf_path = Path(pdf_path)
    extractor = PageExtractor(pdf_cfg, ocr_cfg)
    pages = []

    # Check if PDF has a text layer at all (fast check via pdffonts)
    has_text_layer = _check_text_layer(pdf_path)

    max_pages = pdf_cfg.max_pages

    try:
        # Open both libraries in parallel for maximum coverage
        fitz_doc = fitz.open(str(pdf_path))
        total_pages = len(fitz_doc)
        if max_pages:
            total_pages = min(total_pages, max_pages)

        with pdfplumber.open(str(pdf_path)) as plumber_doc:
            for i in range(total_pages):
                plumber_page = plumber_doc.pages[i] if i < len(plumber_doc.pages) else None
                fitz_page = fitz_doc[i]

                page_data = extractor.extract_page(
                    pdf_path, i,
                    plumber_page=plumber_page,
                    fitz_page=fitz_page,
                )
                pages.append(page_data)

        fitz_doc.close()

    except Exception as e:
        logger.error(f"Document extraction failed for {pdf_path.name}: {e}")
        raise

    methods_used = set(p["extraction_method"] for p in pages)
    return pages, "+".join(sorted(methods_used))


def _check_text_layer(pdf_path: Path) -> bool:
    """
    Quick check: does the PDF have embedded fonts (text layer)?
    Uses pdffonts if available, otherwise PyMuPDF.
    """
    try:
        result = subprocess.run(
            ["pdffonts", str(pdf_path)],
            capture_output=True, text=True, timeout=10
        )
        # pdffonts outputs a header + one line per font
        lines = result.stdout.strip().splitlines()
        return len(lines) > 2   # More than just the header → fonts present
    except FileNotFoundError:
        # pdffonts not installed; fall back to PyMuPDF
        try:
            doc = fitz.open(str(pdf_path))
            page = doc[0]
            text = page.get_text("text")
            doc.close()
            return len(text.strip()) > 50
        except Exception:
            return False


# ─── Text Cleaning ───────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean extracted text:
    - Normalize whitespace
    - Remove control characters
    - Fix common OCR ligature errors
    - Remove page numbers / headers that are just numbers
    """
    if not text:
        return ""

    # Fix common OCR issues
    replacements = {
        "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
        "\x00": "", "\xad": "-",   # soft hyphen
        "–": "-", "—": "-",        # em/en dash → hyphen
        "\u2022": "•",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Remove non-printable control chars (keep \n \t)
    text = re.sub(r'[^\x09\x0A\x20-\x7E\x80-\xFF]', ' ', text)

    # Collapse multiple spaces; normalize line endings
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    return text

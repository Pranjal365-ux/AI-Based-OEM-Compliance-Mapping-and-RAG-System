# rfp/rfp_extractor.py
"""
RFP Requirement Extraction Pipeline
====================================

New flow
--------
  PDF
   └─ extract_pages()          → List[PageText]          (page-wise, preserves page numbers)
   └─ discover_products()      → ProductManifest          (1 LLM call on a page-digest)
   └─ ── user selects product ──
   └─ extract_for_product()    → List[Requirement]        (chunks only the relevant pages)

Public surface
--------------
  extractor = RFPRequirementExtractor()

  # Step 1 – load and discover
  pages    = extractor.extract_pages(pdf_path)
  manifest = extractor.discover_products(pages)
  # manifest.products  →  List[ProductEntry]
  # each entry: { product, start_page, end_page, page_count }

  # Step 2 – user picks a product, then:
  reqs = extractor.extract_for_product(pages, manifest.products[i])
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, List, Tuple

import fitz
from models.schemas import Requirement
from services.llm_services import llm

logger = logging.getLogger(__name__)

# ── tunables ──────────────────────────────────────────────────────────────────
CHUNK_SIZE                   = 3_000   # chars per LLM extraction call
MAX_WORKERS                  = 8       # parallel extraction threads

# Discovery: generous budgets so the LLM receives enough signal
DISCOVERY_PAGE_CHAR_BUDGET   = 600     # chars kept per page in the digest
DISCOVERY_DIGEST_CHAR_BUDGET = 25_000  # total digest cap (covers ~40 pages)

# How many chars of raw page text to show per page in the digest (not just headings)
DISCOVERY_RAW_CHARS_PER_PAGE = 400
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PageText:
    """Raw text extracted from a single PDF page."""
    page_number: int          # 1-based
    text: str


@dataclass
class ProductEntry:
    """One product/service discovered in the RFP, with its page boundaries."""
    product:    str
    start_page: int           # 1-based, inclusive
    end_page:   int           # 1-based, inclusive
    pages: List[int] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        if self.pages:
            return len(self.pages)
        return self.end_page - self.start_page + 1

    def as_dict(self) -> dict:
        return {
            "product":    self.product,
            "start_page": self.start_page,
            "end_page":   self.end_page,
            "pages":      self.pages or list(range(self.start_page, self.end_page + 1)),
            "page_count": self.page_count,
        }


@dataclass
class ProductManifest:
    """Full list of products discovered in the RFP."""
    products: List[ProductEntry] = field(default_factory=list)

    def display(self) -> str:
        """Human-readable summary for printing to the user."""
        if not self.products:
            return "No products discovered."
        lines = ["Discovered products:", ""]
        for i, p in enumerate(self.products):
            page_text = self._format_pages(p.pages) if p.pages else f"{p.start_page}–{p.end_page}"
            lines.append(
                f"  [{i}]  {p.product:<40}  pages {page_text}"
                f"  ({p.page_count} page{'s' if p.page_count != 1 else ''})"
            )
        return "\n".join(lines)

    def as_dict_list(self) -> list[dict]:
        return [p.as_dict() for p in self.products]

    @staticmethod
    def _format_pages(pages: List[int]) -> str:
        if not pages:
            return ""
        ranges: List[str] = []
        start = prev = pages[0]
        for page in pages[1:]:
            if page == prev + 1:
                prev = page
                continue
            ranges.append(str(start) if start == prev else f"{start}–{prev}")
            start = prev = page
        ranges.append(str(start) if start == prev else f"{start}–{prev}")
        return ", ".join(ranges)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class RFPRequirementExtractor:
    """
    Two-phase RFP extractor.

    Phase 1 – Discovery
        extract_pages()      →  page-wise text with preserved page numbers
        discover_products()  →  ProductManifest
                                Product names taken from the RFP verbatim.

    Phase 2 – Extraction  (after user selects a product)
        extract_for_product()  →  List[Requirement]
    """

    QUANT_PATTERN = re.compile(
        r"(?P<metric>.+?)"
        r"(?:>=|=>|at least|minimum|min\.?)\s*"
        r"(?P<value>\d+(?:\.\d+)?)\s*"
        r"(?P<unit>Gbps|Mbps|TB|GB|MB|Users|Sessions|EPS)",
        re.IGNORECASE,
    )

    _THINK_RE       = re.compile(r"<think>.*?</think>", re.DOTALL)
    _FENCE_OPEN_RE  = re.compile(r"^```(?:json)?\s*",   re.MULTILINE)
    _FENCE_CLOSE_RE = re.compile(r"```\s*$",            re.MULTILINE)

    # ── heading patterns ──────────────────────────────────────────────────────

    _NUMBERED_HEADING_RE = re.compile(
        r"^(?P<num>\d+(?:\.\d+)*)"   # section number  e.g. "3.1.2"
        r"[\s\.\):]+"                # separator
        r"(?P<title>[A-Za-z].{2,})"  # title must start with a letter
    )

    _UNIT_WORDS = frozenset({
        "GB", "MB", "TB", "KB", "GHZ", "MHZ", "KHZ", "HZ",
        "GBPS", "MBPS", "KBPS", "BPS", "MS", "SEC", "MIN",
        "HTTP", "HTTPS", "FTP", "SSH", "TCP", "UDP", "IP",
        "N/A", "NA", "TBD", "ID", "NO", "OK", "VS",
    })

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 1A – PAGE EXTRACTION
    # ──────────────────────────────────────────────────────────────────────────

    def extract_pages(self, pdf_path: str) -> List[PageText]:
        """
        Extract text page-by-page.  Returns one PageText per PDF page (1-based).
        Empty pages are included so page numbers stay accurate.
        """
        print(f"Opening PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        pages: List[PageText] = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            pages.append(PageText(page_number=i, text=text))
        doc.close()
        total_chars = sum(len(p.text) for p in pages)
        print(f"Pages extracted: {len(pages)}  |  Total chars: {total_chars}")
        return pages

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 1B – PRODUCT DISCOVERY
    # ──────────────────────────────────────────────────────────────────────────

    def discover_products(self, pages: List[PageText]) -> ProductManifest:
        """
        Identify every product/service in the RFP and determine which pages
        cover each product.

        Strategy
        --------
        1. Build a rich page digest with headings + raw text snippets.
        2. One LLM call on that digest → JSON array of product entries.
        3. If LLM fails or returns empty, fall back to heading-based heuristic.
        4. Parse, validate, return ProductManifest.
        """
        print("\n=== PRODUCT DISCOVERY ===")
        digest = self._build_page_digest(pages)
        print(f"[Discovery] Digest: {len(digest):,} chars  ({len(pages)} pages)")

        manifest = self._llm_discover_products(digest, total_pages=len(pages))

        if not manifest.products:
            print("LLM discovery returned nothing; using heuristic fallback")
            manifest = self._heuristic_discover_products(pages)

        print(f"\n{manifest.display()}")
        return manifest

    # ── heading detection ─────────────────────────────────────────────────────

    def _is_heading_line(self, line: str) -> tuple[bool, int]:
        """
        Returns (is_heading, depth).
        depth = number of numeric components in the section number.
        ALL-CAPS headings get depth=1.
        """
        if not line or len(line) > 120:
            return False, 0

        if any(c in line for c in ("://", "\\", ".com", ".org", ".pdf")):
            return False, 0

        # (A) numbered heading
        m = self._NUMBERED_HEADING_RE.match(line)
        if m:
            title = m.group("title").strip()
            title_words = title.split()
            if sum(1 for c in title if c.isalpha()) < 2:
                return False, 0
            if len(title_words) == 1 and title_words[0].upper() in (
                self._UNIT_WORDS | {"M", "K", "G", "T", "MHZ", "GHZ", "MBPS", "GBPS", "MS", "KB", "MB", "GB", "TB", "HZ"}
            ):
                return False, 0
            depth = len(m.group("num").split("."))
            return True, depth

        # (B) ALL-CAPS heading (≥ 2 words, all caps, no units)
        words = line.split()
        if len(words) < 2 or len(line) < 6:
            return False, 0
        if not all(w.replace("-", "").replace("/", "").replace("&", "").isupper() for w in words):
            return False, 0
        upper_words = {w.upper() for w in words}
        if upper_words & self._UNIT_WORDS:
            return False, 0
        if any(w.replace(".", "").isdigit() for w in words):
            return False, 0
        return True, 1

    # ── digest builder ────────────────────────────────────────────────────────

    def _build_page_digest(self, pages: List[PageText]) -> str:
        """
        Build a rich page-by-page digest for the discovery LLM call.

        Each page block looks like:
            --- Page 5 ---
            Headings: 3. Next-Generation Firewall | 3.1 Performance
            Text: The firewall shall support a minimum throughput of 10 Gbps...
        """
        rows: List[str] = []
        for p in pages:
            headings: List[str] = []
            for raw_line in p.text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                is_heading, depth = self._is_heading_line(line)
                if is_heading:
                    indent = "  " * max(0, depth - 1)
                    headings.append(f"{indent}{line}")

            # Always include raw text snippet so the LLM can infer product
            # context even when the document has no numbered headings
            raw_snippet = " ".join(p.text.split())[:DISCOVERY_RAW_CHARS_PER_PAGE]

            block_parts = [f"--- Page {p.page_number} ---"]
            if headings:
                block_parts.append("Headings: " + " | ".join(headings[:8]))
            if raw_snippet:
                block_parts.append("Text: " + raw_snippet)

            rows.append("\n".join(block_parts))

        full = "\n\n".join(rows)
        if len(full) > DISCOVERY_DIGEST_CHAR_BUDGET:
            full = full[:DISCOVERY_DIGEST_CHAR_BUDGET]
        return full

    # ── LLM discovery call ────────────────────────────────────────────────────

    def _llm_discover_products(
        self, digest: str, total_pages: int
    ) -> ProductManifest:
        """
        Single LLM call to identify all products and their page sets.

        The prompt is explicit, minimal, and asks for a "pages" array.
        Robust JSON extraction handles <think> blocks and code fences.
        """
        prompt = f"""You are an expert RFP analyst. Below is a page-by-page digest of an RFP.
Each page block shows its page number, any section headings, and a text excerpt.

Your task: identify every distinct product, solution, or service that has its own
requirements section in this RFP.

Rules:
- Use the product/solution name derived/concluded from the specifications in the headings or text.
- For each product, list every page number that contains requirements for that product.
- A page may belong to multiple products.
- Exclude: cover page, table of contents, glossary, introduction, and legal/commercial sections
  unless they contain technical requirements.
- Do not merge separate products into one entry.
- Do not split a single product into multiple entries.

Return ONLY a valid JSON array. No markdown, no code fences, no explanation, no preamble.
The array must be valid JSON that can be parsed directly.

Each array element must have exactly these keys:
  "product"  - product or solution name (string)
  "pages"    - sorted array of 1-based page numbers for that product (array of integers)

Example of valid output:
[
  {{"product": "Next-Generation Firewall", "pages": [3, 4, 5, 11]}},
  {{"product": "SIEM Solution", "pages": [6, 7, 8]}}
]

RFP PAGE DIGEST:
{digest}
"""

        response = ""
        try:
            response = llm.generate(prompt, max_tokens=2000)
            logger.debug(f"Discovery raw response (first 1000): {response[:1000]!r}")

            cleaned = self._clean_llm_response(response)

            # Guard: empty response after cleaning
            if not cleaned.strip():
                raise ValueError("LLM returned empty response after cleaning")

            data = self._loads_json(cleaned)

            if not isinstance(data, list):
                raise ValueError(f"Expected JSON array, got {type(data).__name__}")

            by_name: dict[str, ProductEntry] = {}
            for item in data:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("product", "")).strip()
                page_numbers = self._coerce_page_list(item, total_pages)

                # Fallback: accept start_page / end_page if "pages" is missing
                if not page_numbers:
                    start = self._safe_int(item.get("start_page"))
                    end   = self._safe_int(item.get("end_page"))
                    if start and end:
                        start = max(1, min(start, total_pages))
                        end   = max(start, min(end, total_pages))
                        page_numbers = list(range(start, end + 1))

                if self._is_junk_product_name(name) or not page_numbers:
                    logger.debug(f"Rejected junk product name: {name!r}")
                    continue

                key = self._normalise_product_key(name)
                if key in by_name:
                    merged = sorted(set(by_name[key].pages) | set(page_numbers))
                    by_name[key].pages      = merged
                    by_name[key].start_page = min(merged)
                    by_name[key].end_page   = max(merged)
                else:
                    by_name[key] = ProductEntry(
                        product    = name,
                        start_page = min(page_numbers),
                        end_page   = max(page_numbers),
                        pages      = sorted(set(page_numbers)),
                    )

            products = sorted(by_name.values(), key=lambda p: p.start_page)
            return ProductManifest(products=products)

        except Exception as exc:
            logger.warning(f"Product discovery LLM call failed: {exc}")
            if response:
                logger.warning(f"Raw response (first 500 chars): {response[:500]!r}")
            return ProductManifest(products=[])

    # ── heuristic fallback ────────────────────────────────────────────────────

    def _heuristic_discover_products(self, pages: List[PageText]) -> ProductManifest:
        """
        Pure heuristic product discovery when the LLM call fails.

        Finds top-level numbered headings (depth 1) and treats each as a
        product section. Page range is determined by the span until the
        next sibling heading.
        """
        # Collect all (page_number, depth, heading_text) triples
        all_headings: List[Tuple[int, int, str]] = []
        for p in pages:
            for raw_line in p.text.splitlines():
                line = raw_line.strip()
                is_heading, depth = self._is_heading_line(line)
                if is_heading and depth >= 1:
                    all_headings.append((p.page_number, depth, line))

        if not all_headings:
            return ProductManifest(products=[])

        # Determine the minimum depth among found headings — that's the product level
        min_depth = min(d for _, d, _ in all_headings)

        # Filter to top-level headings only
        top_headings = [(pg, txt) for pg, d, txt in all_headings if d == min_depth]

        if not top_headings:
            return ProductManifest(products=[])

        total_pages = max(p.page_number for p in pages)
        products: List[ProductEntry] = []

        for idx, (pg, txt) in enumerate(top_headings):
            next_pg = top_headings[idx + 1][0] - 1 if idx + 1 < len(top_headings) else total_pages
            next_pg = max(pg, next_pg)

            page_set = list(range(pg, next_pg + 1))

            if self._is_junk_product_name(txt):
                continue

            products.append(ProductEntry(
                product    = txt,
                start_page = pg,
                end_page   = next_pg,
                pages      = page_set,
            ))

        return ProductManifest(products=products)

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 2 – PRODUCT-SPECIFIC REQUIREMENT EXTRACTION
    # ──────────────────────────────────────────────────────────────────────────

    def extract_for_product(
        self,
        pages: List[PageText],
        product: ProductEntry,
    ) -> List[Requirement]:
        """
        Extract all requirements for a single product from its page range.

        Steps
        -----
        1. Slice pages to the product's page set.
        2. Run regex pass over sliced text (zero API calls).
        3. Chunk sliced pages, run LLM extraction in parallel.
        4. Merge, deduplicate, assign IDs.
        """
        page_scope     = product.pages or list(range(product.start_page, product.end_page + 1))
        page_scope_set = set(page_scope)
        print(f"\n=== EXTRACTING: {product.product} ===")
        print(f"    Pages: {ProductManifest._format_pages(page_scope)}")

        # Step 1: slice to product pages
        product_pages = [p for p in pages if p.page_number in page_scope_set]
        if not product_pages:
            logger.warning(f"No pages found for {product.product}")
            return []

        total_chars = sum(len(p.text) for p in product_pages)
        print(f"    {len(product_pages)} pages  |  {total_chars:,} chars")

        # Step 2: regex pass (no API call)
        regex_reqs = self._regex_pass(product_pages, product.product)
        print(f"    Regex hits: {len(regex_reqs)}")

        # Step 3: LLM extraction in parallel
        chunks = self._chunk_pages(product_pages)
        workers = min(len(chunks), MAX_WORKERS)
        print(f"    LLM chunks: {len(chunks)}  (max {workers} parallel workers)")
        llm_reqs = self._llm_extract_parallel(chunks, product.product)
        print(f"    LLM requirements (raw): {len(llm_reqs)}")

        # Step 4: merge, dedup, assign IDs
        all_reqs = self._deduplicate(regex_reqs + llm_reqs)
        self._assign_ids(all_reqs)
        print(f"    Final (after dedup): {len(all_reqs)}")
        return all_reqs

    # ──────────────────────────────────────────────────────────────────────────
    # CHUNKING  (page-aware)
    # ──────────────────────────────────────────────────────────────────────────

    def _chunk_pages(
        self, pages: List[PageText]
    ) -> List[Tuple[str, int, int, str]]:
        """
        Pack pages into chunks of ≤ CHUNK_SIZE chars.
        Returns list of (chunk_label, first_page, last_page, text).
        Never splits a page across two chunks.
        """
        chunks: List[Tuple[str, int, int, str]] = []
        current_texts: List[str] = []
        current_len   = 0
        chunk_first   = pages[0].page_number if pages else 1
        chunk_last    = chunk_first
        chunk_idx     = 1

        def flush():
            nonlocal chunk_idx, current_texts, current_len, chunk_first, chunk_last
            if current_texts:
                label = f"Chunk-{chunk_idx} (pp.{chunk_first}–{chunk_last})"
                chunks.append((label, chunk_first, chunk_last, "\n\n".join(current_texts)))
                chunk_idx    += 1
                current_texts = []
                current_len   = 0

        for page in pages:
            page_text = page.text.strip()
            if not page_text:
                continue
            tagged = f"[Page {page.page_number}]\n{page_text}"

            if current_len + len(tagged) > CHUNK_SIZE and current_texts:
                flush()
                chunk_first = page.page_number

            current_texts.append(tagged)
            current_len += len(tagged)
            chunk_last   = page.page_number

        flush()
        return chunks

    # ──────────────────────────────────────────────────────────────────────────
    # REGEX EXTRACTION
    # ──────────────────────────────────────────────────────────────────────────

    def _regex_pass(
        self, pages: List[PageText], product: str
    ) -> List[Requirement]:
        results: List[Requirement] = []
        for page in pages:
            for sentence in self._split_sentences(page.text):
                match = self.QUANT_PATTERN.search(sentence)
                if not match:
                    continue
                results.append(Requirement(
                    requirement_id = "",
                    category       = product,
                    requirement    = match.group("metric").strip(),
                    source_text    = sentence.strip(),
                    mandatory      = self._is_mandatory(sentence),
                    operator       = ">=",
                    value          = match.group("value"),
                    unit           = match.group("unit"),
                    section        = f"Page {page.page_number}",
                ))
        return results

    # ──────────────────────────────────────────────────────────────────────────
    # LLM EXTRACTION  (parallel)
    # ──────────────────────────────────────────────────────────────────────────

    def _llm_extract_parallel(
        self,
        chunks: List[Tuple[str, int, int, str]],
        product: str,
    ) -> List[Requirement]:
        all_reqs: List[Requirement] = []
        if not chunks:
            return all_reqs
        workers = min(len(chunks), MAX_WORKERS)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._llm_extract_chunk, label, fp, lp, text, product): label
                for label, fp, lp, text in chunks
            }
            for future in as_completed(futures):
                label = futures[future]
                try:
                    reqs = future.result()
                    print(f"    {label}: {len(reqs)} requirements")
                    all_reqs.extend(reqs)
                except Exception as exc:
                    logger.warning(f"Chunk {label} failed: {exc}")

        return all_reqs

    def _llm_extract_chunk(
        self,
        chunk_label: str,
        first_page: int,
        last_page: int,
        text: str,
        product: str,
    ) -> List[Requirement]:
        prompt = f"""You are an RFP analyst extracting technical requirements for: {product}

The text below comes from pages {first_page}–{last_page} of an RFP.
Each page is delimited by [Page N].

Extract EVERY requirement stated in the text — both quantitative and qualitative.
Include: performance specs, capacity thresholds, feature support, compliance mandates,
         deployment constraints, integration requirements, operational requirements,
         interface requirements, security requirements, environmental requirements.

Rules:
- Do NOT skip any requirement, even if it seems minor.
- Each requirement must be a standalone, self-contained statement.
- Do NOT invent requirements that are not in the text.
- Preserve exact numeric values and units from the source.

Return ONLY a valid JSON array. No markdown, no code fences, no explanation, no preamble.

Each object must have exactly these keys:
  "requirement"  – concise, self-contained requirement statement (string)
  "category"     – functional sub-area within {product} (e.g. "Performance", "HA",
                   "Logging", "Authentication") — derive from the text
  "mandatory"    – true if text uses shall/must/mandatory/required, else false
  "source_text"  – the exact sentence(s) from the document (preserve original wording)
  "page_number"  – page number where this requirement appears (integer)
  "operator"     – ">=" for numeric thresholds, "supports" for feature requirements
  "value"        – numeric threshold as string (e.g. "10"), or "true" for feature reqs
  "unit"         – unit for numeric specs: Gbps/Mbps/TB/GB/MB/Users/Sessions/EPS — or null

TEXT:
{text}
"""

        response = ""
        try:
            response = llm.generate(prompt, max_tokens=6000)
            cleaned  = self._clean_llm_response(response)

            if not cleaned.strip():
                raise ValueError("LLM returned empty response after cleaning")

            data = self._loads_json(cleaned)

            if not isinstance(data, list):
                raise ValueError(f"Expected list, got {type(data).__name__}")

            results: List[Requirement] = []
            for item in data:
                if not isinstance(item, dict) or not item.get("requirement"):
                    continue
                pg      = self._safe_int(item.get("page_number"))
                section = f"Page {pg}" if pg else chunk_label

                results.append(Requirement(
                    requirement_id = "",
                    category       = str(item.get("category", product)).strip() or product,
                    requirement    = str(item["requirement"]).strip(),
                    source_text    = str(item.get("source_text", "")).strip(),
                    mandatory      = bool(item.get("mandatory", True)),
                    operator       = str(item.get("operator", "supports")),
                    value          = str(item.get("value", "true")),
                    unit           = item.get("unit") or None,
                    section        = section,
                ))
            return results

        except Exception as exc:
            logger.warning(f"LLM extraction failed for {chunk_label}: {exc}")
            if response:
                logger.debug(f"Raw response (first 300): {response[:300]!r}")
            return []

    # ──────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _is_junk_product_name(self, name: str) -> bool:
        """Return True if `name` looks like noise rather than a real product heading."""
        if not name or len(name.strip()) < 3:
            return True
        if any(c in name for c in ("://", "\\", ".com", ".org", ".pdf")):
            return True

        tokens = name.split()
        if len(tokens) == 1:
            t = tokens[0].upper()
            if t in self._UNIT_WORDS or t.replace(".", "").isdigit() or len(t) <= 3:
                return True

        _UNIT_TOKENS = self._UNIT_WORDS | {
            "M", "K", "G", "T", "GHZ", "MHZ", "MBPS", "GBPS",
            "MS", "S", "KB", "MB", "GB", "TB",
        }
        for i, tok in enumerate(tokens[:-1]):
            if tok.replace(".", "").isdigit():
                if tokens[i + 1].upper() in _UNIT_TOKENS:
                    return True

        if not any(c.isalpha() for c in name):
            return True
        if sum(1 for c in name if c.isalpha()) < 2:
            return True

        return False

    def _clean_llm_response(self, response: str) -> str:
        """Strip <think> blocks, code fences, and leading/trailing whitespace."""
        # Remove <think>...</think> blocks (reasoning models)
        response = self._THINK_RE.sub("", response).strip()
        # Remove opening ```json or ``` fences
        response = self._FENCE_OPEN_RE.sub("", response).strip()
        # Remove closing ``` fences
        response = self._FENCE_CLOSE_RE.sub("", response).strip()
        return response

    def _loads_json(self, response: str) -> Any:
        """
        Parse LLM JSON robustly.
        Tries direct parse first, then hunts for the first JSON array or object.
        """
        # Direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Hunt for a JSON array or object in the response
        match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", response)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"No valid JSON found in response. First 200 chars: {response[:200]!r}")

    def _coerce_page_list(self, item: dict, total_pages: int) -> List[int]:
        raw_pages = item.get("pages", [])
        if isinstance(raw_pages, int):
            raw_pages = [raw_pages]
        if not isinstance(raw_pages, list):
            raw_pages = []

        pages: set[int] = set()
        for raw in raw_pages:
            try:
                page = int(raw)
            except (TypeError, ValueError):
                continue
            if 1 <= page <= total_pages:
                pages.add(page)
        return sorted(pages)

    def _normalise_product_key(self, name: str) -> str:
        key = re.sub(r"^\s*\d+(?:\.\d+)*[\s\.\):_-]+", "", name.lower())
        key = re.sub(r"[^a-z0-9]+", " ", key)
        return " ".join(key.split())

    def _is_mandatory(self, text: str) -> bool:
        lower = text.lower()
        if any(w in lower for w in ("shall", "must", "mandatory", "required")):
            return True
        if any(w in lower for w in ("should", "preferred", "optional")):
            return False
        return True

    def _split_sentences(self, text: str) -> List[str]:
        return re.split(r"(?<=[.!?])\s+", text)

    def _deduplicate(self, reqs: List[Requirement]) -> List[Requirement]:
        seen:   set              = set()
        unique: List[Requirement] = []
        for req in reqs:
            # Normalise to catch near-duplicates from regex + LLM overlap
            key = (req.requirement.lower().strip(), req.category.lower().strip())
            if key not in seen:
                seen.add(key)
                unique.append(req)
        return unique

    def _assign_ids(self, reqs: List[Requirement]) -> None:
        for i, req in enumerate(reqs, start=1):
            req.requirement_id = f"REQ-{i:04}"

    @staticmethod
    def _safe_int(val: Any) -> int:
        """Convert val to int safely; return 0 on failure."""
        try:
            return int(val)
        except (TypeError, ValueError):
            return 0
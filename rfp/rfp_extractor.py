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
CHUNK_SIZE        = 3_000   # chars per LLM extraction call
MAX_WORKERS       = 8       # parallel extraction threads
DISCOVERY_PAGE_CHAR_BUDGET = 420
DISCOVERY_DIGEST_CHAR_BUDGET = 12_000
DISCOVERY_PAGE_BATCH_SIZE = 4
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
            page_text = self._format_pages(p.pages) if p.pages else f"{p.start_page}-{p.end_page}"
            lines.append(
                f"  [{i}]  {p.product:<25}  pages {page_text}"
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
            ranges.append(str(start) if start == prev else f"{start}-{prev}")
            start = prev = page
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        return ", ".join(ranges)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class RFPRequirementExtractor:
    """
    Two-phase RFP extractor.

    Phase 1 – Discovery
        extract_pages()      →  page-wise text with preserved page numbers
        discover_products()  →  ProductManifest (1 LLM call)
                                Product names are taken verbatim from the RFP —
                                no external taxonomy is applied.

    Phase 2 – Extraction  (after user selects a product)
        extract_for_product()  →  List[Requirement]
                                  category field = functional sub-topic derived
                                  from the text, not from a fixed category list.
    """

    QUANT_PATTERN = re.compile(
        r"(?P<metric>.+?)"
        r"(?:>=|=>|at least|minimum|min\.?)\s*"
        r"(?P<value>\d+(?:\.\d+)?)\s*"
        r"(?P<unit>Gbps|Mbps|TB|GB|MB|Users|Sessions|EPS)",
        re.IGNORECASE,
    )

    _THINK_RE      = re.compile(r"<think>.*?</think>", re.DOTALL)
    _FENCE_OPEN_RE = re.compile(r"^```(?:json)?\s*",   re.MULTILINE)
    _FENCE_CLOSE_RE = re.compile(r"```\s*$",           re.MULTILINE)

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
        1. Build a page digest with headings, requirement-like lines, and short
           context excerpts.
        2. One LLM call on that outline  →  JSON array of product entries.
        3. Parse, validate, return ProductManifest.
        """
        print("\n=== PRODUCT DISCOVERY ===")
        digest = self._build_page_digest(pages)
        print(f"Digest size: {len(digest)} chars  (from {len(pages)} pages)")

        manifest = self._llm_discover_products(digest, total_pages=len(pages))
        if not manifest.products and len(pages) > DISCOVERY_PAGE_BATCH_SIZE:
            logger.info("Retrying product discovery in page batches")
            manifest = self._discover_products_in_batches(pages)
        print(f"\n{manifest.display()}")
        return manifest

    # ── heading detection ─────────────────────────────────────────────────────
    #
    # A valid heading must be ONE of:
    #   (A) Numbered section:  digits-and-dots prefix, then whitespace, then
    #       at least one letter (rules out "45 M", "3.5 GHz", bare numbers).
    #       The text portion must contain at least one letter to exclude
    #       lines like "1.  " or table row numbers.
    #
    #   (B) ALL-CAPS title:  2+ words, every word ≥ 2 chars, no word is a
    #       pure number or a common unit abbreviation.  Minimum total length
    #       of 6 chars.  This blocks "HTTP", "ID", "N/A", "GB", "45 M".
    #
    # Both forms: max 120 chars, must not look like a URL or file path.
    #
    _NUMBERED_HEADING_RE = re.compile(
        r"^(?P<num>\d+(?:\.\d+)*)"   # section number  e.g. "3.1.2"
        r"[\s\.\):]+"                # separator       e.g. ". " or ") "
        r"(?P<title>[A-Za-z].{2,})"  # title must START with a letter, ≥3 chars
    )

    # Words that disqualify an ALL-CAPS line from being a heading
    _UNIT_WORDS = frozenset({
        "GB", "MB", "TB", "KB", "GHZ", "MHZ", "KHZ", "HZ",
        "GBPS", "MBPS", "KBPS", "BPS", "MS", "SEC", "MIN",
        "HTTP", "HTTPS", "FTP", "SSH", "TCP", "UDP", "IP",
        "N/A", "NA", "TBD", "ID", "NO", "OK", "VS",
    })

    def _is_heading_line(self, line: str) -> tuple[bool, int]:
        """
        Returns (is_heading, depth).
        depth = number of numeric components in the section number (1 for "3.",
        2 for "3.1", 3 for "3.1.2").  ALL-CAPS headings get depth=0.
        """
        if not line or len(line) > 120:
            return False, 0

        # Reject URLs and file paths immediately
        if any(c in line for c in ("://", "\\", ".com", ".org", ".pdf")):
            return False, 0

        # (A) numbered heading
        m = self._NUMBERED_HEADING_RE.match(line)
        if m:
            title = m.group("title").strip()
            title_words = title.split()
            # title must contain at least 2 letters
            if sum(1 for c in title if c.isalpha()) < 2:
                return False, 0
            # reject if the entire title is a single unit/measurement token
            # e.g. "3.5 GHz", "100 Mbps" — title word is a known unit abbreviation
            if len(title_words) == 1 and title_words[0].upper() in (
                self._UNIT_WORDS | {
                    "M", "K", "G", "T", "MHZ", "GHZ", "MBPS", "GBPS",
                    "MS", "KB", "MB", "GB", "TB", "HZ",
                }
            ):
                return False, 0
            depth = len(m.group("num").split("."))
            return True, depth

        # (B) ALL-CAPS heading (no section number)
        words = line.split()
        if len(words) < 2 or len(line) < 6:
            return False, 0
        if not all(w.replace("-", "").replace("/", "").replace("&", "").isupper()
                   for w in words):
            return False, 0
        # Reject if any word is a known unit/abbreviation or a pure number
        upper_words = {w.upper() for w in words}
        if upper_words & self._UNIT_WORDS:
            return False, 0
        if any(w.replace(".", "").isdigit() for w in words):
            return False, 0
        return True, 0

    def _build_page_digest(self, pages: List[PageText]) -> str:
        """
        Build an indented heading outline so the LLM receives structural depth
        information, not just a flat list.

        Output format:
            Page  12 |   3  Next-Generation Firewall
            Page  12 |     3.1  General Requirements
            Page  13 |     3.2  Performance Requirements
            Page  19 |   4  SIEM Solution
            ...

        Indentation = 2 spaces × (depth - 1).  ALL-CAPS headings (depth 0)
        get no indentation.  This lets the LLM visually distinguish parent
        product headings from their child sub-sections without reasoning about
        numbering conventions.
        """
        rows: List[str] = []
        for p in pages:
            headings: List[str] = []
            signals: List[str] = []
            for raw_line in p.text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                is_heading, depth = self._is_heading_line(line)
                if is_heading:
                    indent = "  " * max(0, depth - 1)
                    headings.append(f"{indent}{line}")
                    continue
                if self._looks_like_requirement_signal(line):
                    signals.append(self._compact_line(line, max_chars=135))

            excerpt = self._page_excerpt(p.text, max_chars=180)
            page_rows = [f"Page {p.page_number:>4}"]
            if headings:
                page_rows.append("  Headings: " + " | ".join(headings[:5]))
            if signals:
                page_rows.append("  Signals: " + " | ".join(signals[:5]))
            if excerpt:
                page_rows.append("  Excerpt: " + excerpt)
            rows.append(self._fit_text("\n".join(page_rows), DISCOVERY_PAGE_CHAR_BUDGET))
        return self._fit_text("\n".join(rows), DISCOVERY_DIGEST_CHAR_BUDGET)

    def _looks_like_requirement_signal(self, line: str) -> bool:
        if len(line) < 12 or len(line) > 220:
            return False
        lower = line.lower()
        signal_terms = (
            "shall", "must", "required", "requirement", "support", "provide",
            "capable", "capacity", "throughput", "performance", "integrat",
            "compliance", "certificate", "license", "appliance", "platform",
            "solution", "system", "service", "software", "hardware",
        )
        if any(term in lower for term in signal_terms):
            return True
        if re.search(r"\b\d+(?:\.\d+)?\s*(gbps|mbps|tb|gb|mb|users|sessions|eps|ports?)\b", lower):
            return True
        if re.match(r"^[-*•]|\d+[\.\)]\s+", line):
            return True
        return False

    def _page_excerpt(self, text: str, max_chars: int = 700) -> str:
        cleaned = " ".join(text.split())
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars].rsplit(" ", 1)[0]

    def _compact_line(self, text: str, max_chars: int = 140) -> str:
        return self._fit_text(" ".join(text.split()), max_chars)

    def _fit_text(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        trimmed = text[: max_chars - 3].rsplit(" ", 1)[0]
        return f"{trimmed}..."

    def _llm_discover_products_heading_outline_legacy(
        self, digest: str, total_pages: int
    ) -> ProductManifest:
        """
        Single LLM call on the indented heading outline.

        The prompt is framed as a two-step structural parse:
          Step 1 — identify the shallowest numbered level that contains product
                   names (usually depth-1 headings like "3. SIEM", not depth-2
                   like "3.1 Performance").
          Step 2 — for each product heading, record start/end pages using only
                   the page numbers printed in the outline.

        The LLM is explicitly forbidden from inferring products that do not
        appear as headings in the outline.
        """
        prompt = f"""You are a document structure parser. You will receive an
indented heading outline extracted from an RFP. Each line shows a page number
and a heading. Indentation reflects nesting depth (deeper = child section).

Your task is to identify every product or service that has its own dedicated
requirements section in this RFP, and return the exact page range for each.

Follow these two steps internally, then output only the final JSON:

STEP 1 — Identify the correct heading depth for product sections:
  Look at the outline and find the SHALLOWEST level of numbered headings that
  contain distinct product or service names (e.g. "3. SIEM Solution",
  "4. Next-Generation Firewall").  These are the product headings.
  Deeper headings at that same section number (e.g. "3.1 Performance",
  "3.2 Logging") are sub-sections WITHIN a product — do NOT treat them as
  separate products.

STEP 2 — For each product heading from Step 1:
  - "product"    = the heading text, copied VERBATIM from the outline.
                   Do not paraphrase, shorten, strip numbering, or rename.
  - "start_page" = the page number shown on that heading's line.
  - "end_page"   = the page number shown on the line immediately before the
                   NEXT sibling product heading, or {total_pages} for the last.

Hard rules:
  - Only include headings that are VISIBLE IN THE OUTLINE BELOW.
    Do not infer or hallucinate product names from your general knowledge.
  - Exclude: cover page titles, table-of-contents lines, glossary sections,
    scope/introduction sections, commercial/contractual sections.
  - If a product heading appears in the table of contents AND again in the
    body, use the body occurrence (higher page number).

Return ONLY a JSON array. No markdown, no code fences, no explanation.

Each object must have exactly these keys:
  "product"    – verbatim heading text (string)
  "start_page" – integer, 1-based
  "end_page"   – integer, 1-based

HEADING OUTLINE:
{digest}
"""

        try:
            response = llm.generate(prompt, max_tokens=1200)
            response = self._clean_llm_response(response)
            data = self._loads_json(response)

            if not isinstance(data, list):
                raise ValueError(f"Expected list, got {type(data)}")

            products: List[ProductEntry] = []
            for item in data:
                name  = str(item.get("product", "")).strip()
                start = int(item["start_page"])
                end   = int(item["end_page"])

                # ── post-parse sanity filter ───────────────────────────────
                # Reject entries whose name looks like noise rather than a
                # real product section heading.
                if self._is_junk_product_name(name):
                    logger.debug(f"Rejected junk product name: {name!r}")
                    continue

                # Clamp page numbers to valid range
                start = max(1, min(start, total_pages))
                end   = max(start, min(end, total_pages))

                products.append(ProductEntry(
                    product=name,
                    start_page=start,
                    end_page=end,
                ))

            products.sort(key=lambda p: p.start_page)
            return ProductManifest(products=products)

        except Exception as exc:
            logger.warning(f"Product discovery LLM call failed: {exc}")
            logger.warning(f"Raw response was: {response!r}")
            return ProductManifest(products=[])

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 2 – PRODUCT-SPECIFIC REQUIREMENT EXTRACTION
    # ──────────────────────────────────────────────────────────────────────────

    def _llm_discover_products(
        self, digest: str, total_pages: int
    ) -> ProductManifest:
        """
        Discover demanded products from page-level evidence, not just headings.
        The model may infer product labels from contextual requirements, but it
        must also return the exact pages that justify each product.
        """
        response = ""
        prompt = f"""You are an expert RFP analyst. You will receive a page-wise
digest of an RFP. Each page includes headings and body excerpts that may contain
requirements. The RFP may demand products, platforms, modules, subscriptions,
appliances, software, services, or solution areas without naming them directly.

Your task:
1. Identify ALL distinct demanded products or solution areas in the RFP.
2. Contextually infer a clear product label when the RFP describes capabilities
   without a direct product name.
3. Assign every page that contains requirements, specifications, scope, or
   evaluation criteria for that product to that product.

Hard rules:
- Do not use a fixed product taxonomy or hardcoded product categories.
- Infer labels only from the text in the digest, not from outside knowledge.
- A page may belong to multiple products if it has requirements for multiple
  demanded products.
- Exclude cover page titles, table-of-contents lines, glossary sections,
  instructions to bidders, commercial/contractual/legal sections, and generic
  background unless they contain product requirements.
- Merge synonyms and variants that refer to the same demanded item.
- Keep separate products separate when their requirements could lead to
  different compliance reports.

Return ONLY a JSON array. No markdown, no code fences, no explanation.

Each object must have exactly these keys:
  "product"  - concise product/solution label from the RFP context
  "pages"    - sorted array of unique 1-based page numbers for that product
  "evidence" - short phrase from the digest explaining why this is a product

RFP PAGE DIGEST:
{digest}
"""

        try:
            response = llm.generate(prompt, max_tokens=1200)
            print("\n===== RAW RESPONSE =====")
            print(response[:5000])
            print("========================\n")
            data = self._loads_json(self._clean_llm_response(response))
            if not isinstance(data, list):
                raise ValueError(f"Expected list, got {type(data)}")

            by_name: dict[str, ProductEntry] = {}
            for item in data:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("product", "")).strip()
                page_numbers = self._coerce_page_list(item, total_pages)

                if not page_numbers:
                    start = int(item.get("start_page", 0) or 0)
                    end = int(item.get("end_page", 0) or 0)
                    if start and end:
                        start = max(1, min(start, total_pages))
                        end = max(start, min(end, total_pages))
                        page_numbers = list(range(start, end + 1))

                if self._is_junk_product_name(name) or not page_numbers:
                    logger.debug(f"Rejected junk product name: {name!r}")
                    continue

                key = self._normalise_product_key(name)
                if key in by_name:
                    merged_pages = sorted(set(by_name[key].pages) | set(page_numbers))
                    by_name[key].pages = merged_pages
                    by_name[key].start_page = min(merged_pages)
                    by_name[key].end_page = max(merged_pages)
                    continue

                by_name[key] = ProductEntry(
                    product=name,
                    start_page=min(page_numbers),
                    end_page=max(page_numbers),
                    pages=page_numbers,
                )

            products = sorted(by_name.values(), key=lambda p: p.start_page)
            return ProductManifest(products=products)

        except Exception as exc:
            logger.warning(f"Product discovery LLM call failed: {exc}")
            logger.warning(f"Raw response was: {response!r}")
            return ProductManifest(products=[])

    def _discover_products_in_batches(self, pages: List[PageText]) -> ProductManifest:
        manifests: List[ProductManifest] = []
        for start in range(0, len(pages), DISCOVERY_PAGE_BATCH_SIZE):
            batch = pages[start:start + DISCOVERY_PAGE_BATCH_SIZE]
            digest = self._build_page_digest(batch)
            logger.info(
                "Product discovery batch pages %s-%s digest=%s chars",
                batch[0].page_number,
                batch[-1].page_number,
                len(digest),
            )
            manifest = self._llm_discover_products(digest, total_pages=len(pages))
            manifests.append(manifest)
        return self._merge_product_manifests(manifests)

    def _merge_product_manifests(self, manifests: List[ProductManifest]) -> ProductManifest:
        by_name: dict[str, ProductEntry] = {}
        for manifest in manifests:
            for product in manifest.products:
                pages = product.pages or list(range(product.start_page, product.end_page + 1))
                if not pages:
                    continue
                key = self._normalise_product_key(product.product)
                if key in by_name:
                    merged_pages = sorted(set(by_name[key].pages) | set(pages))
                    by_name[key].pages = merged_pages
                    by_name[key].start_page = min(merged_pages)
                    by_name[key].end_page = max(merged_pages)
                    continue
                by_name[key] = ProductEntry(
                    product=product.product,
                    start_page=min(pages),
                    end_page=max(pages),
                    pages=sorted(set(pages)),
                )

        return ProductManifest(
            products=sorted(by_name.values(), key=lambda p: p.start_page)
        )

    def extract_for_product(
        self,
        pages: List[PageText],
        product: ProductEntry,
    ) -> List[Requirement]:
        """
        Extract all requirements for a single product from its page range.

        Steps
        -----
        1. Slice pages to [start_page, end_page].
        2. Run regex pass over sliced text (zero API calls).
        3. Chunk sliced pages, run LLM extraction in parallel.
        4. Merge, deduplicate, assign IDs.
        """
        page_scope = product.pages or list(range(product.start_page, product.end_page + 1))
        page_scope_set = set(page_scope)
        print(f"\n=== EXTRACTING: {product.product} "
              f"(pages {ProductManifest._format_pages(page_scope)}) ===")

        # Step 1: slice to product pages
        product_pages = [
            p for p in pages
            if p.page_number in page_scope_set
        ]
        if not product_pages:
            logger.warning(f"No pages found for {product.product}")
            return []

        total_chars = sum(len(p.text) for p in product_pages)
        print(f"Pages in scope: {len(product_pages)}  |  Chars: {total_chars}")

        # Step 2: regex pass (no API call)
        regex_reqs = self._regex_pass(product_pages, product.product)
        print(f"Regex requirements: {len(regex_reqs)}")

        # Step 3: LLM extraction in parallel
        chunks = self._chunk_pages(product_pages)
        print(f"LLM chunks: {len(chunks)}  (parallel, max {MAX_WORKERS} workers)")
        llm_reqs = self._llm_extract_parallel(chunks, product.product)
        print(f"LLM requirements (raw): {len(llm_reqs)}")

        # Step 4: merge, dedup, assign IDs
        all_reqs = self._deduplicate(regex_reqs + llm_reqs)
        self._assign_ids(all_reqs)
        print(f"Final requirements (after dedup): {len(all_reqs)}")
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
                label = f"Chunk-{chunk_idx} (pp.{chunk_first}-{chunk_last})"
                chunks.append((label, chunk_first, chunk_last, "\n\n".join(current_texts)))
                chunk_idx += 1
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
                    requirement_id="",
                    category=product,
                    requirement=match.group("metric").strip(),
                    source_text=sentence.strip(),
                    mandatory=self._is_mandatory(sentence),
                    operator=">=",
                    value=match.group("value"),
                    unit=match.group("unit"),
                    section=f"Page {page.page_number}",
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
                    print(f"  {label}: {len(reqs)} requirements")
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
        prompt = f"""You are an RFP analyst extracting requirements for: {product}

The text below comes from pages {first_page}–{last_page} of the RFP.
Each page is marked with [Page N] at the start.

Extract EVERY requirement — quantitative and qualitative.
Include: performance specs, capacity thresholds, feature support, compliance mandates,
         deployment constraints, integration requirements, operational requirements.

Return ONLY a JSON array. No markdown, no code fences, no explanation.

Each object must have exactly these keys:
  "requirement"  – concise, self-contained requirement statement (string)
  "category"     – the sub-topic or functional area within {product} that this
                   requirement belongs to (e.g. "Performance", "High Availability",
                   "Logging", "Authentication") — derive it from the text, do not
                   invent or map to an external taxonomy
  "mandatory"    – true if the text uses shall/must/mandatory/required, else false
  "source_text"  – the exact sentence(s) from the document (preserve original wording)
  "page_number"  – the [Page N] number where this requirement appears (integer)
  "operator"     – ">=" for numeric thresholds, "supports" for feature/capability requirements
  "value"        – numeric threshold as string (e.g. "10"), or "true" for feature requirements
  "unit"         – unit for numeric specs: Gbps/Mbps/TB/GB/MB/Users/Sessions/EPS — or null

TEXT:
{text}
"""

        try:
            response = llm.generate(prompt, max_tokens=6000)
            response = self._clean_llm_response(response)
            data = self._loads_json(response)

            if not isinstance(data, list):
                raise ValueError(f"Expected list, got {type(data)}")

            results: List[Requirement] = []
            for item in data:
                if not isinstance(item, dict) or not item.get("requirement"):
                    continue
                # Determine section label from returned page_number if present
                pg = item.get("page_number")
                section = f"Page {pg}" if pg else chunk_label

                results.append(Requirement(
                    requirement_id="",
                    category=item.get("category", product),
                    requirement=str(item["requirement"]).strip(),
                    source_text=item.get("source_text", ""),
                    mandatory=bool(item.get("mandatory", True)),
                    operator=item.get("operator", "supports"),
                    value=str(item.get("value", "true")),
                    unit=item.get("unit") or None,
                    section=section,
                ))
            return results

        except Exception as exc:
            logger.warning(f"LLM extraction failed for {chunk_label}: {exc}")
            return []

    # ──────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _is_junk_product_name(self, name: str) -> bool:
        """
        Return True if `name` looks like noise rather than a real product section
        heading.  Used as a post-parse guard after the LLM response is parsed.

        Rejects:
          - Empty or very short strings (< 3 chars after stripping)
          - Single-token strings that are all-digits, units, or abbreviations
          - Strings that contain measurement patterns (e.g. "45 M", "3.5 GHz")
          - Strings that are clearly URLs or file paths
        """
        if not name or len(name.strip()) < 3:
            return True

        # URL / path
        if any(c in name for c in ("://", "\\", ".com", ".org", ".pdf")):
            return True

        tokens = name.split()

        # Single token: reject if it's a pure number, known unit, or ≤ 3 chars
        if len(tokens) == 1:
            t = tokens[0].upper()
            if t in self._UNIT_WORDS or t.replace(".", "").isdigit() or len(t) <= 3:
                return True

        # Measurement pattern: a number followed by a unit token
        # e.g. "45 M", "3.5 GHz", "100 Mbps"
        _UNIT_TOKENS = self._UNIT_WORDS | {
            "M", "K", "G", "T", "GHZ", "MHZ", "MBPS", "GBPS",
            "MS", "S", "KB", "MB", "GB", "TB",
        }
        for i, tok in enumerate(tokens[:-1]):
            if tok.replace(".", "").isdigit():
                if tokens[i + 1].upper() in _UNIT_TOKENS:
                    return True

        # Must contain at least one letter (rules out "45", "3.5", "II")
        if not any(c.isalpha() for c in name):
            return True

        # Must have at least 2 letters total
        if sum(1 for c in name if c.isalpha()) < 2:
            return True

        return False

    def _clean_llm_response(self, response: str) -> str:
        response = self._THINK_RE.sub("", response).strip()
        response = self._FENCE_OPEN_RE.sub("", response).strip()
        response = self._FENCE_CLOSE_RE.sub("", response).strip()
        return response

    def _loads_json(self, response: str) -> Any:
        """
        Parse LLM JSON robustly. Some models still wrap valid JSON in short
        prefaces or trailing notes despite strict prompts.
        """
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", response)
            if not match:
                raise
            return json.loads(match.group(1))

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
        seen: set = set()
        unique: List[Requirement] = []
        for req in reqs:
            key = (req.requirement.lower().strip(), req.category)
            if key not in seen:
                seen.add(key)
                unique.append(req)
        return unique

    def _assign_ids(self, reqs: List[Requirement]) -> None:
        for i, req in enumerate(reqs, start=1):
            req.requirement_id = f"REQ-{i:04}"

# rfp/rfp_extractor.py
"""
RFP Requirement Extraction Pipeline  –  Taxonomy-First Edition (fixed)
=======================================================================

Flow
----
  PDF
   └─ extract_pages()        → List[PageText]
   └─ chunk_and_classify()    → ChunkStore   (taxonomy first, LLM only for
                                              chunks the taxonomy can't place)
   └─ store.confirmed_categories()           (filters out one-off / garbage hits)
   └─ extract_for_category()  → List[Requirement]   (regex + LLM, per category)
   └─ run()                   → writes a single JSON file with every confirmed
                                 product/category and its requirements

What changed vs. the previous version
---------------------------------------
1. PRODUCT IDENTIFICATION ("garbage products")
   - Classification now assigns each chunk to its single BEST category
     (no more cloning one chunk into 5+ categories on weak keyword hits).
   - A category is only reported as a real "product" if it has either
       * >= 2 chunks classified into it, OR
       * at least one chunk with a "strong" score (title-word + keyword hit)
     This removes one-off incidental keyword matches (e.g. the word
     "server" appearing once on an unrelated page).

2. LLM CALLS FAILING
   - CHUNK_SIZE / CLASS_CHUNK_SIZE were absurdly small (500 chars), causing
     a flood of tiny LLM calls that got truncated mid-JSON. Restored to
     sane sizes (2500 / 1500 chars).
   - max_tokens raised so responses aren't cut off mid-array.
   - MAX_WORKERS raised from 1 (effectively serial + slow) to a sane default.
   - JSON parsing is now robust:
       * handles dict-wrapped responses (e.g. {"requirements": [...]})
       * salvages complete JSON objects from a TRUNCATED array instead of
         failing the whole chunk
       * retries once (same LLM) if parsing still yields nothing.
   - Category-classification LLM fallback now matches category names that
     appear anywhere in the reply, not just an exact full-string match.

Everything still uses the single `llm` object from services.llm_services —
no second model / provider is introduced anywhere.

What changed in this revision (v3)
-----------------------------------
- Chunking is now paragraph/section-aware (`_split_into_units` /
  `_split_oversized`): chunk boundaries fall on blank lines or detected
  section headings instead of mid-paragraph, and CLASS_CHUNK_SIZE /
  CHUNK_SIZE were raised to 3000 / 5000 chars.
- The extraction prompt now includes a worked few-shot example and
  explicit "testable assertion" criteria for what counts as a real
  requirement vs. boilerplate.
- New `_is_valid_requirement()` filters garbage (too-short fragments,
  boilerplate, malformed numeric values) from BOTH the regex and LLM
  passes, and as a final pass in `extract_for_category`.
- Requirement embeddings now use bge-m3 via Ollama
  (`services.embedding_service`) — the SAME model/endpoint as the OEM
  datasheet KB used by the compliance scorer.
- `Requirement.to_safe_dict()` replaces bare `.dict()` calls (Pydantic v2).
- Requirement IDs are now category-prefixed (e.g. NGFW-0001, ADC-0001)
  instead of a flat REQ-0001 sequence shared across products.
- Per-category and overall pipeline timing is printed via `time.monotonic()`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import fitz
from config.settings import DEFAULT_CONFIG
from models.schemas import Requirement
from services.embedding_service import ChromaBGEM3EmbeddingFunction
from services.llm_services import llm
from rfp.taxonomy import CATEGORY_TAXONOMY          # ← keyword taxonomy

logger = logging.getLogger(__name__)

# ── tunables ───────────────────────────────────────────────────────────────────
CHUNK_SIZE        = 5000   # chars per chunk for requirement EXTRACTION
CLASS_CHUNK_SIZE  = 3000   # chars per chunk for CLASSIFICATION

# NOTE on MAX_WORKERS: this assumes a single LOCAL, CPU-ONLY LLM (e.g. Ollama /
# llama.cpp on an office workstation with no GPU). Such servers process one
# generation at a time on CPU — throwing many parallel threads at it does NOT
# parallelize compute, it just queues requests and can blow up RAM (each queued
# context held in memory) or cause severe slowdowns from cache thrashing/context
# switching between large prompts.
#
# A small value (2) gives a little overlap to hide tokenization/IO overhead
# between calls without contending for the same CPU cores running inference.
# If your server (e.g. llama.cpp with --parallel N) is explicitly configured
# for N concurrent slots, you can raise this to match N — but do not exceed it.
MAX_WORKERS       = 2

EXTRACT_MAX_TOKENS = 6144   # generous headroom so JSON arrays aren't truncated
                            # (larger CHUNK_SIZE → more requirements per call)
CLASSIFY_MAX_TOKENS = 30

MIN_SCORE         = 3       # minimum taxonomy score to accept a category at all
STRONG_SCORE      = 4       # score that counts as a "confident" single-chunk match
MIN_CHUNKS_FOR_PRODUCT = 2  # categories below STRONG_SCORE need >= this many chunks

# Requirement validation thresholds (see _is_valid_requirement)
MIN_REQUIREMENT_CHARS = 8   # below this, a "requirement" is almost certainly
                            # a stray header/page-number/fragment, not a real
                            # requirement statement.

# ── output / vector store paths ─────────────────────────────────────────────
OUTPUT_JSON_PATH   = DEFAULT_CONFIG.paths.requirements_json
CHROMA_DB_PATH     = DEFAULT_CONFIG.paths.chroma_db_path
CHROMA_COLLECTION  = DEFAULT_CONFIG.paths.chroma_collection
# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PageText:
    """Raw text extracted from a single PDF page."""
    page_number: int          # 1-based
    text: str


@dataclass
class TextChunk:
    """A contiguous block of text, classified to one product category."""
    chunk_id:      str
    text:          str
    pages:         List[int]
    category:      str        # taxonomy key  e.g. "ADC", "NGFW", "UNKNOWN"
    score:         float       # classification confidence
    classified_by: str         # "taxonomy" | "llm_fallback" | "unclassified"


@dataclass
class ChunkStore:
    """All chunks from a document, indexed by category."""
    chunks: List[TextChunk] = field(default_factory=list)

    def categories(self) -> List[str]:
        """Every distinct non-UNKNOWN category that has at least one chunk."""
        return sorted({c.category for c in self.chunks if c.category != "UNKNOWN"})

    def chunks_for(self, category: str) -> List[TextChunk]:
        return [c for c in self.chunks if c.category == category]

    def confirmed_categories(self) -> List[str]:
        """
        Categories that look like real products rather than incidental
        keyword hits:
          - at least one chunk scored >= STRONG_SCORE, OR
          - at least MIN_CHUNKS_FOR_PRODUCT chunks classified into it
        """
        confirmed: List[str] = []
        for cat in self.categories():
            chunks = self.chunks_for(cat)
            max_score = max((c.score for c in chunks), default=0.0)
            if max_score >= STRONG_SCORE or len(chunks) >= MIN_CHUNKS_FOR_PRODUCT:
                confirmed.append(cat)
        return confirmed

    def page_range_for(self, category: str) -> Tuple[int, int]:
        pages = [p for c in self.chunks_for(category) for p in c.pages]
        if not pages:
            return (0, 0)
        return (min(pages), max(pages))

    def summary(self) -> str:
        from collections import Counter
        counts    = Counter(c.category for c in self.chunks)
        confirmed = set(self.confirmed_categories())
        lines = ["Chunk classification summary:", ""]
        for cat, n in sorted(counts.items()):
            mark = "OK" if cat in confirmed else ("--" if cat != "UNKNOWN" else "x ")
            lines.append(f"  [{mark}] {cat:<40}  {n} chunk(s)")
        lines.append("")
        lines.append(f"Confirmed products: {len(confirmed)}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# TAXONOMY CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

class TaxonomyClassifier:
    """
    Keyword-based classifier.  No network calls.

    Scoring per category:
      +3 per title_word match (word appears in first 200 chars of chunk)
      +1 per keyword match    (word/phrase appears anywhere in chunk)
      -5 per negative match   (strong negative signal)

    classify() returns the SINGLE best category for a chunk (or "UNKNOWN"
    if nothing reaches MIN_SCORE).
    """

    def __init__(self, taxonomy: Dict[str, Any] = CATEGORY_TAXONOMY):
        self._taxonomy = taxonomy
        self._kw_patterns:    Dict[str, List[re.Pattern]] = {}
        self._title_patterns: Dict[str, List[re.Pattern]] = {}
        self._neg_patterns:   Dict[str, List[re.Pattern]] = {}

        for cat, spec in taxonomy.items():
            self._kw_patterns[cat]    = [re.compile(re.escape(k), re.IGNORECASE)
                                          for k in spec.get("keywords", [])]
            self._title_patterns[cat] = [re.compile(re.escape(k), re.IGNORECASE)
                                          for k in spec.get("title_words", [])]
            self._neg_patterns[cat]   = [re.compile(re.escape(k), re.IGNORECASE)
                                          for k in spec.get("negative", [])]

    def classify(self, text: str) -> Tuple[str, float]:
        """Return (best_category, score). score < MIN_SCORE → ('UNKNOWN', score)."""
        title_region = text[:200]
        best_cat, best_score = "UNKNOWN", 0.0

        for cat in self._taxonomy:
            score = 0.0

            for pat in self._neg_patterns[cat]:
                if pat.search(text):
                    score -= 5

            for pat in self._title_patterns[cat]:
                if pat.search(title_region):
                    score += 3

            for pat in self._kw_patterns[cat]:
                if pat.search(text):
                    score += 1

            if score > best_score:
                best_score = score
                best_cat   = cat

        if best_score >= MIN_SCORE:
            return best_cat, best_score
        return "UNKNOWN", best_score


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ══════════════════════════════════════════════════════════════════════════════

class RFPRequirementExtractor:
    """
    Two-phase RFP extractor.

    Phase 1 – Classification (taxonomy-first, LLM only for ambiguous chunks)
        extract_pages()          →  page-wise text
        chunk_and_classify()     →  ChunkStore

    Phase 2 – Extraction  (after user selects a category)
        extract_for_category()   →  List[Requirement]

    Convenience
        run()                    →  writes a JSON file with all confirmed
                                     products and their requirements.
    """

    # Three accepted shapes, tried in order:
    #   1) "<metric> >= 10 Gbps"            (operator/value/unit AFTER metric)
    #   2) "Minimum 10 Gbps <metric>"       (operator/value/unit BEFORE metric)
    #   3) "<metric> 10 Gbps minimum"       (value/unit then trailing operator,
    #                                        no leading operator at all)
    QUANT_PATTERN_POST = re.compile(
        r"(?P<metric>.+?)"
        r"(?:>=|=>|at least|minimum|min\.?)\s*"
        r"(?P<value>\d+(?:\.\d+)?)\s*"
        r"(?P<unit>Gbps|Mbps|TB|GB|MB|Users|Sessions|EPS)",
        re.IGNORECASE,
    )
    QUANT_PATTERN_PRE = re.compile(
        r"(?:>=|=>|at least|minimum(?: of)?|min\.?)\s*"
        r"(?P<value>\d+(?:\.\d+)?)\s*"
        r"(?:[A-Za-z]+\s+)?"  # optional adjective, e.g. "500 concurrent Sessions"
        r"(?P<unit>Gbps|Mbps|TB|GB|MB|Users|Sessions|EPS)"
        r"(?:\s+(?:of\s+)?(?P<metric>(?!required|mandatory|minimum|is\s|are\s)[A-Za-z][\w\s/-]*?))?"
        r"(?=[.,;:!?]|$|\s+(?:required|mandatory|minimum|is\s|are\s))",
        re.IGNORECASE,
    )
    QUANT_PATTERN_TRAIL = re.compile(
        r"(?P<metric>.+?)"
        r"(?P<value>\d+(?:\.\d+)?)\s*"
        r"(?P<unit>Gbps|Mbps|TB|GB|MB|Users|Sessions|EPS)\s*"
        r"(?:required|minimum|min\.?|or (?:more|greater|higher))\b",
        re.IGNORECASE,
    )

    _THINK_RE       = re.compile(r"<think>.*?</think>", re.DOTALL)
    _FENCE_OPEN_RE  = re.compile(r"^```(?:json)?\s*",   re.MULTILINE)
    _FENCE_CLOSE_RE = re.compile(r"```\s*$",            re.MULTILINE)

    # Generic boilerplate that LLMs occasionally surface as "requirements"
    # but which is not a testable product requirement.
    _BOILERPLATE_RE = re.compile(
        r"\b(?:comply with (?:all|the)?\s*terms|see appendix|"
        r"table of contents|vendor shall comply|terms and conditions of "
        r"this rfp|page \d+ of \d+|this page (?:is|was) intentionally "
        r"(?:left )?blank)\b",
        re.IGNORECASE,
    )

    # A line that LOOKS like a section/sub-section heading:
    #   "3.2 Firewall Requirements", "SECTION 4", "ANNEX A", "TABLE 2",
    #   or a short ALL-CAPS heading line ("NETWORK SECURITY REQUIREMENTS").
    _SECTION_HEADING_RE = re.compile(
        r"^\s*(?:"
        r"\d+(?:\.\d+){0,4}\.?\s+\S"                              # 3.2 Foo
        r"|(?:SECTION|CHAPTER|ANNEX|APPENDIX|PART|TABLE|FIGURE)\s+[\dIVXA-Z]"
        r"|[A-Z][A-Z0-9 /&,\-]{4,}\s*$"                            # ALL CAPS HEADING
        r")",
    )

    def __init__(self):
        self._classifier = TaxonomyClassifier()

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 1A – PAGE EXTRACTION
    # ──────────────────────────────────────────────────────────────────────────

    def extract_pages(self, pdf_path: str) -> List[PageText]:
        """Extract text page-by-page. Returns one PageText per PDF page (1-based)."""
        print(f"Opening PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        pages: List[PageText] = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            pages.append(PageText(page_number=i, text=text))
        doc.close()
        total_chars = sum(len(p.text) for p in pages)
        print(f"Pages extracted: {len(pages)}  |  Total chars: {total_chars:,}")
        return pages

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 1B – CHUNK & CLASSIFY  (taxonomy-first)
    # ──────────────────────────────────────────────────────────────────────────

    def chunk_and_classify(self, pages: List[PageText]) -> ChunkStore:
        """
        Split the document into fixed-size text chunks, then classify each
        chunk to its single best-matching category using the keyword
        taxonomy. Chunks the taxonomy can't place get ONE LLM call (using
        the existing `llm`) to assign a best-guess category.
        """
        print("\n=== CHUNK & CLASSIFY ===")
        raw_chunks = self._build_raw_chunks(pages, CLASS_CHUNK_SIZE)
        print(f"Raw chunks: {len(raw_chunks)}")

        store: ChunkStore = ChunkStore()
        llm_needed: List[Tuple[int, TextChunk]] = []   # (index in store.chunks, chunk)

        for idx, (pages_in_chunk, text) in enumerate(raw_chunks):
            category, score = self._classifier.classify(text)
            chunk = TextChunk(
                chunk_id      = f"chunk-{idx:04d}",
                text          = text,
                pages         = pages_in_chunk,
                category      = category,
                score         = score,
                classified_by = "taxonomy" if category != "UNKNOWN" else "pending",
            )
            store.chunks.append(chunk)
            if category == "UNKNOWN":
                llm_needed.append((len(store.chunks) - 1, chunk))

        taxonomy_hits = len(raw_chunks) - len(llm_needed)
        print(f"Taxonomy classified: {taxonomy_hits}  |  LLM fallback needed: {len(llm_needed)}")

        # ── LLM fallback (only for chunks the taxonomy couldn't place) ─────────
        if llm_needed:
            results = self._llm_classify_batch(llm_needed)
            for store_idx, category in results.items():
                chunk = store.chunks[store_idx]
                if category == "UNKNOWN":
                    chunk.classified_by = "unclassified"
                else:
                    chunk.category      = category
                    # Treat an LLM-assigned category like a MIN_SCORE taxonomy
                    # hit: it needs corroboration (>= MIN_CHUNKS_FOR_PRODUCT)
                    # to be "confirmed" as a product, unless other chunks in
                    # the same category already scored strongly.
                    chunk.score          = float(MIN_SCORE)
                    chunk.classified_by  = "llm_fallback"

        unresolved = sum(1 for c in store.chunks if c.category == "UNKNOWN")
        print(f"Unresolved after LLM: {unresolved}")
        print(f"\n{store.summary()}")
        return store

    def _build_raw_chunks(
        self, pages: List[PageText], chunk_size: int
    ) -> List[Tuple[List[int], str]]:
        """
        Pack pages into text chunks of <= chunk_size chars.

        Unlike a naive "one page = one unit" pack, this splits each page's
        text into paragraph/section units first (see
        `_split_into_units`), so a chunk boundary preferentially falls on
        a blank line or a heading line rather than mid-paragraph. Multiple
        small sections/pages are still packed together up to `chunk_size`;
        a single oversized section is hard-split on sentence boundaries
        (see `_split_oversized`) so no chunk ever exceeds `chunk_size`.

        Returns list of (page_numbers, text).
        """
        chunks:    List[Tuple[List[int], str]] = []
        buf_texts: List[str] = []
        buf_pages: List[int] = []
        buf_len   = 0

        def flush():
            if buf_texts:
                # de-dupe while preserving order
                seen: set = set()
                pages_out: List[int] = []
                for p in buf_pages:
                    if p not in seen:
                        seen.add(p)
                        pages_out.append(p)
                chunks.append((pages_out, "\n\n".join(buf_texts)))
                buf_texts.clear()
                buf_pages.clear()

        for page in pages:
            units = self._split_into_units(page.text)
            if not units:
                continue

            for i, unit in enumerate(units):
                piece_text = f"[Page {page.page_number}]\n{unit}" if i == 0 else unit

                for piece in self._split_oversized(piece_text, chunk_size):
                    if buf_len + len(piece) > chunk_size and buf_texts:
                        flush()
                        buf_len = 0
                    buf_texts.append(piece)
                    buf_pages.append(page.page_number)
                    buf_len += len(piece)

        flush()
        return chunks

    def _split_into_units(self, text: str) -> List[str]:
        """
        Split a page's text into paragraph / section units.

        A new unit starts at:
          - a blank line (standard paragraph break), or
          - a line that LOOKS like a section/sub-section heading
            (`_SECTION_HEADING_RE`), even if not preceded by a blank line.

        This keeps related sentences together while giving the packer in
        `_build_raw_chunks` natural places to break a chunk.
        """
        text = text.strip()
        if not text:
            return []

        lines = text.split("\n")
        units: List[str] = []
        current: List[str] = []

        for line in lines:
            stripped = line.strip()

            if not stripped:
                if current:
                    units.append("\n".join(current).strip())
                    current = []
                continue

            if current and self._SECTION_HEADING_RE.match(line):
                units.append("\n".join(current).strip())
                current = [line]
                continue

            current.append(line)

        if current:
            units.append("\n".join(current).strip())

        return [u for u in units if u]

    def _split_oversized(self, text: str, chunk_size: int) -> List[str]:
        """
        If `text` already fits within `chunk_size`, return it unchanged.
        Otherwise split it on sentence boundaries into pieces that each
        fit, falling back to a hard character split for any single
        sentence that is itself longer than `chunk_size`.
        """
        if len(text) <= chunk_size:
            return [text]

        pieces: List[str] = []
        current = ""
        for sentence in self._split_sentences(text):
            if current and len(current) + 1 + len(sentence) > chunk_size:
                pieces.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip() if current else sentence
        if current:
            pieces.append(current)

        final: List[str] = []
        for piece in pieces:
            if len(piece) <= chunk_size:
                final.append(piece)
            else:
                for j in range(0, len(piece), chunk_size):
                    final.append(piece[j:j + chunk_size])
        return final or [text[:chunk_size]]


    # ── LLM fallback classifier ────────────────────────────────────────────────

    def _llm_classify_batch(
        self,
        llm_needed: List[Tuple[int, TextChunk]],
    ) -> Dict[int, str]:
        """
        For chunks the taxonomy could not classify, call the (single, existing)
        LLM once per chunk in parallel to assign a best-guess category from
        the known taxonomy keys.
        """
        valid_cats = sorted(CATEGORY_TAXONOMY.keys())
        valid_set  = set(valid_cats)
        workers    = max(1, min(len(llm_needed), MAX_WORKERS))
        results: Dict[int, str] = {}

        def _classify_one(idx_chunk: Tuple[int, TextChunk]) -> Tuple[int, str]:
            idx, chunk = idx_chunk
            prompt = f"""You are classifying a chunk of text from a datacenter RFP / OEM datasheet.
Pick the SINGLE best-matching category from this list:

{chr(10).join(f"  - {c}" for c in valid_cats)}

If nothing fits well, reply UNKNOWN.

Reply with ONLY one category key from the list above (e.g. ADC). No sentence,
no punctuation, no explanation.

TEXT:
{chunk.text[:1200]}
"""
            try:
                resp = llm.generate(prompt, max_tokens=CLASSIFY_MAX_TOKENS)
                resp = self._clean_llm_response(resp).strip().upper()

                # Exact match first
                if resp in valid_set:
                    return (idx, resp)

                # Otherwise: did a valid category name appear anywhere in
                # the reply (model added punctuation/explanation)?
                for cat in valid_cats:
                    if re.search(rf"\b{re.escape(cat)}\b", resp):
                        return (idx, cat)
            except Exception as exc:
                logger.warning(f"LLM classify failed for {chunk.chunk_id}: {exc}")
            return (idx, "UNKNOWN")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_classify_one, item): item for item in llm_needed}
            for future in as_completed(futures):
                idx, cat = future.result()
                results[idx] = cat

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 2 – REQUIREMENT EXTRACTION  (category-scoped)
    # ──────────────────────────────────────────────────────────────────────────

    def extract_for_category(
        self,
        store: ChunkStore,
        category: str,
    ) -> List[Requirement]:
        """
        Extract all requirements for a given product category.

        1. Retrieve chunks labelled with `category` from the store.
        2. Run regex pass (zero API calls).
        3. Run parallel LLM extraction over the selected chunks.
        4. Merge, deduplicate, validate, assign category-prefixed IDs.
        """
        matched = store.chunks_for(category)
        if not matched:
            logger.warning(f"No chunks found for category: {category}")
            return []

        start_time = time.monotonic()
        print(f"\n=== EXTRACTING: {category} ===")
        print(f"    Chunks: {len(matched)}")

        ext_chunks = self._repack_for_extraction(matched)
        print(f"    Extraction chunks: {len(ext_chunks)}")

        regex_reqs = self._regex_pass_chunks(matched, category)
        print(f"    Regex hits: {len(regex_reqs)}")

        llm_reqs = self._llm_extract_parallel(ext_chunks, category)
        print(f"    LLM requirements (raw): {len(llm_reqs)}")

        all_reqs = self._deduplicate(regex_reqs + llm_reqs)

        before_validation = len(all_reqs)
        all_reqs = [r for r in all_reqs if self._is_valid_requirement(r)]
        dropped = before_validation - len(all_reqs)
        if dropped:
            print(f"    Dropped as invalid/boilerplate: {dropped}")

        self._assign_ids(all_reqs, category)

        elapsed = time.monotonic() - start_time
        print(f"    Final (after dedup + validation): {len(all_reqs)}")
        print(f"    Time for {category}: {elapsed:.1f}s")
        return all_reqs

    def _repack_for_extraction(
        self, chunks: List[TextChunk]
    ) -> List[Tuple[str, int, int, str]]:
        """
        Re-pack TextChunks into larger chunks (<= CHUNK_SIZE) for LLM extraction.
        Returns list of (label, first_page, last_page, text).
        """
        result:    List[Tuple[str, int, int, str]] = []
        buf_texts: List[str] = []
        buf_pages: List[int] = []
        buf_len   = 0
        idx       = 1

        def flush():
            nonlocal idx
            if buf_texts:
                fp, lp = min(buf_pages), max(buf_pages)
                result.append((f"Chunk-{idx} (pp.{fp}-{lp})", fp, lp, "\n\n".join(buf_texts)))
                idx += 1
                buf_texts.clear()
                buf_pages.clear()

        for chunk in chunks:
            if buf_len + len(chunk.text) > CHUNK_SIZE and buf_texts:
                flush()
                buf_len = 0
            buf_texts.append(chunk.text)
            buf_pages.extend(chunk.pages)
            buf_len += len(chunk.text)

        flush()
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # REGEX EXTRACTION
    # ──────────────────────────────────────────────────────────────────────────

    def _match_quant(self, sentence: str):
        """
        Try all three QUANT_PATTERN variants against `sentence`.
        Returns (metric, value, unit) or None. `metric` may be "" if the
        pattern matched but no descriptive text was found.
        """
        for pattern in (self.QUANT_PATTERN_POST, self.QUANT_PATTERN_PRE, self.QUANT_PATTERN_TRAIL):
            m = pattern.search(sentence)
            if m:
                metric = (m.group("metric") or "").strip(" :-")
                return metric, m.group("value"), m.group("unit")
        return None

    def _regex_pass_chunks(
        self, chunks: List[TextChunk], product: str
    ) -> List[Requirement]:
        results: List[Requirement] = []
        for chunk in chunks:
            for sentence in self._split_sentences(chunk.text):
                hit = self._match_quant(sentence)
                if not hit:
                    continue
                metric, value, unit = hit
                if not metric:
                    # No descriptive text around the number (e.g. "Minimum
                    # 500 Sessions required.") — synthesize a minimal
                    # metric from the unit so the requirement is still
                    # usable rather than discarded.
                    metric = f"{unit} capacity"
                first_page = chunk.pages[0] if chunk.pages else 0
                req = Requirement(
                    requirement_id = "",
                    category       = product,
                    requirement    = metric,
                    source_text    = sentence.strip(),
                    mandatory      = self._is_mandatory(sentence),
                    operator       = ">=",
                    value          = value,
                    unit           = unit,
                    section        = f"Page {first_page}",
                )
                if self._is_valid_requirement(req):
                    results.append(req)
        return results


    # ──────────────────────────────────────────────────────────────────────────
    # LLM EXTRACTION  (parallel, with robust JSON handling + retry)
    # ──────────────────────────────────────────────────────────────────────────

    def _llm_extract_parallel(
        self,
        chunks: List[Tuple[str, int, int, str]],
        product: str,
    ) -> List[Requirement]:
        all_reqs: List[Requirement] = []
        if not chunks:
            return all_reqs
        workers = max(1, min(len(chunks), MAX_WORKERS))

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

    def _build_extraction_prompt(self, text: str, product: str, first_page: int, last_page: int) -> str:
        return f"""You are an RFP analyst extracting technical requirements for: {product}

The text below comes from pages {first_page}-{last_page} of an RFP / datasheet.
Each page is delimited by [Page N].

GOAL
Extract EVERY requirement stated in the text as a TESTABLE ASSERTION — a
statement that a reviewer could check against a vendor datasheet and mark
PASS or FAIL with no further interpretation. Cover both quantitative and
qualitative requirements: performance specs, capacity thresholds, feature
support, compliance mandates, deployment constraints, integration
requirements, operational requirements, interface requirements, security
requirements, environmental requirements.

WHAT MAKES A GOOD "TESTABLE ASSERTION"
- It names ONE specific subject/metric (e.g. "IPS throughput", "SSL VPN
  concurrent users", "support for BGP routing").
- For quantitative requirements, it states the operator, numeric value,
  and unit explicitly (e.g. ">= 10 Gbps").
- For qualitative/feature requirements, it states what must be supported
  or true, phrased so "Yes/No/Partial" is a sufficient verdict
  (e.g. "The appliance shall support active/active high availability.").
- It does NOT bundle multiple unrelated requirements into one sentence —
  split compound requirements ("The system shall support X, Y and Z")
  into separate entries, one per testable claim.
- It does NOT restate generic filler ("the vendor shall comply with this
  RFP", "see Appendix A") — these are NOT testable assertions and must be
  skipped.

RULES
- Do NOT skip any real requirement, even if it seems minor.
- Do NOT invent requirements that are not in the text.
- Preserve exact numeric values and units from the source.
- Return a JSON ARRAY directly (not wrapped in any other object/key).
- Return ONLY valid JSON. No markdown, no code fences, no explanation, no preamble.
- Keep each "source_text" short (one sentence). Do not pad output — be concise so the
  full array fits in the response.
- If the text contains NO real requirements (e.g. it is a cover page, table
  of contents, or boilerplate), return an empty array: []

OUTPUT SCHEMA
Each object must have exactly these keys:
  "requirement"  - concise, self-contained, testable requirement statement (string)
  "category"     - functional sub-area within {product} (e.g. "Performance", "HA",
                   "Logging", "Authentication") - derive from the text
  "mandatory"    - true if text uses shall/must/mandatory/required, else false
  "source_text"  - the exact sentence from the document (preserve original wording)
  "page_number"  - page number where this requirement appears (integer)
  "operator"     - ">=" for numeric thresholds, "supports" for feature requirements
  "value"        - numeric threshold as string (e.g. "10"), or "true" for feature reqs
  "unit"         - unit for numeric specs: Gbps/Mbps/TB/GB/MB/Users/Sessions/EPS - or null

EXAMPLE

Example input text:
  [Page 14]
  3.4 Firewall Performance
  The proposed NGFW shall provide a minimum firewall throughput of 40 Gbps
  and threat prevention throughput of at least 10 Gbps. The appliance must
  support active/active and active/passive high availability modes. SSL VPN
  shall support a minimum of 2000 concurrent users. Vendor shall comply with
  all terms of this RFP.

Example output:
[
  {{
    "requirement": "Firewall throughput >= 40 Gbps",
    "category": "Performance",
    "mandatory": true,
    "source_text": "The proposed NGFW shall provide a minimum firewall throughput of 40 Gbps",
    "page_number": 14,
    "operator": ">=",
    "value": "40",
    "unit": "Gbps"
  }},
  {{
    "requirement": "Threat prevention throughput >= 10 Gbps",
    "category": "Performance",
    "mandatory": true,
    "source_text": "threat prevention throughput of at least 10 Gbps",
    "page_number": 14,
    "operator": ">=",
    "value": "10",
    "unit": "Gbps"
  }},
  {{
    "requirement": "Appliance supports active/active and active/passive high availability",
    "category": "HA",
    "mandatory": true,
    "source_text": "The appliance must support active/active and active/passive high availability modes.",
    "page_number": 14,
    "operator": "supports",
    "value": "true",
    "unit": null
  }},
  {{
    "requirement": "SSL VPN concurrent users >= 2000",
    "category": "VPN",
    "mandatory": true,
    "source_text": "SSL VPN shall support a minimum of 2000 concurrent users.",
    "page_number": 14,
    "operator": ">=",
    "value": "2000",
    "unit": "Users"
  }}
]

Note: "Vendor shall comply with all terms of this RFP" was correctly
SKIPPED — it is generic boilerplate, not a testable product requirement.

Now extract from the following text. Return ONLY the JSON array.

TEXT:
{text}
"""

    def _llm_extract_chunk(
        self,
        chunk_label: str,
        first_page:  int,
        last_page:   int,
        text:        str,
        product:     str,
    ) -> List[Requirement]:
        prompt = self._build_extraction_prompt(text, product, first_page, last_page)

        items: List[dict] = []
        last_response = ""

        # Up to 2 attempts using the SAME llm instance — handles transient
        # empty/truncated/malformed responses without giving up entirely.
        for attempt in range(2):
            try:
                response = llm.generate(prompt, max_tokens=EXTRACT_MAX_TOKENS)
                last_response = response
                items = self._parse_requirements_response(response)
                if items:
                    break
            except Exception as exc:
                logger.warning(f"LLM call failed for {chunk_label} (attempt {attempt + 1}): {exc}")

        if not items:
            if last_response:
                logger.warning(
                    f"LLM extraction yielded no requirements for {chunk_label}. "
                    f"First 200 chars of last response: {last_response[:200]!r}"
                )
            else:
                logger.warning(f"LLM extraction failed for {chunk_label}: no response")
            return []

        results: List[Requirement] = []
        for item in items:
            if not isinstance(item, dict) or not item.get("requirement"):
                continue
            pg      = self._safe_int(item.get("page_number"))
            section = f"Page {pg}" if pg else chunk_label

            req = Requirement(
                requirement_id = "",
                category       = str(item.get("category", product)).strip() or product,
                requirement    = str(item["requirement"]).strip(),
                source_text    = str(item.get("source_text", "")).strip(),
                mandatory      = bool(item.get("mandatory", True)),
                operator       = str(item.get("operator", "supports")),
                value          = str(item.get("value", "true")),
                unit           = item.get("unit") or None,
                section        = section,
            )
            if self._is_valid_requirement(req):
                results.append(req)
        return results

    # ──────────────────────────────────────────────────────────────────────────
    # JSON PARSING  (robust against truncation / wrapping)
    # ──────────────────────────────────────────────────────────────────────────

    def _parse_requirements_response(self, response: str) -> List[dict]:
        """
        Parse the LLM's response into a list of requirement dicts.
        Handles:
          - normal JSON arrays
          - dict-wrapped arrays (e.g. {"requirements": [...]})
          - a single requirement object instead of an array
          - TRUNCATED arrays — salvages every complete {...} object found
        Returns [] if nothing usable could be recovered.
        """
        cleaned = self._clean_llm_response(response)
        if not cleaned.strip():
            return []

        # 1) Direct parse
        try:
            data = json.loads(cleaned)
            items = self._normalize_to_list(data)
            if items:
                return items
        except json.JSONDecodeError:
            pass

        # 2) Find the first [...] or {...} block and try to parse that
        match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", cleaned)
        if match:
            try:
                data = json.loads(match.group(1))
                items = self._normalize_to_list(data)
                if items:
                    return items
            except json.JSONDecodeError:
                pass

        # 3) Salvage: walk the text and pull out every COMPLETE top-level
        #    {...} object, even if the surrounding array/object is truncated
        #    (e.g. response got cut off mid-way through the array).
        objects = self._extract_json_objects(cleaned)
        return objects

    @staticmethod
    def _normalize_to_list(data: Any) -> List[dict]:
        """Coerce a parsed JSON value into a list of dicts."""
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]

        if isinstance(data, dict):
            # Common wrapper keys
            for key in ("requirements", "items", "data", "results", "requirement_list"):
                val = data.get(key)
                if isinstance(val, list):
                    return [d for d in val if isinstance(d, dict)]

            # A single requirement object returned directly
            if "requirement" in data:
                return [data]

            # A dict mapping ids -> requirement objects
            values = list(data.values())
            if values and all(isinstance(v, dict) for v in values):
                return values

        return []

    @staticmethod
    def _extract_json_objects(text: str) -> List[dict]:
        """
        Scan `text` and return every syntactically complete top-level JSON
        object ( {...} ) found, ignoring anything that never closes (i.e.
        the response was cut off mid-object). This lets us salvage all but
        the last, truncated entry of an oversized JSON array.
        """
        objects: List[dict] = []
        depth   = 0
        start   = None
        in_str  = False
        escape  = False

        for i, ch in enumerate(text):
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        candidate = text[start:i + 1]
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict):
                                objects.append(obj)
                        except json.JSONDecodeError:
                            pass
                        start = None

        return objects

    def _clean_llm_response(self, response: str) -> str:
        response = self._THINK_RE.sub("", response).strip()
        response = self._FENCE_OPEN_RE.sub("", response).strip()
        response = self._FENCE_CLOSE_RE.sub("", response).strip()
        return response

    # ──────────────────────────────────────────────────────────────────────────
    # REQUIREMENT VALIDATION  (filter garbage from regex + LLM passes)
    # ──────────────────────────────────────────────────────────────────────────

    def _is_valid_requirement(self, req: Requirement) -> bool:
        """
        Reject entries that aren't real, testable requirements:
          - too short to be meaningful (stray fragments, page numbers,
            lone bullet markers)
          - no alphabetic content (pure numbers/punctuation)
          - generic RFP boilerplate ("comply with all terms of this RFP",
            "see Appendix A", "Table of Contents", ...)
          - numeric requirements where "value" isn't actually numeric
            (a malformed LLM response)
        """
        text = (req.requirement or "").strip()

        if len(text) < MIN_REQUIREMENT_CHARS:
            return False

        if not re.search(r"[A-Za-z]{3,}", text):
            return False

        if self._BOILERPLATE_RE.search(text):
            return False
        if req.source_text and self._BOILERPLATE_RE.search(req.source_text):
            return False

        if req.operator == ">=" and req.unit:
            try:
                float(req.value)
            except (TypeError, ValueError):
                return False

        return True

    # ──────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────────

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
        """
        Two-pass deduplication:

        1) Exact dedup on (requirement text, category) — catches identical
           strings (e.g. two LLM chunks returning the same requirement).

        2) Semantic dedup for regex-vs-LLM overlap: the regex pass and the
           LLM pass can both surface the SAME underlying spec (same numeric
           value/unit pulled from the same source sentence) but phrase the
           `requirement` field differently — e.g. regex: "The firewall shall
           support throughput", LLM: "Firewall throughput >= 40 Gbps". These
           won't collide on exact text. We collapse entries that share the
           same (value, unit, source_text) — preferring the LLM-derived
           version since it's typically the more complete/self-contained
           statement.
        """
        # Pass 1: exact text dedup, preserving order
        seen:   set               = set()
        stage1: List[Requirement] = []
        for req in reqs:
            key = (req.requirement.lower().strip(), req.category.lower().strip())
            if key not in seen:
                seen.add(key)
                stage1.append(req)

        # Pass 2: semantic (value/unit/source_text) dedup, LLM wins
        def norm_value(v: str) -> str:
            try:
                return f"{float(v):g}"
            except (TypeError, ValueError):
                return str(v).strip().lower()

        def norm_source(s: str) -> str:
            return re.sub(r"\s+", " ", (s or "")).strip().lower()

        def sem_key(req: Requirement):
            if not req.unit:
                return None
            return (norm_value(req.value), str(req.unit).lower(), norm_source(req.source_text))

        # We don't have an explicit "origin" field, so detect regex-origin
        # entries by the fact that `requirement` text is a verbatim prefix of
        # `source_text` (how _regex_pass_chunks builds it) AND there exists
        # another entry with the same semantic key that is NOT such a prefix
        # (i.e. came from the LLM).
        def is_regex_shaped(req: Requirement) -> bool:
            return bool(req.requirement) and req.source_text.lower().startswith(req.requirement.lower())

        llm_sem_keys = set()
        for req in stage1:
            k = sem_key(req)
            if k is not None and not is_regex_shaped(req):
                llm_sem_keys.add(k)

        unique: List[Requirement] = []
        for req in stage1:
            k = sem_key(req)
            if k is not None and is_regex_shaped(req) and k in llm_sem_keys:
                # an LLM entry already covers this exact numeric spec; drop
                # the regex-shaped duplicate
                continue
            unique.append(req)

        return unique

    def _assign_ids(self, reqs: List[Requirement], category: str) -> None:
        """
        Assign category-prefixed, sequential IDs, e.g. NGFW-0001, NGFW-0002.

        The prefix is derived from the PRODUCT category passed to
        `extract_for_category` (not `req.category`, which is the
        functional sub-area like "Performance" / "HA"), so all
        requirements for one product share one ID namespace.
        """
        prefix = re.sub(r"[^A-Z0-9]+", "", category.upper()).strip("_") or "REQ"
        for i, req in enumerate(reqs, start=1):
            req.requirement_id = f"{prefix}-{i:04d}"

    @staticmethod
    def _safe_int(val: Any) -> int:
        try:
            return int(val)
        except (TypeError, ValueError):
            return 0

    # ──────────────────────────────────────────────────────────────────────────
    # END-TO-END CONVENIENCE: write everything to one JSON file
    # ──────────────────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────────────
    # VECTOR STORE: embed requirements into the existing Chroma DB
    # ──────────────────────────────────────────────────────────────────────────

    def _embed_requirements_into_chroma(
        self,
        result: dict,
        chroma_path: str = CHROMA_DB_PATH,
        collection_name: str = CHROMA_COLLECTION,
    ) -> int:
        """
        Push every extracted requirement into a persistent Chroma collection
        so they're immediately queryable for compliance matching (e.g.
        "does our proposal satisfy requirement X?").

        Each requirement becomes one Chroma document:
          - id:        f"{source_file}::{category}::{requirement_id}"
          - document:  the requirement text (embedded)
          - metadata:  category, requirement_id, mandatory, operator,
                       value, unit, section, source_file, source_text

        Uses bge-m3 via Ollama (`services.embedding_service`) — the SAME
        embedding model/endpoint used for the OEM datasheet KB, so the
        compliance scorer's vector search compares like with like.

        Returns the number of requirements embedded. If chromadb isn't
        installed, or the Ollama embedding endpoint is unreachable, logs a
        warning and returns 0 without failing the whole pipeline — the
        JSON file is still written.
        """
        try:
            import chromadb
        except ImportError:
            logger.warning(
                "chromadb not installed — skipping vector store embedding. "
                "Install with: pip install chromadb"
            )
            return 0

        all_reqs = []
        for product in result.get("products", []):
            for req in product.get("requirements", []):
                all_reqs.append((product["category"], req))

        if not all_reqs:
            print("\nNo requirements to embed.")
            return 0

        embed_fn = ChromaBGEM3EmbeddingFunction()
        if not embed_fn._service.ping():
            logger.warning(
                f"bge-m3 embedding endpoint at {embed_fn._service.base_url} "
                "is unreachable — skipping vector store embedding. The "
                "JSON file has still been written."
            )
            return 0

        os.makedirs(chroma_path, exist_ok=True)
        client = chromadb.PersistentClient(path=chroma_path)

        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

        source_file = result.get("source_file", "unknown")

        ids:        List[str] = []
        documents:  List[str] = []
        metadatas:  List[dict] = []

        for category, req in all_reqs:
            req_id = req.get("requirement_id", "")
            doc_id = f"{source_file}::{category}::{req_id}"
            ids.append(doc_id)
            documents.append(req.get("requirement", ""))
            metadatas.append({
                "source_file":   str(source_file),
                "category":      str(category),
                "requirement_id": str(req_id),
                "mandatory":     bool(req.get("mandatory", False)),
                "operator":      str(req.get("operator", "")),
                "value":         str(req.get("value", "")),
                "unit":          str(req.get("unit") or ""),
                "section":       str(req.get("section", "")),
                "source_text":   str(req.get("source_text", ""))[:1000],
            })

        # Chroma upsert is idempotent on `ids`, so re-running the pipeline
        # on the same PDF updates rather than duplicates entries.
        BATCH = 100
        for i in range(0, len(ids), BATCH):
            collection.upsert(
                ids       = ids[i:i + BATCH],
                documents = documents[i:i + BATCH],
                metadatas = metadatas[i:i + BATCH],
            )

        print(f"\nEmbedded {len(ids)} requirements into Chroma "
              f"collection '{collection_name}' at '{chroma_path}'.")
        return len(ids)

    # ──────────────────────────────────────────────────────────────────────────
    # END-TO-END CONVENIENCE: write everything to one JSON file + embed
    # ──────────────────────────────────────────────────────────────────────────

    def run(
        self,
        pdf_path: str,
        output_json_path: str = OUTPUT_JSON_PATH,
        chroma_path: str = CHROMA_DB_PATH,
        chroma_collection: str = CHROMA_COLLECTION,
        embed: bool = True,
    ) -> dict:
        """
        Full pipeline: extract pages -> classify -> for every CONFIRMED
        product category, extract requirements -> write one JSON file
        (default: data/requirements.json) -> embed every requirement into
        the Chroma vector store for compliance-matching lookups.

        Returns the same dict that gets written to disk.
        """
        pipeline_start = time.monotonic()
        pages = self.extract_pages(pdf_path)
        store = self.chunk_and_classify(pages)

        categories = store.confirmed_categories()
        print(f"\n=== CONFIRMED PRODUCTS: {len(categories)} ===")
        for cat in categories:
            start, end = store.page_range_for(cat)
            print(f"  - {cat}  (pages {start}-{end})")

        result: dict = {
            "source_file": pdf_path,
            "products": [],
        }

        for cat in categories:
            start, end = store.page_range_for(cat)
            reqs = self.extract_for_category(store, cat)
            result["products"].append({
                "category":          cat,
                "page_range":        {"start": start, "end": end},
                "requirement_count": len(reqs),
                "requirements":      [r.to_safe_dict() for r in reqs],
            })

        out_dir = os.path.dirname(output_json_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)

        print(f"\nWrote {output_json_path}")

        if embed:
            self._embed_requirements_into_chroma(
                result, chroma_path=chroma_path, collection_name=chroma_collection,
            )

        total_elapsed = time.monotonic() - pipeline_start
        print(f"\n=== DONE in {total_elapsed:.1f}s "
              f"({total_elapsed / 60:.1f} min) ===")

        return result
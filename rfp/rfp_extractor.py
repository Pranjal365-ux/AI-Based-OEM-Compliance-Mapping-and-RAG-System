# rfp/rfp_extractor.py
"""
RFP Requirement Extraction Pipeline – Page-Range Edition
=========================================================

Flow
----
  PDF
   └─ extract_pages()                   → List[PageText]
   └─ user selects a page range         → (start_page, end_page)
   └─ extract_requirements_from_range() → List[Requirement]
   └─ run()                             → writes a single JSON file with every
                                           requirement found in that page range

What changed vs. the previous version
---------------------------------------
The previous version classified every chunk of the RFP into a product
category (NGFW, ADC, WAF, ...) using a keyword taxonomy + LLM fallback, then
asked the user to pick ONE category before extracting its requirements.

That whole classification step has been removed. The new flow is:

  1. The RFP is opened and split into pages.
  2. The user picks a page range to scan (e.g. "pages 12-30" — the section
     of the RFP that covers the product they care about).
  3. Every requirement on those pages is extracted directly (regex pass for
     quick numeric thresholds + LLM pass for everything else). No product
     category is assigned to the requirement set as a whole.
  4. The resulting requirements are written to JSON and (optionally) embedded
     into Chroma, ready to be matched against the OEM knowledge base so the
     best-fitting products can be found and a compliance report generated.

Each individual requirement still carries a `category` field, but this is now
just a functional grouping derived by the LLM (e.g. "Performance", "High
Availability", "Security", "Power") — it is NOT a product-taxonomy label and
has nothing to do with the OEM product categories (NGFW/ADC/etc.) that used
to drive the old flow.

Everything still uses the single `llm` object from services.llm_services —
no second model / provider is introduced anywhere.
"""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, List, Tuple

import fitz
from models.schemas import Requirement
from services.llm_services import llm

logger = logging.getLogger(__name__)

# ── tunables ───────────────────────────────────────────────────────────────────
CHUNK_SIZE = 2500   # chars per chunk for requirement EXTRACTION

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
MAX_WORKERS = 1

EXTRACT_MAX_TOKENS = 4096   # generous headroom so JSON arrays aren't truncated

# ── output / vector store paths ─────────────────────────────────────────────
OUTPUT_JSON_PATH  = "data/requirements.json"
CHROMA_DB_PATH    = "data/chroma_db"
CHROMA_COLLECTION = "rfp_requirements"
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"   # small, fast, CPU-friendly
# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PageText:
    """Raw text extracted from a single PDF page."""
    page_number: int          # 1-based
    text: str


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ══════════════════════════════════════════════════════════════════════════════

class RFPRequirementExtractor:
    """
    Page-range RFP extractor.

    extract_pages()                  →  page-wise text for the whole document
    extract_requirements_from_range  →  every requirement found on a chosen
                                         page range (no product classification)

    Convenience
        run()                    →  writes a JSON file with every requirement
                                     found in the selected page range, and
                                     (optionally) embeds them into Chroma.
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

    def __init__(self):
        pass

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE EXTRACTION
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

    def page_count(self, pages: List[PageText]) -> int:
        """Convenience helper so a caller can validate a chosen page range."""
        return len(pages)

    # ──────────────────────────────────────────────────────────────────────────
    # REQUIREMENT EXTRACTION OVER A PAGE RANGE
    # ──────────────────────────────────────────────────────────────────────────

    def extract_requirements_from_range(
        self,
        pages: List[PageText],
        start_page: int,
        end_page: int,
    ) -> List[Requirement]:
        """
        Extract every requirement found on pages [start_page, end_page]
        (inclusive, 1-based). No product/category classification is performed
        — every requirement on the selected pages is extracted.

        1. Select the pages in range.
        2. Run a regex pass for quick numeric thresholds (zero API calls).
        3. Run parallel LLM extraction over the selected pages.
        4. Merge, deduplicate, assign IDs.
        """
        selected = self._select_pages(pages, start_page, end_page)
        if not selected:
            logger.warning(
                f"No pages found in range {start_page}-{end_page} "
                f"(document has {len(pages)} page(s))"
            )
            return []

        print(f"\n=== EXTRACTING REQUIREMENTS: pages {start_page}-{end_page} ===")
        print(f"    Pages selected: {len(selected)}")

        regex_reqs = self._regex_pass_pages(selected)
        print(f"    Regex hits: {len(regex_reqs)}")

        ext_chunks = self._repack_pages_for_extraction(selected)
        print(f"    Extraction chunks: {len(ext_chunks)}")

        llm_reqs = self._llm_extract_parallel(ext_chunks)
        print(f"    LLM requirements (raw): {len(llm_reqs)}")

        all_reqs = self._deduplicate(regex_reqs + llm_reqs)
        self._assign_ids(all_reqs)
        print(f"    Final (after dedup): {len(all_reqs)}")
        return all_reqs

    @staticmethod
    def _select_pages(
        pages: List[PageText], start_page: int, end_page: int
    ) -> List[PageText]:
        lo, hi = min(start_page, end_page), max(start_page, end_page)
        return [p for p in pages if lo <= p.page_number <= hi]

    def _repack_pages_for_extraction(
        self, pages: List[PageText]
    ) -> List[Tuple[str, int, int, str]]:
        """
        Re-pack pages into chunks (<= CHUNK_SIZE) for LLM extraction.
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

        for page in pages:
            tagged = f"[Page {page.page_number}]\n{page.text.strip()}"
            if buf_len + len(tagged) > CHUNK_SIZE and buf_texts:
                flush()
                buf_len = 0
            buf_texts.append(tagged)
            buf_pages.append(page.page_number)
            buf_len += len(tagged)

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

    def _regex_pass_pages(self, pages: List[PageText]) -> List[Requirement]:
        results: List[Requirement] = []
        for page in pages:
            for sentence in self._split_sentences(page.text):
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
                results.append(Requirement(
                    requirement_id = "",
                    category       = "General",
                    requirement    = metric,
                    source_text    = sentence.strip(),
                    mandatory      = self._is_mandatory(sentence),
                    operator       = ">=",
                    value          = value,
                    unit           = unit,
                    section        = f"Page {page.page_number}",
                ))
        return results

    # ──────────────────────────────────────────────────────────────────────────
    # LLM EXTRACTION  (parallel, with robust JSON handling + retry)
    # ──────────────────────────────────────────────────────────────────────────

    def _llm_extract_parallel(
        self,
        chunks: List[Tuple[str, int, int, str]],
    ) -> List[Requirement]:
        all_reqs: List[Requirement] = []
        if not chunks:
            return all_reqs
        workers = max(1, min(len(chunks), MAX_WORKERS))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._llm_extract_chunk, label, fp, lp, text): label
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

    def _build_extraction_prompt(self, text: str, first_page: int, last_page: int) -> str:
        return f"""You are an RFP analyst extracting technical requirements from a Request for Proposal (RFP) document.

The text below comes from pages {first_page}-{last_page} of the RFP.
Each page is delimited by [Page N].

Extract EVERY requirement stated in the text — both quantitative and qualitative.
Include: performance specs, capacity thresholds, feature support, compliance mandates,
         deployment constraints, integration requirements, operational requirements,
         interface requirements, security requirements, environmental requirements.

Rules:
- do not think return json immediately
- Do NOT skip any requirement, even if it seems minor.
- Each requirement must be a standalone, self-contained statement.
- Do NOT invent requirements that are not in the text.
- Preserve exact numeric values and units from the source.
- Return a JSON ARRAY directly (not wrapped in any other object/key).
- Return ONLY valid JSON. No markdown, no code fences, no explanation, no preamble.
- Keep each "source_text" short (one sentence). Do not pad output — be concise so the
  full array fits in the response.

Each object must have exactly these keys:
  "requirement"  - concise, self-contained requirement statement (string)
  "category"     - functional area this requirement belongs to (e.g. "Performance", "High
                   Availability", "Security", "Power", "Environmental", "Compliance",
                   "Management", "Connectivity") - derive from the text
  "mandatory"    - true if text uses shall/must/mandatory/required, else false
  "source_text"  - the exact sentence from the document (preserve original wording)
  "page_number"  - page number where this requirement appears (integer)
  "operator"     - ">=" for numeric thresholds, "supports" for feature requirements
  "value"        - numeric threshold as string (e.g. "10"), or "true" for feature reqs
  "unit"         - unit for numeric specs: Gbps/Mbps/TB/GB/MB/Users/Sessions/EPS - or null

TEXT:
{text}
"""

    def _llm_extract_chunk(
        self,
        chunk_label: str,
        first_page:  int,
        last_page:   int,
        text:        str,
    ) -> List[Requirement]:
        prompt = self._build_extraction_prompt(text, first_page, last_page)

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

            results.append(Requirement(
                requirement_id = "",
                category       = str(item.get("category", "General")).strip() or "General",
                requirement    = str(item["requirement"]).strip(),
                source_text    = str(item.get("source_text", "")).strip(),
                mandatory      = bool(item.get("mandatory", True)),
                operator       = str(item.get("operator", "supports")),
                value          = str(item.get("value", "true")),
                unit           = item.get("unit") or None,
                section        = section,
            ))
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
           `requirement` field differently — e.g. regex: "Gbps capacity",
           LLM: "Firewall throughput >= 40 Gbps". These won't collide on
           exact text. We collapse entries that share the same (value, unit,
           source_text) — preferring the LLM-derived version since it's
           typically the more complete/self-contained statement.
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
        # `source_text` (how _regex_pass_pages builds it) AND there exists
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

    def _assign_ids(self, reqs: List[Requirement]) -> None:
        for i, req in enumerate(reqs, start=1):
            req.requirement_id = f"REQ-{i:04d}"

    @staticmethod
    def _safe_int(val: Any) -> int:
        try:
            return int(val)
        except (TypeError, ValueError):
            return 0

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
          - id:        f"{source_file}::{requirement_id}"
          - document:  the requirement text (embedded)
          - metadata:  category, requirement_id, mandatory, operator,
                       value, unit, section, source_file, source_text

        Uses sentence-transformers locally (CPU) via Chroma's default
        embedding function — no external API calls, safe for an
        air-gapped/offline workstation.

        Returns the number of requirements embedded. If chromadb (or the
        embedding deps) aren't installed, logs a warning and returns 0
        without failing the whole pipeline — the JSON file is still written.
        """
        try:
            import chromadb
            from chromadb.utils import embedding_functions
        except ImportError:
            logger.warning(
                "chromadb (or its deps) not installed — skipping vector "
                "store embedding. Install with: pip install chromadb "
                "sentence-transformers"
            )
            return 0

        all_reqs = result.get("requirements", [])
        if not all_reqs:
            print("\nNo requirements to embed.")
            return 0

        os.makedirs(chroma_path, exist_ok=True)
        client = chromadb.PersistentClient(path=chroma_path)

        # all-MiniLM-L6-v2: ~80MB, runs comfortably on CPU, good default
        # for short requirement-style sentences.
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )

        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

        source_file = result.get("source_file", "unknown")

        ids:       List[str] = []
        documents: List[str] = []
        metadatas: List[dict] = []

        for req in all_reqs:
            req_id = req.get("requirement_id", "")
            doc_id = f"{source_file}::{req_id}"
            ids.append(doc_id)
            documents.append(req.get("requirement", ""))
            metadatas.append({
                "source_file":    str(source_file),
                "category":       str(req.get("category", "")),
                "requirement_id": str(req_id),
                "mandatory":      bool(req.get("mandatory", False)),
                "operator":       str(req.get("operator", "")),
                "value":          str(req.get("value", "")),
                "unit":           str(req.get("unit") or ""),
                "section":        str(req.get("section", "")),
                "source_text":    str(req.get("source_text", ""))[:1000],
            })

        # Chroma upsert is idempotent on `ids`, so re-running the pipeline
        # on the same PDF/page-range updates rather than duplicates entries.
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
        start_page: int,
        end_page: int,
        output_json_path: str = OUTPUT_JSON_PATH,
        chroma_path: str = CHROMA_DB_PATH,
        chroma_collection: str = CHROMA_COLLECTION,
        embed: bool = True,
    ) -> dict:
        """
        Full pipeline: extract pages -> extract every requirement on the
        chosen page range -> write one JSON file (default:
        data/requirements.json) -> embed every requirement into the Chroma
        vector store for compliance-matching lookups.

        Returns the same dict that gets written to disk.
        """
        pages = self.extract_pages(pdf_path)
        reqs = self.extract_requirements_from_range(pages, start_page, end_page)

        result: dict = {
            "source_file": pdf_path,
            "page_range": {"start": start_page, "end": end_page},
            "requirement_count": len(reqs),
            "requirements": [r.dict() for r in reqs],
        }

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

        return result
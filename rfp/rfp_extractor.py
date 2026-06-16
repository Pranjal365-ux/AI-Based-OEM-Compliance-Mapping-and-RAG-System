# rfp/rfp_extractor.py
"""
RFP Requirement Extraction Pipeline – Page-Range Edition
=========================================================

Flow
----
  PDF  ──►  extract_pages()
             └─ returns List[PageText] (one per PDF page, 1-based)

  User picks a page range  ──►  run(pdf_path, start_page, end_page)
    1. regex pass  → quick numeric thresholds  (0 LLM calls)
    2. LLM pass    → all other requirements     (parallel, fast model)
    3. deduplicate + assign IDs
    4. write  data/requirements/<stem>_pp{start}-{end}.json
    5. embed  into Chroma at data/vector_store/
               collection: rfp_requirements_<stem>
               using the SAME locally-hosted bge-m3 model as the KB

The extracted requirements are then ready for the compliance-matching stage:
  vector_store.search_for_requirement(req.requirement)  →  top OEM KB chunks
  llm.generate(compliance_prompt)                       →  compliance report

Runtime
-------
The previous version used qwen3:8b for extraction. That model emits a large
<think>…</think> block before the JSON answer, adding 2-5 min per chunk on a
CPU-only Ollama server. We now call llm.generate_fast() which routes to
cfg.llm.extraction_model (default: qwen2.5:7b) — a non-reasoning instruct
model that returns pure JSON immediately. 5 pages should take ~2-4 minutes
instead of 40.

Embedding
---------
Uses the same EmbeddingService (bge-m3 via Ollama) as the OEM KB so that
requirement vectors and datasheet chunk vectors share the same embedding
space — a prerequisite for meaningful cosine similarity comparison at match
time. All settings are pulled from DEFAULT_CONFIG so there is one place to
change the URL or model.
"""

from __future__ import annotations

import json
import logging
import os
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz                          # PyMuPDF
import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import DEFAULT_CONFIG
from models.schemas import Requirement
from services.embedding_service import EmbeddingService
from services.llm_services import llm

logger = logging.getLogger(__name__)

# ── pull tunables from config (single source of truth) ───────────────────────
_CFG      = DEFAULT_CONFIG
_RFP      = _CFG.rfp
_EMB      = _CFG.embedding
_VS       = _CFG.vector_store

CHUNK_SIZE         = _RFP.chunk_size_chars
MAX_WORKERS        = _RFP.max_workers
EXTRACT_MAX_TOKENS = _RFP.extraction_max_tokens
OUTPUT_DIR         = Path(_RFP.output_dir)
CHROMA_DIR         = Path(_VS.persist_directory)   # same dir as OEM KB
COLLECTION_PREFIX  = _RFP.chroma_collection_prefix
DISTANCE_METRIC    = _RFP.distance_metric
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PageText:
    """Raw text extracted from a single PDF page."""
    page_number: int   # 1-based
    text: str


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ══════════════════════════════════════════════════════════════════════════════

class RFPRequirementExtractor:
    """
    Page-range RFP extractor.

    Primary entry point:
        run(pdf_path, start_page, end_page)
            → extracts requirements, writes JSON, embeds into Chroma
            → returns the result dict

    Lower-level access:
        pages = extract_pages(pdf_path)
        reqs  = extract_requirements_from_range(pages, start, end)
    """

    # ── quantitative requirement patterns (tried in order) ───────────────────
    QUANT_PATTERN_POST = re.compile(
        r"(?P<metric>.+?)"
        r"(?:>=|=>|at least|minimum|min\.?)\s*"
        r"(?P<value>\d+(?:\.\d+)?)\s*"
        r"(?P<unit>Gbps|Mbps|TB|GB|MB|Users|Sessions|EPS|CPS|RPS|M\b)",
        re.IGNORECASE,
    )
    QUANT_PATTERN_PRE = re.compile(
        r"(?:>=|=>|at least|minimum(?: of)?|min\.?)\s*"
        r"(?P<value>\d+(?:\.\d+)?)\s*"
        r"(?:[A-Za-z]+\s+)?"
        r"(?P<unit>Gbps|Mbps|TB|GB|MB|Users|Sessions|EPS|CPS|RPS|M\b)"
        r"(?:\s+(?:of\s+)?(?P<metric>(?!required|mandatory|minimum|is\s|are\s)[A-Za-z][\w\s/-]*?))?"
        r"(?=[.,;:!?]|$|\s+(?:required|mandatory|minimum|is\s|are\s))",
        re.IGNORECASE,
    )
    QUANT_PATTERN_TRAIL = re.compile(
        r"(?P<metric>.+?)"
        r"(?P<value>\d+(?:\.\d+)?)\s*"
        r"(?P<unit>Gbps|Mbps|TB|GB|MB|Users|Sessions|EPS|CPS|RPS|M\b)\s*"
        r"(?:required|minimum|min\.?|or (?:more|greater|higher))\b",
        re.IGNORECASE,
    )

    _THINK_RE       = re.compile(r"<think>.*?</think>", re.DOTALL)
    _FENCE_OPEN_RE  = re.compile(r"^```(?:json)?\s*",   re.MULTILINE)
    _FENCE_CLOSE_RE = re.compile(r"```\s*$",            re.MULTILINE)

    def __init__(self):
        self._embedder: Optional[EmbeddingService] = None
        self._chroma:   Optional[chromadb.PersistentClient] = None

    # ── lazy initialisation ───────────────────────────────────────────────────

    def _get_embedder(self) -> EmbeddingService:
        if self._embedder is None:
            self._embedder = EmbeddingService(
                base_url=_EMB.base_url,
                model=_EMB.model_name,
                timeout=_EMB.timeout_seconds,
            )
        return self._embedder

    def _get_chroma(self) -> chromadb.PersistentClient:
        if self._chroma is None:
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            self._chroma = chromadb.PersistentClient(
                path=str(CHROMA_DIR),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._chroma

    def _get_collection(self, collection_name: str):
        client = self._get_chroma()
        return client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": DISTANCE_METRIC},
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE EXTRACTION
    # ──────────────────────────────────────────────────────────────────────────

    def extract_pages(self, pdf_path: str) -> List[PageText]:
        """Extract text page-by-page from a PDF. Returns one PageText per page (1-based)."""
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
        Extract every requirement on pages [start_page, end_page] (inclusive, 1-based).
        No product/category classification is performed.

        Steps:
          1. Select pages in range
          2. Regex pass for numeric thresholds (zero LLM calls)
          3. Parallel LLM pass (fast model, thinking suppressed)
          4. Merge, deduplicate, assign IDs
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
    def _select_pages(pages: List[PageText], start_page: int, end_page: int) -> List[PageText]:
        lo, hi = min(start_page, end_page), max(start_page, end_page)
        return [p for p in pages if lo <= p.page_number <= hi]

    def _repack_pages_for_extraction(
        self, pages: List[PageText]
    ) -> List[Tuple[str, int, int, str]]:
        """
        Re-pack pages into chunks (<= CHUNK_SIZE chars) for LLM extraction.
        A short tail of the previous page is prepended when a new chunk starts
        mid-document so the model can recognise sentences split across pages.
        Returns list of (label, first_page, last_page, text).
        """
        OVERLAP_CHARS = 250

        result:    List[Tuple[str, int, int, str]] = []
        buf_texts: List[str] = []
        buf_pages: List[int] = []
        buf_len   = 0
        idx       = 1
        prev_page_num: Optional[int] = None
        prev_tail = ""

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
                if prev_tail:
                    context = (
                        f"[Context - end of page {prev_page_num}, continuity only - "
                        f"do not extract requirements from this block]\n...{prev_tail}"
                    )
                    buf_texts.append(context)
                    buf_len += len(context)

            buf_texts.append(tagged)
            buf_pages.append(page.page_number)
            buf_len += len(tagged)
            prev_page_num = page.page_number
            prev_tail = page.text.strip()[-OVERLAP_CHARS:]

        flush()
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # REGEX EXTRACTION
    # ──────────────────────────────────────────────────────────────────────────

    def _match_quant(self, sentence: str):
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
                    metric = f"{unit} capacity"
                results.append(Requirement(
                    requirement_id="",
                    category="General",
                    requirement=metric,
                    source_text=sentence.strip(),
                    mandatory=self._is_mandatory(sentence),
                    operator=">=",
                    value=value,
                    unit=unit,
                    section=f"Page {page.page_number}",
                ))
        return results

    # ──────────────────────────────────────────────────────────────────────────
    # LLM EXTRACTION  (parallel, fast model, robust JSON parsing)
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
        return f"""You are an RFP analyst extracting technical requirements from a Request for Proposal document.

Text is from pages {first_page}-{last_page}. Each page is marked [Page N].
A block marked "[Context - end of page N, continuity only - do not extract requirements from this block]"
is ONLY for continuity context — do not extract any requirement from it.

Extract EVERY requirement from [Page N] sections — both quantitative and qualitative.
Include: performance specs, capacity thresholds, feature support, compliance mandates,
deployment constraints, integration, operational, interface, security, environmental requirements.

Rules:
- Do NOT skip any requirement-bearing sentence, even if it seems minor.
- Each requirement must be self-contained and independently verifiable.
- Do NOT invent requirements not present in the text.
- Preserve exact numeric values and units.
- When a bullet list of related items describes ONE capability (e.g. supported protocols,
  OS list, management interfaces), extract as ONE requirement listing all items.
- Only split into multiple requirements when there are DISTINCT independently-verifiable specs.
- IGNORE document furniture: page headers/footers, column headings like "Specification"/"Qty",
  isolated quantity values like "One (01)", table row numbers. Never turn these into requirements.
- Skip any sentence visibly cut off mid-thought by a page break.
- Return a JSON ARRAY only — no wrapping object, no markdown, no code fences, no preamble.

Each item must have exactly these keys:
  "requirement"  - concise self-contained statement
  "category"     - functional area: Performance / High Availability / Security / Power /
                   Environmental / Compliance / Management / Connectivity / Virtualization /
                   Network / Interface / Capacity / Feature Support / Deployment
  "mandatory"    - true if shall/must/mandatory/required, false if should/preferred/optional
  "source_text"  - exact sentence from document (one sentence only)
  "page_number"  - integer page number
  "operator"     - ">=" for numeric thresholds, "supports" for feature/capability requirements
  "value"        - numeric threshold as string e.g. "10", or "true" for feature reqs
  "unit"         - Gbps/Mbps/TB/GB/MB/Users/Sessions/EPS/CPS/RPS or null

TEXT:
{text}
"""

    def _llm_extract_chunk(
        self,
        chunk_label: str,
        first_page: int,
        last_page: int,
        text: str,
    ) -> List[Requirement]:
        prompt = self._build_extraction_prompt(text, first_page, last_page)

        items: List[dict] = []
        last_response = ""

        for attempt in range(2):
            try:
                # generate_fast() routes to the non-reasoning extraction
                # model (qwen2.5:7b by default) — no thinking overhead
                response = llm.generate_fast(prompt, max_tokens=EXTRACT_MAX_TOKENS)
                last_response = response
                items = self._parse_requirements_response(response)
                if items:
                    break
            except Exception as exc:
                logger.warning(f"LLM call failed for {chunk_label} (attempt {attempt + 1}): {exc}")

        if not items:
            if last_response:
                logger.warning(
                    f"No requirements parsed for {chunk_label}. "
                    f"Response start: {last_response[:300]!r}"
                )
            else:
                logger.warning(f"No response for {chunk_label}")
            return []

        results: List[Requirement] = []
        for item in items:
            if not isinstance(item, dict) or not item.get("requirement"):
                continue
            pg      = self._safe_int(item.get("page_number"))
            section = f"Page {pg}" if pg else chunk_label
            # Safely handle null operator/value/unit from LLM
            operator = item.get("operator") or "supports"
            value    = item.get("value")
            if value is None or str(value).lower() in ("null", "none", ""):
                value = "true"
            unit = item.get("unit") or None
            if unit and str(unit).lower() in ("null", "none", ""):
                unit = None

            results.append(Requirement(
                requirement_id="",
                category=str(item.get("category", "General")).strip() or "General",
                requirement=str(item["requirement"]).strip(),
                source_text=str(item.get("source_text", "")).strip(),
                mandatory=bool(item.get("mandatory", True)),
                operator=str(operator),
                value=str(value),
                unit=unit,
                section=section,
            ))
        return results

    # ──────────────────────────────────────────────────────────────────────────
    # JSON PARSING  (robust against truncation / wrapping)
    # ──────────────────────────────────────────────────────────────────────────

    def _parse_requirements_response(self, response: str) -> List[dict]:
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

        # 2) Find first [...] or {...}
        match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", cleaned)
        if match:
            try:
                data = json.loads(match.group(1))
                items = self._normalize_to_list(data)
                if items:
                    return items
            except json.JSONDecodeError:
                pass

        # 3) Salvage complete {...} objects from truncated arrays
        return self._extract_json_objects(cleaned)

    @staticmethod
    def _normalize_to_list(data: Any) -> List[dict]:
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict):
            for key in ("requirements", "items", "data", "results", "requirement_list"):
                val = data.get(key)
                if isinstance(val, list):
                    return [d for d in val if isinstance(d, dict)]
            if "requirement" in data:
                return [data]
            values = list(data.values())
            if values and all(isinstance(v, dict) for v in values):
                return values
        return []

    @staticmethod
    def _extract_json_objects(text: str) -> List[dict]:
        objects: List[dict] = []
        depth = 0
        start = None
        in_str = False
        escape = False
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
                        try:
                            obj = json.loads(text[start:i + 1])
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
        """Two-pass dedup: exact text, then semantic (value+unit+source) with LLM winning."""
        seen: set = set()
        stage1: List[Requirement] = []
        for req in reqs:
            key = (req.requirement.lower().strip(), req.category.lower().strip())
            if key not in seen:
                seen.add(key)
                stage1.append(req)

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

    @staticmethod
    def _make_collection_name(pdf_path: str, start_page: int, end_page: int) -> str:
        """Stable, filesystem-safe Chroma collection name for this RFP + page range."""
        stem = Path(pdf_path).stem
        # Sanitise: Chroma collection names must be 3-63 chars, alphanumeric + underscores/hyphens
        safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", stem)[:40]
        return f"rfp_{safe}_pp{start_page}_{end_page}"

    # ──────────────────────────────────────────────────────────────────────────
    # PERSISTENCE: JSON + Chroma
    # ──────────────────────────────────────────────────────────────────────────

    def _save_json(
        self,
        reqs: List[Requirement],
        pdf_path: str,
        start_page: int,
        end_page: int,
    ) -> Path:
        """Write requirements to data/requirements/<stem>_pp{start}-{end}.json"""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = Path(pdf_path).stem
        safe_stem = re.sub(r"[^a-zA-Z0-9_\-]", "_", stem)[:60]
        out_path = OUTPUT_DIR / f"{safe_stem}_pp{start_page}-{end_page}.json"

        payload = {
            "source_file":       pdf_path,
            "page_range":        {"start": start_page, "end": end_page},
            "requirement_count": len(reqs),
            "requirements":      [r.dict() for r in reqs],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n✓ Requirements JSON written → {out_path}")
        return out_path

    def _embed_into_chroma(
        self,
        reqs: List[Requirement],
        pdf_path: str,
        start_page: int,
        end_page: int,
    ) -> int:
        """
        Embed all requirements into a persistent Chroma collection using
        the same locally-hosted bge-m3 model as the OEM knowledge base.

        Collection name: rfp_<pdf_stem>_pp{start}_{end}
        Stored in:       data/vector_store/  (same dir as OEM KB)

        Each requirement becomes one document:
          id       = REQ-XXXX
          document = requirement text  (what gets embedded)
          metadata = all other fields for filtered lookup at match time

        Returns number of requirements embedded.
        """
        if not reqs:
            print("\nNo requirements to embed.")
            return 0

        collection_name = self._make_collection_name(pdf_path, start_page, end_page)
        collection = self._get_collection(collection_name)

        # Embed in one batched call (bge-m3 handles up to 512 tokens per text;
        # requirement strings are always well under that limit)
        embedder  = self._get_embedder()
        texts     = [r.requirement for r in reqs]

        print(f"\nEmbedding {len(texts)} requirements via {_EMB.model_name} @ {_EMB.base_url} …")
        embeddings = []
        for start in range(0, len(texts), _EMB.batch_size):
            batch = texts[start:start + _EMB.batch_size]
            embeddings.extend(embedder.embed(batch))

        ids        = [r.requirement_id for r in reqs]
        metadatas  = [
            {
                "requirement_id": r.requirement_id,
                "category":       r.category,
                "mandatory":      r.mandatory,
                "operator":       r.operator or "",
                "value":          str(r.value) if r.value is not None else "",
                "unit":           r.unit or "",
                "section":        r.section or "",
                "source_text":    (r.source_text or "")[:1000],
                "source_file":    pdf_path,
                "page_range":     f"{start_page}-{end_page}",
            }
            for r in reqs
        ]

        # Upsert in batches of 100 (Chroma default limit)
        BATCH = 100
        for i in range(0, len(ids), BATCH):
            collection.upsert(
                ids        = ids[i:i + BATCH],
                embeddings = embeddings[i:i + BATCH],
                documents  = texts[i:i + BATCH],
                metadatas  = metadatas[i:i + BATCH],
            )

        print(f"✓ Embedded into Chroma collection '{collection_name}' @ {CHROMA_DIR}")
        print(f"  Total chunks in collection: {collection.count()}")
        return len(reqs)

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: full pipeline
    # ──────────────────────────────────────────────────────────────────────────

    def run(
        self,
        pdf_path: str,
        start_page: int,
        end_page: int,
        embed: bool = True,
    ) -> Dict[str, Any]:
        """
        Full pipeline:
          1. Extract pages from PDF
          2. Extract every requirement on the chosen page range
          3. Write  data/requirements/<stem>_pp{start}-{end}.json
          4. Embed  into data/vector_store/ using bge-m3 (same as OEM KB)

        Returns the result dict (same content as the JSON file).
        """
        pages = self.extract_pages(pdf_path)
        reqs  = self.extract_requirements_from_range(pages, start_page, end_page)

        json_path = self._save_json(reqs, pdf_path, start_page, end_page)

        if embed:
            self._embed_into_chroma(reqs, pdf_path, start_page, end_page)

        result = {
            "source_file":       pdf_path,
            "page_range":        {"start": start_page, "end": end_page},
            "requirement_count": len(reqs),
            "json_path":         str(json_path),
            "chroma_collection": self._make_collection_name(pdf_path, start_page, end_page),
            "requirements":      [r.dict() for r in reqs],
        }
        return result
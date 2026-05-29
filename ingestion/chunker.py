# =============================================================================
# chunker.py — Semantic chunking with row-level spec decomposition
# =============================================================================
#
# Three chunk types produced per document:
#
#   1. semantic   — prose paragraphs split at ~512 tokens with overlap
#                   These carry narrative / feature descriptions.
#
#   2. spec_row   — ONE chunk per spec-table cell (metric × model)
#                   "SSL Inspection Throughput: 540 Gbps  [FG-7121F]"
#                   This is the fix for the "entire table as one chunk" problem.
#                   Each row-cell is independently retrievable, so a query for
#                   "SSL inspection throughput" hits exactly the right number.
#
#   3. table_full — The entire Markdown table kept as a single chunk.
#                   Kept as a fallback so the LLM can see context when needed.
#
# Deduplication:
#   Pass 1 — exact MD5 hash (O(n))
#   Pass 2 — near-duplicate Jaccard on char-trigrams, sliding window of 50
# =============================================================================

import re
import logging
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_CHARS, DEDUP_THRESHOLD

logger = logging.getLogger("ingestion")

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
    length_function=len,
    is_separator_regex=False,
)

TABLE_BLOCK_RE = re.compile(r"(\[TABLE\].*?\[/TABLE\])", re.DOTALL)
TABLE_OPEN_RE  = re.compile(r"^\[TABLE\]")


# ── Deduplication helpers ─────────────────────────────────────────────────────

def _exact_hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()

def _ngram_set(text: str, n: int = 3) -> set:
    norm = re.sub(r"\s+", " ", text.lower().strip())
    return {norm[i:i + n] for i in range(len(norm) - n + 1)}

def _jaccard(a: str, b: str) -> float:
    sa, sb = _ngram_set(a), _ngram_set(b)
    if not sa and not sb: return 1.0
    if not sa or  not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

def _deduplicate(chunks: list) -> list:
    # Pass 1: exact
    seen   = set()
    pass1  = []
    for c in chunks:
        h = _exact_hash(c["text"])
        if h not in seen:
            seen.add(h)
            pass1.append(c)
    exact_removed = len(chunks) - len(pass1)

    # Pass 2: near-dup with sliding window
    unique       = []
    near_removed = 0
    WINDOW       = 50
    for c in pass1:
        if any(_jaccard(c["text"], p["text"]) >= DEDUP_THRESHOLD
               for p in unique[-WINDOW:]):
            near_removed += 1
        else:
            unique.append(c)

    if exact_removed or near_removed:
        logger.info(f"  [dedup] -{exact_removed} exact, -{near_removed} near-dup")
    return unique


# ── Table parsing ─────────────────────────────────────────────────────────────

def _parse_markdown_table(table_text: str):
    """
    Parse a Markdown table into (headers, data_rows).
    Handles separator rows, leading/trailing pipes, and footnote noise in cell values.
    Returns ([], []) if the table cannot be parsed.
    """
    body = table_text.replace("[TABLE]", "").replace("[/TABLE]", "").strip()
    lines = [l.strip() for l in body.split("\n") if l.strip() and "|" in l]
    if len(lines) < 2:
        return [], []

    def split_row(row: str) -> list:
        s = row.strip().strip("|")
        return [cell.strip() for cell in s.split("|")]

    headers = split_row(lines[0])

    # Detect and skip separator row (--- / :--- / ---:)
    data_start = 1
    if len(lines) > 1:
        sep_cells = split_row(lines[1])
        if all(re.match(r"^[\s\-\:\+]+$", c) for c in sep_cells if c):
            data_start = 2

    data_rows = []
    for line in lines[data_start:]:
        cells = split_row(line)
        # Pad or trim to match header count
        if len(cells) < len(headers):
            cells += [""] * (len(headers) - len(cells))
        else:
            cells = cells[:len(headers)]
        data_rows.append(cells)

    return headers, data_rows


def _clean_metric_name(name: str) -> str:
    """Strip trailing footnote numbers/symbols from a metric name."""
    # Remove trailing digit-only or digit+comma patterns: "SSL Inspection 2" → "SSL Inspection"
    return re.sub(r"[\s,]+[\d,\s]+$", "", name.strip()).strip()


def _clean_cell_value(val: str) -> str:
    """
    Normalise a spec value:
      - Strip footnote superscripts like "540 Gbps 2,5" → "540 Gbps"
      - Collapse whitespace
    """
    val = val.strip()
    # Remove trailing footnote markers: digits, commas, spaces at end
    val = re.sub(r"[\s,]+[\d,]+$", "", val)
    return val.strip()


def _is_blank_value(val: str) -> bool:
    return not val or val in {"-", "—", "N/A", "NA", "N.A.", "n/a", "None", "TBD"}


# ── Segment splitter ──────────────────────────────────────────────────────────

def _split_into_segments(text: str) -> list:
    """Split page text into (segment_text, is_table) pairs."""
    parts  = TABLE_BLOCK_RE.split(text)
    result = []
    for part in parts:
        is_table = bool(TABLE_OPEN_RE.match(part.strip()))
        result.append((part.strip(), is_table))
    return result


# ── Chunk builder helpers ─────────────────────────────────────────────────────

def _make_chunk_id(doc_name: str, page: int, text: str) -> str:
    return hashlib.md5(f"{doc_name}|{page}|{text}".encode()).hexdigest()


def _base_meta(doc_meta: dict, page: int, chunk_idx: int,
                has_table: bool, chunk_type: str) -> dict:
    """Build the standard metadata dict shared by all chunk types."""
    return {
        "chunk_type":     chunk_type,
        "vendor":         doc_meta["vendor"],
        "product_family": doc_meta.get("product_family", ""),
        "model":          doc_meta.get("product_family", ""),
        "category":       doc_meta["category"],
        "doc_name":       doc_meta["doc_name"],
        "page":           page,
        "chunk_idx":      chunk_idx,
        "has_table":      has_table,
    }


# ── Main chunking ─────────────────────────────────────────────────────────────

def chunk_pages(pages: list, doc_meta: dict) -> list:
    """
    Convert cleaned pages into chunks ready for embedding.

    doc_meta must contain:
        vendor, product_family, models (list), category, doc_name

    Returns list of:
    {
        "text":     str,
        "metadata": { chunk_type, vendor, product_family, model (if spec_row),
                      metric, value, category, doc_name, page, chunk_idx,
                      has_table, chunk_id }
    }
    """
    vendor         = doc_meta["vendor"]
    product_family = doc_meta.get("product_family", "")
    category       = doc_meta["category"]
    models         = doc_meta.get("models", [])  # may be empty

    # Context string for spec rows — used as "grounding" text for the LLM
    context_str = (
        f"{vendor} {product_family} — "
        f"{category} technical specification"
    ).strip(" —")

    prose_raw   = []
    table_blocks = []   # list of (table_text, page_num)

    for page in pages:
        page_num = page["page"]
        for segment, is_table in _split_into_segments(page["text"]):
            if not segment:
                continue
            if is_table:
                table_blocks.append((segment, page_num))
            else:
                for split in _splitter.split_text(segment):
                    if len(split.strip()) >= MIN_CHUNK_CHARS:
                        prose_raw.append({"text": split.strip(), "page": page_num})

    # Dedup prose
    unique_prose = _deduplicate(prose_raw)

    final_chunks = []
    idx          = 0

    # ── 1. Semantic (prose) chunks ────────────────────────────────────────────
    for chunk in unique_prose:
        text     = chunk["text"]
        page_num = chunk["page"]
        chunk_id = _make_chunk_id(doc_meta["doc_name"], page_num, text)
        meta     = _base_meta(doc_meta, page_num, idx, False, "semantic")
        meta["chunk_id"] = chunk_id
        final_chunks.append({"text": text, "metadata": meta})
        idx += 1

    # ── 2. Table chunks ───────────────────────────────────────────────────────
    for table_text, page_num in table_blocks:

        # A) Full table fallback chunk
        full_id  = _make_chunk_id(doc_meta["doc_name"], page_num, table_text)
        full_meta = _base_meta(doc_meta, page_num, idx, True, "table_full")
        full_meta["chunk_id"] = full_id
        final_chunks.append({"text": table_text, "metadata": full_meta})
        idx += 1

        # B) Row-level spec chunks — ONE chunk per (metric, model/column) cell
        headers, data_rows = _parse_markdown_table(table_text)
        if not headers or not data_rows:
            continue

        # Determine model names from header columns.
        # Convention: col 0 = metric name, col 1..N = model/value columns.
        col_models = []
        for h in headers[1:]:
            h_clean = h.strip()
            # Try to find matching known model codes in this header cell
            matched = []
            for known_model in models:
                if known_model.lower() in h_clean.lower():
                    matched.append(known_model)
            
            if matched:
                col_models.append(matched)
            else:
                # If PyMuPDF generic header like "Col1", check if row 0 has models
                if re.match(r"^Col\d+$", h_clean) or not h_clean:
                    col_models.append([])
                else:
                    col_models.append([h_clean])

        for row_idx, cells in enumerate(data_rows):
            if not cells:
                continue

            # If headers were generic, try to use the first row as headers
            if row_idx == 0 and all(not m for m in col_models):
                for i, cell in enumerate(cells[1:]):
                    c_clean = cell.strip()
                    matched = []
                    for known_model in models:
                        if known_model.lower() in c_clean.lower():
                            matched.append(known_model)
                    if matched:
                        col_models[i] = matched
                    elif c_clean and not re.match(r"^Col\d+$", c_clean):
                        col_models[i] = [c_clean]
                continue

            metric_raw  = cells[0].strip()
            if not metric_raw or re.match(r"^[\s\-\:\+]+$", metric_raw) or re.match(r"^Col\d+$", metric_raw):
                continue

            metric_name = _clean_metric_name(metric_raw)
            if not metric_name:
                continue

            # One chunk per value column per model
            for col_i, col_model_list in enumerate(col_models):
                if not col_model_list:
                    continue
                    
                val_raw = cells[col_i + 1] if col_i + 1 < len(cells) else ""
                val     = _clean_cell_value(val_raw)

                if _is_blank_value(val):
                    continue

                for col_model in col_model_list:
                    # ── Spec row chunk text ───────────────────────────────────────
                    chunk_text = (
                        f"Vendor: {vendor}\n"
                        f"Product: {product_family}\n"
                        f"Model: {col_model}\n\n"
                        f"Feature: {metric_name}\n"
                        f"Value: {val}\n\n"
                        f"Context: {context_str}"
                    )

                    chunk_id = _make_chunk_id(doc_meta["doc_name"], page_num, chunk_text)

                    meta = _base_meta(doc_meta, page_num, idx, True, "spec_row")
                    meta.update({
                        "model":    col_model,
                        "metric":   metric_name,
                        "value":    val,
                        "chunk_id": chunk_id,
                    })
                    final_chunks.append({"text": chunk_text, "metadata": meta})
                    idx += 1

    # Summary
    type_counts = {}
    for c in final_chunks:
        t = c["metadata"]["chunk_type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    logger.info(
        f"  [chunker] Total: {len(final_chunks)} chunks | "
        + " | ".join(f"{k}: {v}" for k, v in type_counts.items())
    )
    return final_chunks

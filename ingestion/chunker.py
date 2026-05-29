# =============================================================================
# chunker.py - Semantic chunks + model-specific spec_row table chunks
# =============================================================================

import hashlib
import logging
import re

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size, chunk_overlap, **_):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_text(self, text):
            text = text.strip()
            if len(text) <= self.chunk_size:
                return [text] if text else []
            chunks = []
            start = 0
            while start < len(text):
                end = min(len(text), start + self.chunk_size)
                chunks.append(text[start:end].strip())
                if end == len(text):
                    break
                start = max(end - self.chunk_overlap, start + 1)
            return [c for c in chunks if c]

try:
    import fitz
except ImportError:
    fitz = None

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
TABLE_OPEN_RE = re.compile(r"^\[TABLE\]")

MODEL_CODE_RE = re.compile(
    r"(?<![A-Z0-9-])("
    r"FG-\d{3,5}[A-Z]{0,4}(?:-\d{1,2})?(?:-DC)?"
    r"|PA-\d{3,5}[A-Z]{0,2}"
    r"|(?:BIG-IP\s+)?[ir]\d{4,5}"
    r"|[A-Z]{2,6}-\d{3,5}[A-Z]{0,3}(?:-DC)?"
    r")(?![A-Z0-9-])",
    re.IGNORECASE,
)

COMPONENT_CODE_RE = re.compile(
    r"^F(?:IM|PM|AN|AC|SW)-\d{3,5}[A-Z]{0,3}(?:-\d{1,2})?(?:-DC)?$",
    re.IGNORECASE,
)

MODEL_VALUE_BLOCKLIST_RE = re.compile(
    r"^(?:AES|SHA)[-\s]?\d+$|^\d+[GK]$|^\d{1,4}[WV]$|^(?:EN|IEC)\s?\d+",
    re.IGNORECASE,
)

GENERIC_TABLE_HEADERS = {
    "DESCRIPTION", "SKU", "URL", "VALUE", "FEATURE",
    "SPECIFICATION", "SPECIFICATIONS", "MODEL", "MODEL NUMBER",
    "PART NUMBER", "PRODUCT", "PRODUCT FAMILY", "ITEM", "NOTES",
    "ORDER INFORMATION", "ORDERING INFORMATION",
}

SECTION_LABELS = {
    "PERFORMANCE", "CAPACITY", "INTERFACES", "HARDWARE", "SYSTEM",
    "DIMENSIONS", "POWER", "ENVIRONMENT", "COMPLIANCE", "CERTIFICATIONS",
}

BLANK_VALUES = {
    "", "-", "--", "N/A", "NA", "N.A.", "n/a", "None", "TBD",
    "nan", "NaN", "none", "n.a.", "not available",
}


def _exact_hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()


def _ngram_set(text: str, n: int = 3) -> set:
    norm = re.sub(r"\s+", " ", text.lower().strip())
    return {norm[i:i + n] for i in range(len(norm) - n + 1)}


def _jaccard(a: str, b: str) -> float:
    sa, sb = _ngram_set(a), _ngram_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _deduplicate(chunks: list) -> list:
    seen = set()
    pass1 = []
    for c in chunks:
        h = _exact_hash(c["text"])
        if h not in seen:
            seen.add(h)
            pass1.append(c)

    unique = []
    for c in pass1:
        if not any(_jaccard(c["text"], p["text"]) >= DEDUP_THRESHOLD for p in unique[-50:]):
            unique.append(c)

    removed = len(chunks) - len(unique)
    if removed:
        logger.info(f"  [dedup] -{removed} duplicate chunks")
    return unique


def _normalise_label(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9]+", " ", text.upper())).strip()


def _is_generic_header(text: str) -> bool:
    return _normalise_label(text) in GENERIC_TABLE_HEADERS


def _is_section_label(text: str) -> bool:
    return _normalise_label(text) in SECTION_LABELS


def _is_component_code(text: str) -> bool:
    return bool(COMPONENT_CODE_RE.fullmatch(text.strip()))


def _is_placeholder_cell(text: str) -> bool:
    return bool(re.fullmatch(r"Col\d+", text.strip(), flags=re.IGNORECASE))


def _clean_metric_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name.strip())
    return re.sub(r"[\s,]+[\d,\s]+$", "", name).strip()


def _clean_cell_value(val: str) -> str:
    val = re.sub(r"\s+", " ", val.strip())
    return re.sub(r"[\s,]+[\d,]+$", "", val).strip()


def _is_blank_value(val: str) -> bool:
    return not val or val.lower() in {v.lower() for v in BLANK_VALUES}


def _is_likely_spec_value(text: str) -> bool:
    text = text.strip()
    if _is_placeholder_cell(text):
        return False
    if _is_blank_value(text):
        return False
    if re.search(r"\d", text):
        return True
    return _normalise_label(text) in {"YES", "NO", "SUPPORTED", "OPTIONAL", "INCLUDED"}


def _is_likely_metric_label(text: str, known_models: list) -> bool:
    text = _clean_metric_name(text)
    if _is_placeholder_cell(text):
        return False
    if not text or _is_blank_value(text) or _is_generic_header(text) or _is_section_label(text):
        return False
    if _extract_models_from_cell(text, known_models):
        return False
    letters = len(re.findall(r"[A-Za-z]", text))
    digits = len(re.findall(r"\d", text))
    return letters >= 3 and letters >= digits


def _known_model_set(doc_meta: dict) -> list:
    family_ids = {m.upper() for m in doc_meta.get("family_identifiers", [])}
    components = {m.upper() for m in doc_meta.get("components", [])}
    models = []
    for model in doc_meta.get("models", []):
        upper = model.upper()
        if upper in family_ids or upper in components or _is_component_code(model):
            continue
        if model not in models:
            models.append(model)
    return sorted(models, key=len, reverse=True)


def _extract_models_from_cell(cell: str, known_models: list) -> list:
    """Return only proven appliance models from a table cell."""
    if not cell or _is_generic_header(cell) or _is_placeholder_cell(cell):
        return []
    if MODEL_VALUE_BLOCKLIST_RE.match(cell.strip()):
        return []

    allowed = {m.upper() for m in known_models}
    found = []

    for known_model in sorted(known_models, key=len, reverse=True):
        pattern = rf"(?<![A-Z0-9-]){re.escape(known_model)}(?![A-Z0-9-])"
        if re.search(pattern, cell, flags=re.IGNORECASE):
            found.append(known_model)

    for m in MODEL_CODE_RE.finditer(cell):
        code = m.group(1).strip()
        canonical = code.upper() if code.lower().startswith("pa-") else code
        if _is_component_code(canonical) or MODEL_VALUE_BLOCKLIST_RE.match(canonical):
            continue
        if allowed and canonical.upper() not in allowed:
            continue
        if canonical not in found:
            found.append(canonical)

    return found


def _models_in_text_order(text: str, known_models: list) -> list:
    matches = []
    for known_model in known_models:
        pattern = rf"(?<![A-Z0-9-]){re.escape(known_model)}(?![A-Z0-9-])"
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            matches.append((m.start(), m.end(), known_model))

    ordered = []
    seen = set()
    for start, end, model in sorted(matches):
        if (start, end, model.upper()) in seen:
            continue
        seen.add((start, end, model.upper()))
        ordered.append((start, end, model))
    return ordered


def _extract_model_groups_from_text(text: str, known_models: list) -> list:
    """Extract grouped model headers like 'FG-7081F/FG-7081F-DC'."""
    groups = []

    candidates = [text, text.replace("\n", " ")]
    for line in text.splitlines():
        candidates.append(line)
        candidates.extend(line.split("|"))

    for candidate in candidates:
        ordered = _models_in_text_order(candidate, known_models)
        if len(ordered) < 2:
            continue

        current = [ordered[0][2]]
        prev_end = ordered[0][1]
        for start, end, model in ordered[1:]:
            separator = candidate[prev_end:start]
            if "/" in separator or "\\" in separator:
                current.append(model)
            else:
                if len(current) > 1:
                    groups.append(current)
                current = [model]
            prev_end = end

        if len(current) > 1:
            groups.append(current)

    deduped = []
    seen = set()
    for group in groups:
        key = tuple(group)
        if key not in seen:
            seen.add(key)
            deduped.append(group)
    return deduped


def _infer_contextual_column_model_map(headers: list, data_rows: list, known_models: list, context_text: str) -> tuple[dict, int]:
    rows = [headers] + data_rows
    width = max(len(row) for row in rows)

    metric_scores = {}
    for col_idx in range(width):
        score = 0
        for row in data_rows[:12]:
            if col_idx < len(row) and _is_likely_metric_label(row[col_idx], known_models):
                score += 1
        metric_scores[col_idx] = score

    metric_col = max(metric_scores, key=metric_scores.get) if metric_scores else 0
    if metric_scores.get(metric_col, 0) == 0:
        metric_col = 0

    value_cols = []
    for col_idx in range(width):
        if col_idx == metric_col:
            continue
        if any(col_idx < len(row) and not _is_blank_value(row[col_idx]) for row in data_rows):
            value_cols.append(col_idx)

    groups = _extract_model_groups_from_text(context_text, known_models)
    if not groups:
        return {}, metric_col

    col_model_map = {}
    if len(groups) >= len(value_cols):
        for col_idx, group in zip(value_cols, groups):
            col_model_map[col_idx] = group
    elif len(groups) == 1:
        for col_idx in value_cols:
            col_model_map[col_idx] = groups[0]

    return col_model_map, metric_col


def _parse_markdown_table(table_text: str):
    body = table_text.replace("[TABLE]", "").replace("[/TABLE]", "").strip()
    lines = [line.strip() for line in body.split("\n") if line.strip() and "|" in line]
    if len(lines) < 2:
        return [], []

    def split_row(row: str) -> list:
        return [cell.strip() for cell in row.strip().strip("|").split("|")]

    rows = [split_row(line) for line in lines]
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]

    data_start = 1
    if len(rows) > 1:
        sep = [cell for cell in rows[1] if cell]
        if sep and all(re.match(r"^[\s\-:\+]+$", cell) for cell in sep):
            data_start = 2

    return rows[0], rows[data_start:]


def _infer_column_model_map(headers: list, data_rows: list, known_models: list) -> tuple[dict, int, int]:
    rows = [headers] + data_rows
    scan_limit = min(len(rows), 8)
    col_model_map = {}
    last_model_header_row = -1

    for row_idx in range(scan_limit):
        for col_idx, cell in enumerate(rows[row_idx]):
            models = _extract_models_from_cell(cell, known_models)
            if not models:
                continue
            existing = col_model_map.setdefault(col_idx, [])
            for model in models:
                if model not in existing:
                    existing.append(model)
            last_model_header_row = max(last_model_header_row, row_idx)

    if not col_model_map:
        return {}, 0, 0

    width = max(len(row) for row in rows)
    metric_candidates = [i for i in range(width) if i not in col_model_map]
    metric_col = min(metric_candidates) if metric_candidates else 0
    data_start_offset = max(0, last_model_header_row)
    return col_model_map, metric_col, data_start_offset


def _infer_row_model_table(headers: list, data_rows: list, known_models: list) -> tuple[int, bool]:
    rows = [headers] + data_rows[:5]
    width = max(len(row) for row in rows)
    best_col = -1
    best_score = 0

    for col_idx in range(width):
        score = 0
        for row in rows:
            if col_idx < len(row) and _extract_models_from_cell(row[col_idx], known_models):
                score += 1
        if score > best_score:
            best_score = score
            best_col = col_idx

    return best_col, best_score >= 2


def _row_model_metric_value_chunks(headers: list, data_rows: list, known_models: list) -> list:
    """
    Handle rows shaped like:
        PA-3220 | 7.5 Gbps | Firewall throughput (appmix)*

    These appear when the PDF table has no real header row. The first physical
    row is parsed as Markdown headers, so using headers as spec names reverses
    spec_name/spec_value. This function treats headers as data too.
    """
    out = []
    rows = [headers] + data_rows

    for row in rows:
        model_cols = []
        for idx, cell in enumerate(row):
            models = _extract_models_from_cell(cell, known_models)
            if models:
                model_cols.append((idx, models))

        if not model_cols:
            continue

        metric_candidates = [
            (idx, cell)
            for idx, cell in enumerate(row)
            if all(idx != model_idx for model_idx, _ in model_cols)
            and _is_likely_metric_label(cell, known_models)
        ]
        value_candidates = [
            (idx, cell)
            for idx, cell in enumerate(row)
            if all(idx != model_idx for model_idx, _ in model_cols)
            and _is_likely_spec_value(cell)
        ]

        if not metric_candidates or not value_candidates:
            continue

        metric_idx, metric_raw = max(metric_candidates, key=lambda item: len(item[1]))
        value_idx, value_raw = min(value_candidates, key=lambda item: abs(item[0] - metric_idx))
        spec_name = _clean_metric_name(metric_raw)
        value = _clean_cell_value(value_raw)

        if not spec_name or _is_blank_value(value):
            continue

        for _, models in model_cols:
            for model in models:
                out.append((model, spec_name, value))

    return out


def _split_into_segments(text: str) -> list:
    parts = TABLE_BLOCK_RE.split(text)
    return [(part.strip(), bool(TABLE_OPEN_RE.match(part.strip()))) for part in parts]


def _make_chunk_id(doc_name: str, page: int, text: str) -> str:
    return hashlib.md5(f"{doc_name}|{page}|{text}".encode()).hexdigest()


def _base_meta(doc_meta: dict, page: int, chunk_idx: int, has_table: bool, chunk_type: str) -> dict:
    family = doc_meta.get("family", doc_meta.get("product_family", ""))
    return {
        "chunk_type": chunk_type,
        "vendor": doc_meta["vendor"],
        "family": family,
        "product_family": doc_meta.get("product_family", family),
        "display_family": doc_meta.get("display_family", family),
        "model": "",
        "component": "",
        "category": doc_meta["category"],
        "doc_name": doc_meta["doc_name"],
        "source_document": doc_meta["doc_name"],
        "page": page,
        "chunk_idx": chunk_idx,
        "has_table": has_table,
    }


def _make_spec_chunk_text(vendor, family, model, spec_name, value, context_str):
    return (
        f"Vendor: {vendor}\n"
        f"Family: {family}\n"
        f"Model: {model}\n\n"
        f"Feature: {spec_name}\n"
        f"Value: {value}\n\n"
        f"Context: {context_str}"
    )


def _append_spec_chunk(final_chunks, doc_meta, idx, page_num, vendor, family, model, spec_name, value, context_str):
    chunk_text = _make_spec_chunk_text(vendor, family, model, spec_name, value, context_str)
    chunk_id = _make_chunk_id(doc_meta["doc_name"], page_num, chunk_text)
    meta = _base_meta(doc_meta, page_num, idx, True, "spec_row")
    meta.update({
        "model": model,
        "metric": spec_name,
        "value": value,
        "spec_name": spec_name,
        "spec_value": value,
        "chunk_id": chunk_id,
    })
    final_chunks.append({"text": chunk_text, "metadata": meta})
    return idx + 1


def chunk_pages(pages: list, doc_meta: dict, pdf_path: str = "") -> list:
    vendor = doc_meta["vendor"]
    family = doc_meta.get("family", doc_meta.get("product_family", ""))
    category = doc_meta["category"]
    models = _known_model_set(doc_meta)
    context_str = f"{vendor} {family} {category} technical specification".strip()

    prose_raw = []
    table_blocks = []

    for page in pages:
        page_num = page["page"]
        page_text = page["text"]
        for segment, is_table in _split_into_segments(page["text"]):
            if not segment:
                continue
            if is_table:
                table_blocks.append((segment, page_num, page_text))
            else:
                for split in _splitter.split_text(segment):
                    if len(split.strip()) >= MIN_CHUNK_CHARS:
                        prose_raw.append({"text": split.strip(), "page": page_num})

    final_chunks = []
    idx = 0

    for chunk in _deduplicate(prose_raw):
        text = chunk["text"]
        page_num = chunk["page"]
        meta = _base_meta(doc_meta, page_num, idx, False, "semantic")
        meta["chunk_id"] = _make_chunk_id(doc_meta["doc_name"], page_num, text)
        final_chunks.append({"text": text, "metadata": meta})
        idx += 1

    for table_text, page_num, page_text in table_blocks:
        full_meta = _base_meta(doc_meta, page_num, idx, True, "table_full")
        full_meta["chunk_id"] = _make_chunk_id(doc_meta["doc_name"], page_num, table_text)
        final_chunks.append({"text": table_text, "metadata": full_meta})
        idx += 1

        headers, data_rows = _parse_markdown_table(table_text)
        if not headers or not data_rows or not models:
            continue

        row_triples = _row_model_metric_value_chunks(headers, data_rows, models)
        if row_triples and any(_extract_models_from_cell(cell, models) for cell in headers):
            for model, spec_name, value in row_triples:
                idx = _append_spec_chunk(
                    final_chunks, doc_meta, idx, page_num, vendor, family,
                    model, spec_name, value, context_str
                )
            continue

        col_model_map, metric_col, data_start_offset = _infer_column_model_map(headers, data_rows, models)
        row_model_col, is_row_model_table = _infer_row_model_table(headers, data_rows, models)

        if is_row_model_table and (not col_model_map or set(col_model_map) == {row_model_col}):
            for cells in data_rows:
                if row_model_col >= len(cells):
                    continue
                row_models = _extract_models_from_cell(cells[row_model_col], models)
                if not row_models:
                    continue
                for col_i, header in enumerate(headers):
                    if col_i == row_model_col or col_i >= len(cells):
                        continue
                    spec_name = _clean_metric_name(header)
                    value = _clean_cell_value(cells[col_i])
                    if not spec_name or _is_generic_header(spec_name) or _is_blank_value(value):
                        continue
                    for model in row_models:
                        idx = _append_spec_chunk(
                            final_chunks, doc_meta, idx, page_num, vendor, family,
                            model, spec_name, value, context_str
                        )
            continue

        if not col_model_map:
            col_model_map, metric_col = _infer_contextual_column_model_map(
                headers,
                data_rows,
                models,
                page_text + "\n" + table_text,
            )
            data_start_offset = 0
            if not col_model_map:
                logger.debug(f"  [chunker] table page {page_num}: no model columns detected")
                continue

        for cells in data_rows[data_start_offset:]:
            metric_raw = cells[metric_col].strip() if metric_col < len(cells) else ""
            if (
                not metric_raw
                or re.match(r"^[\s\-:\+]+$", metric_raw)
                or _is_placeholder_cell(metric_raw)
                or _is_generic_header(metric_raw)
                or _is_section_label(metric_raw)
                or _extract_models_from_cell(metric_raw, models)
            ):
                continue

            spec_name = _clean_metric_name(metric_raw)
            if not spec_name:
                continue

            for col_i, col_models in sorted(col_model_map.items()):
                value = _clean_cell_value(cells[col_i] if col_i < len(cells) else "")
                if _is_blank_value(value) or _is_generic_header(value) or _is_placeholder_cell(value):
                    continue
                for model in col_models:
                    idx = _append_spec_chunk(
                        final_chunks, doc_meta, idx, page_num, vendor, family,
                        model, spec_name, value, context_str
                    )

    type_counts = {}
    for c in final_chunks:
        chunk_type = c["metadata"]["chunk_type"]
        type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1

    logger.info(
        f"  [chunker] Total: {len(final_chunks)} chunks | "
        + " | ".join(f"{k}: {v}" for k, v in sorted(type_counts.items()))
    )
    return final_chunks

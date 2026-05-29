# =============================================================================
# cleaner.py — Text cleaning pipeline before chunking
# =============================================================================

import re
import unicodedata
import logging
from collections import Counter

logger = logging.getLogger("ingestion")

LIGATURE_MAP = {
    "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi",
    "\ufb04": "ffl", "\ufb00": "ff", "\ufb05": "st",
    "\u2019": "'",  "\u2018": "'",  "\u201c": '"', "\u201d": '"',
    "\u2013": "-",  "\u2014": "-",  "\u00ad": "",
    "\u00a0": " ",  "\u200b": "",   "\ufeff": "",
}

def _fix_unicode(text: str) -> str:
    for bad, good in LIGATURE_MAP.items():
        text = text.replace(bad, good)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\S\n\t ]+", " ", text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")
    return text

BULLET_PATTERN = re.compile(
    r"^[\s]*[●•◦▪▸►➔✓✔✗✘◆◇○□■⬛⬜★☆→⇒≫»·‣⁃–]\s*", re.MULTILINE
)

def _normalise_bullets(text: str) -> str:
    return BULLET_PATTERN.sub("- ", text)

def build_header_footer_filter(all_page_texts: list) -> set:
    line_page_count: Counter = Counter()
    for page_text in all_page_texts:
        lines = [l.strip() for l in page_text.split("\n") if l.strip()]
        candidate_lines = set(lines[:3] + lines[-3:])
        for line in candidate_lines:
            if 5 < len(line) < 200:
                line_page_count[line] += 1
    boilerplate = {line for line, count in line_page_count.items() if count >= 3}
    if boilerplate:
        logger.info(f"  [cleaner] Stripping {len(boilerplate)} header/footer lines.")
    return boilerplate

def _remove_boilerplate(text: str, boilerplate: set) -> str:
    if not boilerplate:
        return text
    lines = text.split("\n")
    return "\n".join(l for l in lines if l.strip() not in boilerplate)

def _clean_whitespace(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")

def _fix_hyphenation(text: str) -> str:
    return HYPHEN_BREAK.sub(r"\1\2", text)

NOISE_LINE = re.compile(r"^[\s\d\.\-\_\|]{0,6}$")

def _remove_noise_lines(text: str) -> str:
    return "\n".join(l for l in text.split("\n") if not NOISE_LINE.match(l))

TABLE_MARKER = "<!-- TABLE -->"

def _split_table_blocks(text: str) -> list:
    parts = text.split(TABLE_MARKER)
    return [(part, i % 2 == 1) for i, part in enumerate(parts)]

def _detect_and_wrap_raw_tables(text: str) -> str:
    lines = text.split("\n")
    n = len(lines)
    is_table_line = ["|" in l.strip() for l in lines]
    in_table = False
    table_lines = []
    output_parts = []
    for i in range(n):
        is_part = is_table_line[i] and (
            (i > 0 and is_table_line[i - 1]) or
            (i < n - 1 and is_table_line[i + 1])
        )
        if is_part:
            in_table = True
            table_lines.append(lines[i])
        else:
            if in_table:
                output_parts.append(
                    "<!-- TABLE -->\n" + "\n".join(table_lines) + "\n<!-- TABLE -->"
                )
                table_lines = []
                in_table = False
            output_parts.append(lines[i])
    if in_table:
        output_parts.append(
            "<!-- TABLE -->\n" + "\n".join(table_lines) + "\n<!-- TABLE -->"
        )
    return "\n".join(output_parts)

def clean_text(text: str, boilerplate=None) -> str:
    parts = text.split(TABLE_MARKER)
    for i in range(0, len(parts), 2):
        parts[i] = _detect_and_wrap_raw_tables(parts[i])
    text = TABLE_MARKER.join(parts)

    segments = _split_table_blocks(text)
    cleaned_segments = []

    for segment, is_table in segments:
        if is_table:
            cleaned_segments.append(segment.strip())
        else:
            s = _fix_unicode(segment)
            s = _fix_hyphenation(s)
            s = _normalise_bullets(s)
            if boilerplate:
                s = _remove_boilerplate(s, boilerplate)
            s = _remove_noise_lines(s)
            s = _clean_whitespace(s)
            cleaned_segments.append(s)

    result_parts = []
    for i, (segment, is_table) in enumerate(segments):
        cleaned = cleaned_segments[i]
        if not cleaned:
            continue
        if is_table:
            result_parts.append("\n\n[TABLE]\n" + cleaned + "\n[/TABLE]")
        else:
            result_parts.append(cleaned)

    return "\n\n".join(p for p in result_parts if p.strip())

def clean_document(pages: list) -> list:
    all_texts   = [p["text"] for p in pages]
    boilerplate = build_header_footer_filter(all_texts)
    cleaned_pages = []
    for page in pages:
        cleaned = clean_text(page["text"], boilerplate)
        if cleaned.strip():
            cleaned_pages.append({**page, "text": cleaned})
        else:
            logger.debug(f"  [cleaner] Page {page['page']} empty after cleaning.")
    logger.info(f"  [cleaner] {len(pages)} -> {len(cleaned_pages)} pages after cleaning.")
    return cleaned_pages

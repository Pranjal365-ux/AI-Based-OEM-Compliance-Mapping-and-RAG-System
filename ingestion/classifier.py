# =============================================================================
# classifier.py — Robust category detection for OEM documents
# =============================================================================

import re
import logging
from config import CATEGORY_TAXONOMY

logger = logging.getLogger("ingestion")


def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _score_category(cat_name: str, cat_def: dict,
                    filename_norm: str, head_norm: str, body_norm: str) -> float:
    score = 0.0
    for kw in cat_def.get("keywords", []):
        kw_n = _normalise(kw)
        if kw_n in filename_norm: score += 3.0
        if kw_n in head_norm:     score += 2.0
        if kw_n in body_norm:     score += 1.0
    for tw in cat_def.get("title_words", []):
        if _normalise(tw) in filename_norm:
            score += 5.0
    for neg in cat_def.get("negative", []):
        neg_n = _normalise(neg)
        if neg_n in filename_norm: score -= 4.0
        if neg_n in head_norm:     score -= 2.0
    return max(score, 0.0)


def detect_category(filename: str, full_text: str):
    filename_norm = _normalise(filename)
    head_norm     = _normalise(full_text[:1000])
    body_norm     = _normalise(full_text)

    scores = {
        cat: _score_category(cat, defn, filename_norm, head_norm, body_norm)
        for cat, defn in CATEGORY_TAXONOMY.items()
    }
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_cat, top_score   = ranked[0]
    second_score         = ranked[1][1] if len(ranked) > 1 else 0.0

    if top_score == 0.0:
        logger.warning(f"  [classifier] No match for '{filename}' → GENERAL")
        return "GENERAL", 0.0

    total      = top_score + second_score
    confidence = (top_score - second_score) / total if total > 0 else 0.0

    if confidence < 0.35:
        logger.warning(
            f"  [classifier] Low confidence ({confidence:.2f}) '{filename}' "
            f"→ {top_cat} ({top_score:.1f}) vs {ranked[1][0]} ({second_score:.1f})"
        )

    logger.info(f"  [classifier] '{filename}' -> {top_cat} (score={top_score:.1f}, conf={confidence:.2f})")
    return top_cat, round(confidence, 3)

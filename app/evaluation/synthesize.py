"""
Stratified eval-pool selection for synthetic QA generation.

Selects a diverse subset of page Documents (parent pool) produced by
``app.rag.ingestion.load_pdf_file``. Filters short pages and References,
then round-robin samples across canonical sections so generated triplets
do not all come from the Introduction.
"""

from __future__ import annotations

import random
from collections import defaultdict

from langchain_core.documents import Document

from app.rag.ingestion import canonical_section

MIN_CHARS_DEFAULT = 500


def is_picture_text_page(text: str) -> bool:
    """Detect figure/mindmap picture-text dumps unsuitable for QA synth."""
    if not text:
        return True
    picture_markers = text.count("<br>") + text.count("<!-- Start of picture text")
    if picture_markers > 0 and (picture_markers * 30) > len(text) * 0.3:
        return True
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True
    frag = sum(
        1
        for ln in lines
        if len(ln.split()) < 4 or ln.count("|") >= 2 or "<" in ln or ">" in ln
    )
    has_fig = any(
        ln.lower().startswith("fig.") or ln.lower().startswith("figure")
        for ln in lines
    )
    avg_len = sum(len(ln) for ln in lines) / len(lines)
    if has_fig and avg_len < 60 and frag / len(lines) > 0.4:
        return True
    if frag / len(lines) > 0.6 and avg_len < 40:
        return True
    return False


def select_eval_pool(
    page_docs: list[Document],
    total: int = 25,
    min_chars: int = MIN_CHARS_DEFAULT,
    seed: int = 42,
) -> list[Document]:
    """Return up to ``total`` pages evenly spread across sections."""
    cands = [
        d
        for d in page_docs
        if len(d.page_content or "") >= min_chars
        and canonical_section((d.metadata or {}).get("section")) != "References"
        and not is_picture_text_page(d.page_content or "")
    ]
    if not cands:
        return []

    groups: dict[str, list[Document]] = defaultdict(list)
    for doc in cands:
        groups[canonical_section((doc.metadata or {}).get("section"))].append(doc)

    rnd = random.Random(seed)
    for docs_in_section in groups.values():
        rnd.shuffle(docs_in_section)

    pool: list[Document] = []
    while len(pool) < min(total, len(cands)):
        progressed = False
        for section in sorted(groups):
            if groups[section] and len(pool) < total:
                pool.append(groups[section].pop(0))
                progressed = True
        if not progressed:
            break
    return pool

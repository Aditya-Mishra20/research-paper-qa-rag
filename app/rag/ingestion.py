"""
Document ingestion for PDF and text research papers.

Extraction is intentionally model-free: PyMuPDF reads PDFs from disk, and all
embedding/generation intelligence is handled by API modules later in the RAG
pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

try:
    import pymupdf
except ImportError:  # pragma: no cover - older PyMuPDF exposes fitz first.
    import fitz as pymupdf

try:
    import pymupdf4llm

    _HAS_MD = True
except ImportError:  # pragma: no cover - markdown extraction is optional.
    pymupdf4llm = None  # type: ignore
    _HAS_MD = False

HYPHEN_RE = re.compile(r"(\w)-\n(\w)")
WS_RE = re.compile(r"[ \t]+")


SECTION_RE = re.compile(
    r"^#{0,6}\s*(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(abstract|introduction|related work|related-work|background|"
    r"method(?:ology|ologies|s)?|approach|architecture|model|framework|"
    r"experiment(?:s|al(?: setup| results)?)?|evaluation|results?(?: and discussion)?|"
    r"discussion|analysis|ablation(?:s| studies)?|conclusion(?:s| and future work)?|"
    r"future work|limitations?|references|appendix(?: [A-Z])?)\s*:?\s*$",
    re.IGNORECASE,
)


CANONICAL_SECTIONS = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "related work": "Related Work",
    "related-work": "Related Work",
    "background": "Background",
    "method": "Method",
    "methodology": "Method",
    "methods": "Method",
    "methodologies": "Method",
    "approach": "Method",
    "architecture": "Method",
    "model": "Method",
    "framework": "Method",
    "experiments": "Experiments",
    "experiment": "Experiments",
    "evaluation": "Experiments",
    "results": "Results",
    "result": "Results",
    "discussion": "Discussion",
    "analysis": "Analysis",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "future work": "Conclusion",
    "limitation": "Limitations",
    "limitations": "Limitations",
    "ablation": "Experiments",
    "ablations": "Experiments",
    "appendix": "Appendix",
    "references": "References",
}


def canonical_section(section: str | None) -> str:
    """Map raw heading variants to a stable bucket for stratified sampling."""
    if not section:
        return "Unknown"
    key = re.sub(r"^#{0,6}\s*(\d+(\.\d+)*\.?\s+)?", "", section).strip()
    key = key.strip().strip(":").strip().lower()
    return CANONICAL_SECTIONS.get(key, section.strip().strip("#: ").title() or "Unknown")


def _dehyphenate(text: str) -> str:
    """Join words split by hyphen + newline; drop lone page numbers."""
    text = HYPHEN_RE.sub(r"\1\2", text)
    lines = [WS_RE.sub(" ", ln).strip() for ln in text.splitlines()]
    out: list[str] = []
    for ln in lines:
        if re.fullmatch(r"\d{1,4}", ln or ""):
            continue
        out.append(ln)
    return "\n".join(out).strip()


def _strip_headers_footers(page_texts: list[str]) -> list[str]:
    """Remove first/last lines repeated on >40% of pages (running headers)."""
    if len(page_texts) < 3:
        return page_texts
    firsts: dict[str, int] = {}
    lasts: dict[str, int] = {}
    for t in page_texts:
        lines = t.splitlines()
        if not lines:
            continue
        firsts[lines[0].strip()] = firsts.get(lines[0].strip(), 0) + 1
        lasts[lines[-1].strip()] = lasts.get(lines[-1].strip(), 0) + 1
    n = len(page_texts)
    bad_first = {k for k, c in firsts.items() if c / n > 0.4 and 0 < len(k) < 120}
    bad_last = {k for k, c in lasts.items() if c / n > 0.4 and 0 < len(k) < 120}
    cleaned: list[str] = []
    for t in page_texts:
        lines = t.splitlines()
        if lines and lines[0].strip() in bad_first:
            lines = lines[1:]
        if lines and lines[-1].strip() in bad_last:
            lines = lines[:-1]
        cleaned.append("\n".join(lines).strip())
    return cleaned


def _extract_page_text(page: Any, file: Path, pageno: int) -> str:
    """Prefer markdown (keeps tables/headings), fallback to plain text."""
    if _HAS_MD:
        try:
            md = pymupdf4llm.to_markdown(str(file), pages=[pageno - 1])
            if md and len(md.strip()) >= 50:
                return md.strip()
        except Exception:
            pass
    try:
        return (page.get_text("text", sort=True) or "").strip()
    except Exception:
        return ""


def detect_section(text: str, current: str | None = None) -> str | None:
    """Return the first recognizable section heading in a page of text."""
    keywords = (
        "abstract", "introduction", "related work", "background", "method",
        "methodology", "methods", "approach", "architecture", "model",
        "framework", "experiment", "evaluation", "result", "discussion",
        "analysis", "ablation", "conclusion", "future work", "limitation",
        "references", "appendix",
    )
    for line in text.splitlines():
        normalized = line.strip().strip(":").strip()
        if not normalized or len(normalized) > 150:
            continue
        if SECTION_RE.match(normalized):
            return canonical_section(normalized)
        # Tolerant markdown heading: "# _A. Methodology Overview_",
        # "**Abstract**", "A. Results and Discussion"
        stripped = normalized.lstrip("#").strip().strip("*_").strip()
        stripped = re.sub(r"^[A-Z]\.\s+", "", stripped).strip("*_ ").strip()
        if stripped.startswith("#") or line.lstrip().startswith("#") or re.match(
            r"^[A-Z]\.\s+\w+", normalized
        ):
            lowered = stripped.lower().strip(":")
            for kw in keywords:
                if kw in lowered:
                    return canonical_section(kw)
    return current


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Chroma accepts only primitive non-null metadata values."""
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif isinstance(value, (list, tuple)):
            clean[key] = ", ".join(str(item) for item in value)
        else:
            clean[key] = str(value)
    return clean


def load_pdf_file(file: Path, metadata: dict[str, Any] | None = None) -> list[Document]:
    """Extract one LangChain document per readable PDF page."""
    base_metadata = metadata or {}
    try:
        pdf = pymupdf.open(file)
    except Exception:
        return []

    documents: list[Document] = []
    pdf_metadata = pdf.metadata or {}
    title = base_metadata.get("title") or pdf_metadata.get("title") or file.stem
    authors = base_metadata.get("authors") or pdf_metadata.get("author") or ""
    current_section: str | None = None

    try:
        raw_texts: list[str] = []
        for index, page in enumerate(pdf, start=1):
            raw_texts.append(_extract_page_text(page, file, index))

        raw_texts = [_dehyphenate(t) for t in raw_texts]
        raw_texts = _strip_headers_footers(raw_texts)

        for index, text in enumerate(raw_texts, start=1):
            text = (text or "").strip()
            if len(text) < 50:
                continue

            current_section = detect_section(text, current_section)
            page_metadata = {
                **base_metadata,
                "source": str(file),
                "source_filename": file.name,
                "page": index,
                "title": title,
                "authors": authors,
                "section": current_section or "Unknown",
                "is_scanned": False,
            }
            documents.append(
                Document(
                    page_content=text,
                    metadata=_clean_metadata(page_metadata),
                )
            )
    finally:
        pdf.close()

    return documents


def load_text_file(file: Path, metadata: dict[str, Any] | None = None) -> list[Document]:
    """Load a text file as a single document."""
    text = file.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    base_metadata = metadata or {}
    doc_metadata = {
        **base_metadata,
        "source": str(file),
        "source_filename": file.name,
        "page": 1,
        "title": base_metadata.get("title") or file.stem,
        "authors": base_metadata.get("authors") or "",
        "section": detect_section(text) or "Unknown",
        "is_scanned": False,
    }
    return [Document(page_content=text, metadata=_clean_metadata(doc_metadata))]


def load_file(file: Path, metadata: dict[str, Any] | None = None) -> list[Document]:
    """Dispatch to the correct local loader."""
    suffix = file.suffix.lower()
    if suffix == ".pdf":
        return load_pdf_file(file, metadata)
    if suffix == ".txt":
        return load_text_file(file, metadata)
    return []


def load_all_documents(data_dir: str) -> list[Document]:
    """Load all supported files from a directory tree."""
    data_path = Path(data_dir).resolve()
    if not data_path.exists():
        return []

    documents: list[Document] = []
    for file in list(data_path.glob("**/*.pdf")) + list(data_path.glob("**/*.txt")):
        documents.extend(load_file(file))
    return documents

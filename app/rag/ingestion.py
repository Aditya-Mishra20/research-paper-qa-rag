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


SECTION_RE = re.compile(
    r"^\s*(abstract|introduction|related work|background|method(?:ology)?|"
    r"approach|experiments?|evaluation|results?|discussion|conclusion|"
    r"references)\s*$",
    re.IGNORECASE,
)


def detect_section(text: str, current: str | None = None) -> str | None:
    """Return the first recognizable section heading in a page of text."""
    for line in text.splitlines():
        normalized = line.strip().strip(":").strip()
        if SECTION_RE.match(normalized):
            return normalized.title()
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
        for index, page in enumerate(pdf, start=1):
            try:
                text = (page.get_text("text", sort=True) or "").strip()
            except Exception:
                continue
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

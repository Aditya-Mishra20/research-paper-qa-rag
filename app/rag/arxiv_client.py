"""
arXiv fetch helper for free canonical paper metadata and PDFs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import arxiv


ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")


def fetch_arxiv_pdf(arxiv_id: str, upload_dir: Path) -> dict[str, Any]:
    """Download an arXiv PDF and return metadata for ingestion."""
    paper_id = arxiv_id.strip()
    if not ARXIV_ID_RE.match(paper_id):
        raise ValueError("Invalid arXiv ID. Expected format like 2401.12345 or 2401.12345v2.")

    client = arxiv.Client(delay_seconds=3)
    search = arxiv.Search(id_list=[paper_id])
    paper = next(client.results(search), None)
    if paper is None:
        raise ValueError(f"No arXiv paper found for {paper_id}.")

    upload_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{paper_id}.pdf"
    paper.download_pdf(dirpath=str(upload_dir), filename=filename)
    authors = ", ".join(author.name for author in paper.authors)
    published = paper.published.date().isoformat() if paper.published else ""
    return {
        "path": upload_dir / filename,
        "metadata": {
            "doc_id": paper_id,
            "arxiv_id": paper_id,
            "title": paper.title,
            "authors": authors,
            "published": published,
            "pdf_url": paper.pdf_url,
            "summary": paper.summary,
        },
    }

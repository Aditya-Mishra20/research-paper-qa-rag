"""
End-to-end ingestion pipeline for local files.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from app.rag.chunking import chunk_documents
from app.rag.embeddings import EMBEDDING_MODEL, embed_texts
from app.rag.ingestion import load_file
from app.rag.vectorstore import add_chunks, sha_exists

# Quota-capped: eval triplets per new document (raise when quota allows).
EVAL_TRIPLET_COUNT = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ingest_file(path: Path, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load, chunk, embed, and store one file."""
    sha256 = sha256_file(path)
    base_metadata = dict(metadata or {})
    doc_id = str(base_metadata.get("doc_id") or base_metadata.get("arxiv_id") or sha256[:12])
    base_metadata.update(
        {
            "doc_id": doc_id,
            "sha256": sha256,
            "embedding_model": EMBEDDING_MODEL,
        }
    )

    if sha_exists(sha256):
        return {
            "filename": path.name,
            "doc_id": doc_id,
            "pages": 0,
            "chunks": 0,
            "skipped": True,
            "reason": "duplicate",
        }

    documents = load_file(path, base_metadata)
    chunks = chunk_documents(documents)
    if not chunks:
        return {
            "filename": path.name,
            "doc_id": doc_id,
            "pages": 0,
            "chunks": 0,
            "skipped": False,
            "reason": "no_text",
        }

    embeddings = embed_texts([chunk.page_content for chunk in chunks])
    stored = add_chunks(chunks, embeddings)
    pages = len({chunk.metadata.get("page") for chunk in chunks})

    # Best-effort eval triplets: cached by sha256, never breaks ingest
    # (quota errors, 503s, or missing keys are swallowed).
    triplets_path: str | None = None
    try:
        from app.evaluation.synthesize import select_eval_pool
        from app.evaluation.triplets import build_triplets

        pool = select_eval_pool(documents, total=EVAL_TRIPLET_COUNT)
        if pool:
            triplets_path = str(build_triplets(pool, sha256, doc_id))
    except Exception:
        triplets_path = None

    return {
        "filename": path.name,
        "doc_id": doc_id,
        "pages": pages,
        "chunks": stored,
        "skipped": False,
        "eval_triplets": triplets_path,
    }

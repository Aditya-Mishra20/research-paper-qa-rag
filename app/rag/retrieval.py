"""
Retrieval against Chroma using API-generated query embeddings.
"""

from __future__ import annotations

from typing import Any

from app.rag.embeddings import embed_query
from app.rag.vectorstore import get_collection


def _citation(metadata: dict[str, Any]) -> str:
    title = metadata.get("title") or metadata.get("source_filename") or "Untitled"
    page = metadata.get("page") or "?"
    return f"{title}, p.{page}"


def retrieve(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Return top-k chunks with metadata, citation, and cosine-derived score."""
    collection = get_collection()
    query_embedding = embed_query(query)
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=max(1, top_k),
        include=["documents", "metadatas", "distances"],
    )

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    hits: list[dict[str, Any]] = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        metadata = metadata or {}
        hits.append(
            {
                "text": text,
                "metadata": metadata,
                "citation": _citation(metadata),
                "score": max(0.0, 1.0 - float(distance)),
            }
        )
    return hits

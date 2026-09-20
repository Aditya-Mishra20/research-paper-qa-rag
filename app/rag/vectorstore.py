"""
Chroma vector store helpers using explicit API-provided embeddings.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import chromadb
from langchain_core.documents import Document


CHROMA_PATH = Path("chroma_db")
COLLECTION_NAME = "papers"


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )


def _metadata_for_chroma(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
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


def sha_exists(sha256: str) -> bool:
    collection = get_collection()
    existing = collection.get(where={"sha256": sha256}, limit=1)
    return bool(existing.get("ids"))


def add_chunks(
    chunks: list[Document],
    embeddings: list[list[float]],
    batch_size: int = 5000,
) -> int:
    """Replace chunks for each doc_id, then add the supplied chunk embeddings."""
    if not chunks:
        return 0
    if len(chunks) != len(embeddings):
        raise ValueError("Chunk and embedding counts must match.")

    collection = get_collection()
    doc_ids = {str(chunk.metadata["doc_id"]) for chunk in chunks}
    for doc_id in doc_ids:
        collection.delete(where={"doc_id": doc_id})

    ids = [str(chunk.metadata["chunk_id"]) for chunk in chunks]
    documents = [chunk.page_content for chunk in chunks]
    metadatas = [_metadata_for_chroma(chunk.metadata) for chunk in chunks]

    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )
    return len(chunks)


def list_documents() -> list[dict[str, Any]]:
    """Return distinct documents currently known to Chroma."""
    collection = get_collection()
    result = collection.get(include=["metadatas"])
    documents: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for metadata in result.get("metadatas") or []:
        if not metadata:
            continue
        doc_id = str(metadata.get("doc_id", ""))
        if not doc_id or doc_id in documents:
            continue
        documents[doc_id] = {
            "doc_id": doc_id,
            "title": metadata.get("title") or metadata.get("source_filename") or doc_id,
            "source_filename": metadata.get("source_filename", ""),
            "authors": metadata.get("authors", ""),
            "arxiv_id": metadata.get("arxiv_id", ""),
            "published": metadata.get("published", ""),
            "sha256": metadata.get("sha256", ""),
        }
    return list(documents.values())

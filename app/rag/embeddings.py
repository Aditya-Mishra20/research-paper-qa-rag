"""
Local CPU embeddings with Sentence-Transformers.

The first call downloads sentence-transformers/all-MiniLM-L6-v2 from Hugging
Face Hub, then later runs can reuse the local cache offline.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_model: Any | None = None


class EmbeddingError(RuntimeError):
    """Raised when the embedding model cannot complete a request."""


def _get_model() -> Any:
    """Lazily load the local embedding model on CPU."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        except Exception as exc:
            raise EmbeddingError(str(exc)) from exc
    return _model


def embed_texts(texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
    """Embed texts with normalized MiniLM vectors."""
    items = list(texts)
    if not items:
        return []

    try:
        embeddings = _get_model().encode(
            [text or " " for text in items],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        raise EmbeddingError(str(exc)) from exc
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query."""
    vectors = embed_texts([query], batch_size=1)
    return vectors[0]

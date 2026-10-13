"""
Section-aware chunking for research paper documents.

Token-based: sized for the MiniLM embedder (256-token hard limit) using its
own WordPiece tokenizer, with prev/next links for context expansion.
"""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 256  # tokens (MiniLM truncates beyond 256)
DEFAULT_CHUNK_OVERLAP = 50  # tokens

_TOKENIZER: Callable[[str], int] | None = None
_TOKENIZER_FAILED = False


def _token_length_function() -> Callable[[str], int]:
    """MiniLM token counter; falls back to chars//4 when offline."""
    global _TOKENIZER, _TOKENIZER_FAILED
    if _TOKENIZER is not None:
        return _TOKENIZER
    if not _TOKENIZER_FAILED:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2", use_fast=True
            )

            def _count(text: str) -> int:
                return len(tok.encode(text, add_special_tokens=False))

            _TOKENIZER = _count
            return _TOKENIZER
        except Exception:
            _TOKENIZER_FAILED = True
    return lambda s: max(1, len(s) // 4)


def chunk_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents (token units) without crossing page boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=_token_length_function(),
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks: list[Document] = []
    for document in documents:
        split_docs = splitter.split_documents([document])
        for index, chunk in enumerate(split_docs):
            metadata = dict(chunk.metadata)
            doc_id = metadata.get("doc_id", "doc")
            page = metadata.get("page", "1")
            chunk_id = f"{doc_id}:p{page}:{index}"
            metadata["chunk_id"] = chunk_id
            metadata["chunk_index"] = index
            chunks.append(Document(page_content=chunk.page_content, metadata=metadata))

    # Neighbor links within each doc for context expansion at generation.
    order: dict[str, list[int]] = {}
    for i, chunk in enumerate(chunks):
        order.setdefault(str(chunk.metadata.get("doc_id", "doc")), []).append(i)
    for idxs in order.values():
        for pos, i in enumerate(idxs):
            prev_id = str(chunks[idxs[pos - 1]].metadata["chunk_id"]) if pos > 0 else ""
            next_id = (
                str(chunks[idxs[pos + 1]].metadata["chunk_id"])
                if pos < len(idxs) - 1
                else ""
            )
            chunks[i].metadata["prev_chunk_id"] = prev_id
            chunks[i].metadata["next_chunk_id"] = next_id
    return chunks

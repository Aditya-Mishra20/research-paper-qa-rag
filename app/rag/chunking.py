"""
Section-aware chunking for research paper documents.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def chunk_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents without crossing page/section document boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
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
    return chunks

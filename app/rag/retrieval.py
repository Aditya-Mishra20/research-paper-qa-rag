"""
Retrieval & Vector Store Module

Purpose:
- Generates text embeddings for query and document chunks using embedding models (e.g. OpenAI, HuggingFace/SentenceTransformers).
- Manages vector database indexes (e.g. Pinecone, Qdrant, ChromaDB, FAISS).
- Executes similarity search, hybrid search (dense + sparse BM25), and optional reranking (e.g. Cohere Rerank).
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Research Paper QA API",
    description="FastAPI Backend for Retrieval-Augmented Generation (RAG) system",
    version="0.1.0",
)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]


@app.get("/")
def root():
    return {"message": "Research Paper QA RAG System API is running."}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """
    Query endpoint for research paper question-answering.
    """
    try:
        # TODO: Integrate RAG agent pipeline
        return QueryResponse(
            query=request.query,
            answer="This is a placeholder response for your query.",
            sources=["sample_paper_1.pdf"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

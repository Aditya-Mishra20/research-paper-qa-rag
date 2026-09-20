from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.rag.arxiv_client import fetch_arxiv_pdf
from app.rag.generation import GenerationError, generate_answer
from app.rag.pipeline import ingest_file
from app.rag.retrieval import retrieve
from app.rag.vectorstore import list_documents

load_dotenv()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_FILE_SIZE = 20 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".txt"}

app = FastAPI(
    title="Research Paper QA API",
    description="FastAPI Backend for Retrieval-Augmented Generation (RAG) system",
    version="0.1.0",
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class ArxivFetchRequest(BaseModel):
    arxiv_id: str


@app.get("/")
def root():
    return {"message": "Research Paper QA RAG System API is running."}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/documents")
def documents():
    return {"documents": list_documents()}


@app.post("/upload")
async def upload_papers(files: list[UploadFile] = File(...)):
    """Upload and ingest multiple PDF/TXT files with partial success."""
    saved: list[dict] = []
    errors: list[str] = []

    for file in files:
        filename = Path(file.filename or "").name
        suffix = Path(filename).suffix.lower()
        if not filename:
            errors.append("unnamed: missing filename")
            continue
        if suffix not in ALLOWED_EXTENSIONS:
            errors.append(f"{filename}: only PDF/TXT allowed")
            continue

        content = await file.read()
        if not content:
            errors.append(f"{filename}: empty file")
            continue
        if len(content) > MAX_FILE_SIZE:
            errors.append(f"{filename}: file too large (max 20 MB)")
            continue

        path = UPLOAD_DIR / filename
        try:
            path.write_bytes(content)
            saved.append(ingest_file(path))
        except Exception as exc:
            errors.append(f"{filename}: {exc}")

    return {"saved": saved, "errors": errors}


@app.post("/arxiv/fetch")
def fetch_arxiv(request: ArxivFetchRequest):
    """Fetch an arXiv PDF, then ingest it into Chroma."""
    try:
        fetched = fetch_arxiv_pdf(request.arxiv_id, UPLOAD_DIR)
        result = ingest_file(fetched["path"], fetched["metadata"])
        return {"paper": fetched["metadata"], "ingestion": result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _is_quota_error(message: str) -> bool:
    lowered = message.lower()
    return "429" in message or "quota" in lowered or "rate" in lowered


@app.post("/query")
def query_rag(request: QueryRequest):
    """Retrieve relevant chunks and generate a cited answer."""
    try:
        contexts = retrieve(request.query, request.top_k)
        generated = generate_answer(request.query, contexts)
        return {
            "query": request.query,
            "answer": generated["answer"],
            "sources": generated["sources"],
        }
    except GenerationError as exc:
        # Only real quota errors are 429; anything else (e.g. a closed
        # HTTP client, auth failure) is a 500 so it isn't misdiagnosed.
        if _is_quota_error(str(exc)):
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

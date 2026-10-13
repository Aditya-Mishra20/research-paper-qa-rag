# Research Paper QA RAG System

A modular Python-based Retrieval-Augmented Generation (RAG) system for question-answering over research papers. Built with FastAPI, Streamlit, MCP Server, and Evaluation suites.

---

## 🎬 Demo

Upload a PDF or drop an arXiv ID, ask anything, get answers where every claim cites its page.

<video src="brag-output/brag.mp4" poster="brag-output/brag.jpg" controls preload="metadata" width="100%"></video>

> I stopped reading papers line-by-line — now I interrogate them. Upload a PDF or drop an arXiv ID, get answers where every claim cites its page. Free-tier RAG: FastAPI + Chroma + local MiniLM, zero embedding quota.

Can't see the video above? [Watch `brag-output/brag.mp4`](brag-output/brag.mp4) · [Poster](brag-output/brag.jpg)

---

## 📁 Project Structure

```text
├── app/
│   ├── main.py                 # FastAPI backend entrypoint
│   ├── streamlit_app.py        # Streamlit frontend UI
│   ├── rag/
│   │   ├── ingestion.py        # Document loading & extraction
│   │   ├── chunking.py         # Text splitting & chunking strategies
│   │   ├── retrieval.py        # Embeddings & vector search retriever
│   │   └── generation.py       # LLM generation & response synthesis
│   ├── agent/
│   │   └── agent.py            # LangChain / Custom RAG Agent pipeline
│   ├── mcp_server/
│   │   └── server.py           # Model Context Protocol (MCP) server integration
│   └── evaluation/
│       ├── golden_set.json     # Ground truth evaluation dataset
│       └── run_evals.py        # RAG evaluation metrics script (e.g. Ragas / TruLens)
├── data/                       # Sample research papers (.pdf, .txt)
├── .env.example                # Template for environment variables
├── .gitignore                  # Git exclusion rules
├── pyproject.toml              # Single source of truth for dependencies
├── uv.lock                     # Locked dependency versions
└── README.md                   # Project documentation
```

---

## 🛠️ Environment Setup & Installation (using `uv`)

This project uses [`uv`](https://github.com/astral-sh/uv) for fast Python package management.

### 1. Create Virtual Environment
```bash
uv venv
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate     # On Windows
```

### 2. Install Dependencies
Using `uv` (`pyproject.toml` is the single source of truth):
```bash
uv sync
```

### 3. Setup Environment Variables
Copy `.env.example` to `.env` and fill in required API keys:
```bash
cp .env.example .env
```

---

## 🚀 Running the System

### Start FastAPI Server
```bash
uvicorn app.main:app --reload
```
API Documentation will be available at `http://localhost:8000/docs`.

### Start Streamlit Frontend UI
```bash
streamlit run app/streamlit_app.py
```

### Run MCP Server
```bash
python -m app.mcp_server.server
```

### Run Evaluations
```bash
python app/evaluation/run_evals.py
```

---

## 🔒 Security Notice

- Ensure `.env` is never committed to Git (`.env` is included in `.gitignore`).
- Keep API credentials secure.

---

## 📝 Decision Log

- Clean PDF text in-memory: markdown-first extraction, de-hyphenate, strip repeated headers/footers.
- Detect sections tolerantly (`#`, `1.`, `A.` prefixes, markdown bold/italic) and map to stable buckets.
- Stratified eval pool: 25 pages spread across sections, skip `<500` chars, `References`, picture-text dumps.
- Synthetic triplets per-PDF `(Q, GT, Source)`, cached by `sha256`, temp `0.0`, retry/backoff; auto-built on ingest (best-effort, quota-capped at 5).
- Generation on `gemini-3-flash-preview`, judges + loop on local Ollama (`qwen2.5:1.5b`); plain `gemini-3-flash` ID verified nonexistent (404).
- Token chunking (256/50, MiniLM tokenizer) with prev/next neighbor links; Chroma refreshed.
- Test loop: recall 60%@k=3 → 100%@k=5 and @k=10; top_k=5 confirmed.
- Judges: evidence-anchored rubric + cosine/citation guardrails; 1.5b judge unusable (all-zero vs 0.7+ cosine) — needs 7b judge.
- Negatives: 1/4 pass; OOD answered from parametric memory — traced to citation-filter fallback keeping raw answers; score-gated refusal pending.
- No reranker / Ragas yet: single-stage Chroma cosine only, eval runner still stub.

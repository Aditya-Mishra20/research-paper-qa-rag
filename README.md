# Research Paper QA RAG System

A modular Python-based Retrieval-Augmented Generation (RAG) system for question-answering over research papers. Built with FastAPI, Streamlit, MCP Server, and Evaluation suites.

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
├── notebooks/                  # Experimental Jupyter notebooks
├── .env.example                # Template for environment variables
├── .gitignore                  # Git exclusion rules
├── pyproject.toml              # Project dependencies & tool configurations
├── requirements.txt            # Package dependencies list
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
Using `uv`:
```bash
uv pip install -r requirements.txt
# or via pyproject.toml
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

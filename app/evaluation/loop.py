"""
Step 5 test loop: run each eval triplet through retrieve + generate.

For every (Q, GT, Source) triplet, retrieves at each top_k and generates an
answer with the raw (unstripped) LLM text preserved. Writes
``data/evals/results_{sha12}.json`` with per-k retrieved ids/pages plus a
page-hit flag (was the Source page retrieved?) so top_k issues show without
any judge calls.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.rag.generation import generate_answer
from app.rag.retrieval import retrieve

RESULTS_DIR = Path("data/evals")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_triplet(triplet: dict, top_ks: tuple[int, ...] = (3, 5, 10)) -> dict:
    """Retrieve + generate for one triplet at each top_k."""
    question = triplet.get("q", "")
    source_page = triplet.get("page")
    per_k: dict[str, dict] = {}
    for k in top_ks:
        retrieved = retrieve(question, top_k=k)
        pages = [str((r.get("metadata") or {}).get("page")) for r in retrieved]
        generated = generate_answer(question, retrieved, strip_uncited=False)
        per_k[str(k)] = {
            "retrieved_ids": [
                str((r.get("metadata") or {}).get("chunk_id", "")) for r in retrieved
            ],
            "retrieved_pages": pages,
            "page_hit": str(source_page) in pages,
            "answer_raw": generated.get("answer_raw", ""),
            "answer": generated.get("answer", ""),
        }
    return {"q": question, "gt": triplet.get("gt", ""), "page": source_page,
            "section": triplet.get("section"), "per_k": per_k}


def run_eval(triplets_path: str | Path, top_ks: tuple[int, ...] = (3, 5, 10)) -> Path:
    """Run all triplets in a file; write results_{sha12}.json."""
    triplets_path = Path(triplets_path)
    triplets = json.loads(triplets_path.read_text(encoding="utf-8"))
    results = [run_triplet(t, top_ks) for t in triplets]
    out = RESULTS_DIR / f"results_{triplets_path.stem.split('_', 1)[1]}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return out

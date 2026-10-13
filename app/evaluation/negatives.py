"""
Step 7 negatives: adversarial refusal / fallback tests.

No judge LLM needed: graded by normalized string match (refusal) and
retrieval-score threshold (vague). A fail here that retrieval passes
means the pipeline answers unanswerable questions confidently.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.rag.generation import generate_answer
from app.rag.retrieval import retrieve

RESULTS_DIR = Path("data/evals")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_OF_DOMAIN_QS = [
    "How do I bake a chocolate cake from scratch?",
    "What is the capital of France?",
]

VAGUE_QS = [
    "What were the results?",
    "Tell me about it.",
]

REFUSAL_MARKER = "not found"
VAGUE_SCORE_THRESHOLD = 0.3


def _check_refusal(answer: str) -> bool:
    return REFUSAL_MARKER in (answer or "").lower()


def run_negatives(top_k: int = 5) -> dict:
    """Run fixed adversarial prompts; return graded cases."""
    cases: list[dict] = []
    for q in OUT_OF_DOMAIN_QS:
        contexts = retrieve(q, top_k=top_k)
        generated = generate_answer(q, contexts)
        answer = generated.get("answer", "")
        passed = _check_refusal(answer)
        cases.append(
            {
                "type": "out_of_domain",
                "q": q,
                "top_score": (contexts[0].get("score") if contexts else 0.0),
                "answer": answer,
                "expected": "refusal",
                "passed": passed,
            }
        )
    for q in VAGUE_QS:
        contexts = retrieve(q, top_k=top_k)
        top_score = float(contexts[0].get("score") or 0.0) if contexts else 0.0
        generated = generate_answer(q, contexts)
        answer = generated.get("answer", "")
        # Vague Q should fallback when evidence is thin; currently no
        # score-gated fallback exists, so this documents the gap.
        passed = (top_score < VAGUE_SCORE_THRESHOLD) and (
            _check_refusal(answer) or "clarif" in answer.lower()
        )
        cases.append(
            {
                "type": "vague",
                "q": q,
                "top_score": round(top_score, 3),
                "answer": answer,
                "expected": "fallback_when_thin",
                "passed": passed,
            }
        )
    return {"cases": cases,
            "passed": sum(1 for c in cases if c["passed"]),
            "total": len(cases)}


def run_negatives_file(doc_sha: str, top_k: int = 5) -> Path:
    """Run negatives and write negatives_{sha12}.json."""
    out = RESULTS_DIR / f"negatives_{doc_sha[:12]}.json"
    out.write_text(
        json.dumps(run_negatives(top_k), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out

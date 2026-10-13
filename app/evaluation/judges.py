"""
Step 6 judges: score context relevance, faithfulness, answer relevance.

Evidence-anchored: every score must carry a quote or 'none'; quoteless
high scores are marked suspect. Non-LLM guardrails ride along:
citation-regex presence and embedding cosine (answer vs retrieved).
Same local backend as generation (Ollama when OLLAMA_MODEL is set).
"""

from __future__ import annotations

import json
from pathlib import Path

from app.evaluation.triplets import judge_call
from app.rag.generation import CITATION_RE

RESULTS_DIR = Path("data/evals")

CTX_PROMPT = """You are a strict evaluator. Source passage and retrieved chunks below.

SOURCE:
{source}

RETRIEVED:
{retrieved}

QUESTION: {q}

Does the RETRIEVED text contain the facts needed to answer the QUESTION?
Reply STRICT JSON only: {{"score": <0.0-1.0>, "evidence": "<short supporting quote from RETRIEVED, or 'none'>"}}
Score 1 only if ALL needed facts are present. Be harsh."""

FAITH_PROMPT = """You are a strict evaluator. Retrieved context and system answer below.

RETRIEVED:
{retrieved}

ANSWER: {answer}

Is EVERY statement in the ANSWER fully supported by RETRIEVED?
Reply STRICT JSON only: {{"score": <0.0-1.0>, "evidence": "<unsupported quote from ANSWER, or 'none' if fully supported>"}}
Score 1 only if zero unsupported statements. Be harsh."""

REL_PROMPT = """You are a strict evaluator.

QUESTION: {q}

ANSWER: {answer}

Does the ANSWER directly address the QUESTION?
Reply STRICT JSON only: {{"score": <0.0-1.0>, "evidence": "<off-topic quote from ANSWER, or 'none'>"}}
Score 1 only if fully on-topic. Be harsh."""


def _parse_score(raw: str) -> tuple[float, str]:
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix(
        "```"
    ).strip()
    start, end = clean.find("{"), clean.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON: {raw[:200]}")
    obj = json.loads(clean[start : end + 1])
    try:
        score = max(0.0, min(1.0, float(obj.get("score", 0))))
    except (TypeError, ValueError):
        score = 0.0
    return score, str(obj.get("evidence", "none")).strip() or "none"


def _status(score: float, evidence: str) -> str:
    if score < 0.5:
        return "fail"
    if score <= 0.7 or evidence.lower() == "none":
        return "suspect"
    return "pass"


def _cosine_guard(answer: str, retrieved_texts: list[str]) -> float:
    """Embedding cosine(answer, mean retrieved). -1.0 when unavailable."""
    try:
        from app.rag.embeddings import embed_texts
    except Exception:
        return -1.0
    texts = [t for t in [answer, *retrieved_texts] if (t or "").strip()]
    if len(texts) < 2:
        return -1.0
    try:
        vecs = embed_texts(texts)
    except Exception:
        return -1.0
    ans, rest = vecs[0], vecs[1:]
    dim = len(ans)
    mean = [sum(v[i] for v in rest) / len(rest) for i in range(dim)]
    dot = sum(a * m for a, m in zip(ans, mean))
    na = sum(a * a for a in ans) ** 0.5
    nm = sum(m * m for m in mean) ** 0.5
    if na == 0 or nm == 0:
        return -1.0
    return round(dot / (na * nm), 3)


def score_triplet(
    question: str,
    source_text: str,
    retrieved_texts: list[str],
    answer_raw: str,
) -> dict:
    """Score one triplet on all three judges + guardrails."""
    retrieved_blob = "\n---\n".join(retrieved_texts)[:6000]
    ctx_raw = judge_call(
        CTX_PROMPT.format(
            source=(source_text or "")[:4000], retrieved=retrieved_blob, q=question
        ),
        temperature=0.0,
    )
    faith_raw = judge_call(
        FAITH_PROMPT.format(retrieved=retrieved_blob, answer=(answer_raw or "")[:3000]),
        temperature=0.0,
    )
    rel_raw = judge_call(
        REL_PROMPT.format(q=question, answer=(answer_raw or "")[:3000]),
        temperature=0.0,
    )
    ctx, ctx_ev = _parse_score(ctx_raw)
    faith, faith_ev = _parse_score(faith_raw)
    rel, rel_ev = _parse_score(rel_raw)
    return {
        "context": {"score": ctx, "evidence": ctx_ev, "status": _status(ctx, ctx_ev)},
        "faithful": {
            "score": faith,
            "evidence": faith_ev,
            "status": _status(faith, faith_ev),
        },
        "relevant": {"score": rel, "evidence": rel_ev, "status": _status(rel, rel_ev)},
        "has_citation": bool(CITATION_RE.search(answer_raw or "")),
        "answer_retrieved_cosine": _cosine_guard(answer_raw or "", retrieved_texts),
    }


def score_file(
    results_path: str | Path, triplets_path: str | Path
) -> Path:
    """Score every triplet at top_k=5; write scored_results_{sha12}.json."""
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))
    triplets = json.loads(Path(triplets_path).read_text(encoding="utf-8"))
    sources = {i: t.get("source_text", "") for i, t in enumerate(triplets)}
    for i, res in enumerate(results):
        per_k = res.get("per_k", {})
        k5 = per_k.get("5", {})
        # Recover retrieved texts from stored ids via Chroma is overkill;
        # use answer + question only if ids missing. Prefer live re-retrieve:
        from app.rag.retrieval import retrieve

        retrieved = retrieve(res.get("q", ""), top_k=5)
        texts = [r.get("text", "") for r in retrieved]
        res["judges"] = score_triplet(
            res.get("q", ""), sources.get(i, ""), texts, k5.get("answer_raw", "")
        )
    stem = Path(results_path).stem.replace("results_", "scored_results_")
    out = RESULTS_DIR / f"{stem}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return out

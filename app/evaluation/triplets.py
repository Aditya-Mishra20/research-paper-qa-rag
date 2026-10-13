"""
Synthetic triplet generation for per-PDF evaluation.

Builds cached (Q, GT, Source) triplets from the stratified parent pool.
Uses the same Gemini REST endpoint as generation but with temperature 0.0
and 429 retry/backoff. Cache key is the file sha256.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from langchain_core.documents import Document

from app.rag.generation import GenerationError, generate_text

EVAL_DIR = Path("data/evals")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

SYNTH_PROMPT = """You are an expert AI evaluator. Based STRICTLY on the text snippet below from a research paper, generate ONE highly specific question a user might ask, plus the ideal ground-truth answer derived directly from the text.
Rules: question cannot be answered without this context. No outside knowledge. Answer <= 3 sentences.

TEXT:
{chunk}

Return STRICT JSON only, no markdown fences:
{{"q": "...", "gt": "..."}}"""


def judge_call(prompt: str, temperature: float = 0.0, retries: int = 3) -> str:
    """Call the configured backend (Ollama or Gemini) with retry/backoff."""
    last: Exception | None = None
    for _ in range(retries):
        try:
            return generate_text(prompt, temperature=temperature)
        except GenerationError as exc:
            last = exc
            if "429" in str(exc) or "quota" in str(exc).lower():
                time.sleep(60)
                continue
            if (
                "503" in str(exc)
                or "UNAVAILABLE" in str(exc)
                or "500" in str(exc)
                or "Ollama transport" in str(exc)
            ):
                time.sleep(15)
                continue
            raise
    raise GenerationError(f"synth failed after {retries}: {last}")


def parse_pair(raw: str) -> tuple[str, str]:
    """Extract (q, gt) from strict-JSON LLM output, tolerating fences."""
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix(
        "```"
    ).strip()
    start, end = clean.find("{"), clean.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object: {raw[:200]}")
    obj = json.loads(clean[start : end + 1])
    question, gt = str(obj.get("q", "")).strip(), str(obj.get("gt", "")).strip()
    if not question or not gt:
        raise ValueError(f"empty q/gt: {raw[:200]}")
    return question, gt


def build_triplets(
    pool: list[Document], sha256: str, doc_id: str
) -> Path:
    """Generate and cache triplets for one document; reuse on re-upload."""
    out = EVAL_DIR / f"triplets_{sha256[:12]}.json"
    if out.exists():
        return out
    triplets: list[dict] = []
    for doc in pool:
        metadata = doc.metadata or {}
        raw = judge_call(
            SYNTH_PROMPT.format(chunk=(doc.page_content or "")[:4000]),
            temperature=0.0,
        )
        try:
            question, gt = parse_pair(raw)
        except Exception:
            continue
        triplets.append(
            {
                "q": question,
                "gt": gt,
                "source_text": doc.page_content,
                "page": metadata.get("page"),
                "section": metadata.get("section"),
                "doc_id": doc_id,
                "sha256": sha256,
            }
        )
    out.write_text(
        json.dumps(triplets, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out

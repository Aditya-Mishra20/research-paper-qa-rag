"""
Cited answer generation with Gemini.

Uses the REST endpoint directly with a fresh httpx transport per call
instead of the SDK's shared client, which can be left in a closed state
by reloader-recycled workers.
"""

from __future__ import annotations

import os
import re
from typing import Any

import httpx


GENERATION_MODEL = "gemini-2.5-flash-lite"
GENERATE_URL = (
    "https://generativelanguage.googleapis.com"
    f"/v1beta/models/{GENERATION_MODEL}:generateContent"
)
CITATION_RE = re.compile(r"\[[^\[\]]+?[,]?\s*p\.\d+\]")


class GenerationError(RuntimeError):
    """Raised when answer generation cannot complete."""


def _api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise GenerationError("GEMINI_API_KEY is required for generation.")
    return api_key


def _generate_text(prompt: str) -> str:
    """Call generateContent with a fresh transport; never reuse a client."""
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                GENERATE_URL,
                headers={
                    "x-goog-api-key": _api_key(),
                    "Content-Type": "application/json",
                },
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.1},
                },
            )
    except GenerationError:
        raise
    except Exception as exc:
        raise GenerationError(f"Generation transport error: {exc}") from exc
    if response.status_code == 429:
        raise GenerationError("Free quota exceeded, retry in 60s.")
    if response.status_code >= 400:
        try:
            detail: Any = response.json()
        except Exception:
            detail = response.text
        raise GenerationError(f"Gemini API error {response.status_code}: {detail}")
    try:
        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as exc:
        raise GenerationError(
            f"Unexpected Gemini response: {response.text[:300]}"
        ) from exc
    return (text or "").strip()


def _context_block(contexts: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, item in enumerate(contexts, start=1):
        metadata = item.get("metadata") or {}
        title = metadata.get("title") or metadata.get("source_filename") or "Untitled"
        page = metadata.get("page") or "?"
        section = metadata.get("section") or "Unknown"
        text = item.get("text") or ""
        blocks.append(f"[{index}][{title}, p.{page} | {section}]\n{text}")
    return "\n\n".join(blocks)


def _drop_uncited_sentences(answer: str) -> str:
    if "Not found in uploaded papers." in answer:
        return "Not found in uploaded papers."

    pieces = re.split(r"(?<=[.!?])\s+", answer.strip())
    kept = [piece.strip() for piece in pieces if CITATION_RE.search(piece)]
    if kept:
        return " ".join(kept)
    # Fallback: the model answered but under-cited. Keep the raw answer with
    # sources attached instead of falsely reporting "Not found".
    return answer.strip() if answer.strip() else "Not found in uploaded papers."


def generate_answer(question: str, contexts: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate an answer where factual sentences must carry page citations."""
    if not contexts:
        return {"answer": "Not found in uploaded papers.", "sources": []}

    prompt = f"""System: Use ONLY the CONTEXT below. Every factual sentence must end with a citation copied exactly from a CONTEXT header, e.g. [2505.10468v4 Ai agents and agentic ai, p.6]. Copy titles exactly as shown. If the answer is not present, say "Not found in uploaded papers."

CONTEXT:
{_context_block(contexts)}

QUESTION:
{question}

ANSWER:"""

    answer = _generate_text(prompt)
    answer = _drop_uncited_sentences(answer)
    sources = []
    seen = set()
    for item in contexts:
        citation = item.get("citation")
        if citation and citation not in seen:
            sources.append(
                {
                    "citation": citation,
                    "score": item.get("score"),
                    "metadata": item.get("metadata", {}),
                    "text": item.get("text", ""),
                }
            )
            seen.add(citation)
    return {"answer": answer, "sources": sources}

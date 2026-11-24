"""Optional LLM reranker over the fused candidates (listwise, one model call).

Ollama has no rerank endpoint and a cross-encoder would mean adding PyTorch,
so this asks the local chat model to order the top candidates by usefulness.
Failures (bad output, Ollama down) keep the original order.
"""

from __future__ import annotations

import re

import httpx

from .answer import default_model, default_ollama_url
from .search import Hit

RERANK_N = 20  # candidates shown to the model
PASSAGE_WORDS = 70

PROMPT = """Rank news passages by how useful they are for answering a question.

Question: {question}

Passages:
{passages}

Reply with the passage numbers that help answer the question, most useful first, separated by commas.
Leave out passages that don't help. Reply with numbers only."""


class RerankError(RuntimeError):
    pass


def _passage(i: int, h: Hit) -> str:
    words = h.body.split()
    text = " ".join(words[:PASSAGE_WORDS]) + (" …" if len(words) > PASSAGE_WORDS else "")
    when = (h.published_at or "")[:10]
    return f"[{i}] {h.title} ({h.source}, {when}): {text}"


def parse_order(text: str, n: int) -> list[int]:
    """0-based indices from the model's reply, deduped, in range."""
    order: list[int] = []
    for tok in re.findall(r"\d+", text):
        i = int(tok) - 1
        if 0 <= i < n and i not in order:
            order.append(i)
    return order


def apply_order(hits: list[Hit], order: list[int]) -> list[Hit]:
    """Model-ranked hits first, then everything it left out, in the original order."""
    chosen = set(order)
    return [hits[i] for i in order] + [h for i, h in enumerate(hits) if i not in chosen]


class LLMReranker:
    def __init__(self, model: str | None = None, base_url: str | None = None, n: int = RERANK_N, timeout: float = 120.0):
        self.model = model or default_model()
        self.base_url = (base_url or default_ollama_url()).rstrip("/")
        self.n = n
        self.timeout = timeout

    def ask(self, prompt: str) -> str:
        try:
            resp = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model, "stream": False,
                    "messages": [{"role": "user", "content": prompt}],
                    "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 80},
                },
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise RerankError(f"can't reach Ollama: {type(exc).__name__}") from exc
        if resp.status_code != 200:
            raise RerankError(f"Ollama returned {resp.status_code}")
        return resp.json().get("message", {}).get("content", "")

    def __call__(self, question: str, hits: list[Hit]) -> list[Hit]:
        head, tail = hits[: self.n], hits[self.n :]
        if len(head) < 2:
            return hits
        passages = "\n".join(_passage(i, h) for i, h in enumerate(head, 1))
        try:
            order = parse_order(self.ask(PROMPT.format(question=question, passages=passages)), len(head))
        except RerankError:
            return hits
        return apply_order(head, order) + tail

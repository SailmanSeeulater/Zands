"""Build a grounded prompt from retrieved chunks and ask a local Ollama model."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import date

import httpx

from .search import Hit

DEFAULT_MODEL = "qwen2.5:latest"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_NUM_CTX = 16384  # Ollama's default context is too small for 8 chunks


def default_model() -> str:
    return os.environ.get("ZANDS_MODEL", DEFAULT_MODEL)


def default_ollama_url() -> str:
    host = os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_URL)
    if not host.startswith("http"):
        host = "http://" + host
    return host.rstrip("/")


SYSTEM_TEMPLATE = """You are a careful news assistant. Today's date is {today}.

Answer the user's question using ONLY the numbered news sources provided in the message.
- Cite every factual claim with the source number in square brackets, like [1] or [2][3].
- If the sources do not cover the question, or only partly cover it, say so plainly. Do not fill gaps from memory.
- Your own training data is older than these sources; when they conflict, trust the sources.
- Mention dates when timing matters, relative to today's date.
- Be concise."""


def format_sources(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        when = h.published_at[:10] if h.published_at else "date unknown"
        blocks.append(f"[{i}] {h.title}\nSource: {h.source} | Published: {when} | {h.url}\n\n{h.body}")
    return "\n\n---\n\n".join(blocks)


def build_messages(question: str, hits: list[Hit], today: date | None = None) -> list[dict]:
    today = today or date.today()
    user = f"Sources:\n\n{format_sources(hits)}\n\n===\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_TEMPLATE.format(today=today.strftime("%A, %B %d, %Y").replace(" 0", " "))},
        {"role": "user", "content": user},
    ]


class OllamaError(RuntimeError):
    pass


def stream_chat(
    messages: list[dict],
    model: str | None = None,
    base_url: str | None = None,
    num_ctx: int = DEFAULT_NUM_CTX,
    temperature: float = 0.2,
    timeout: float = 300.0,
) -> Iterator[str]:
    """Yield the model's reply in pieces as Ollama streams it."""
    model = model or default_model()
    base_url = base_url or default_ollama_url()
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"num_ctx": num_ctx, "temperature": temperature},
    }
    try:
        with httpx.stream("POST", f"{base_url}/api/chat", json=payload, timeout=timeout) as resp:
            if resp.status_code != 200:
                resp.read()
                raise OllamaError(f"Ollama returned {resp.status_code}: {resp.text[:300]}")
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if "error" in data:
                    raise OllamaError(data["error"])
                piece = data.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if data.get("done"):
                    break
    except httpx.ConnectError as exc:
        raise OllamaError(f"can't reach Ollama at {base_url} (is `ollama serve` running?)") from exc

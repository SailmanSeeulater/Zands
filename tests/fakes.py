"""Offline stand-ins for Ollama."""

import hashlib
import math
import re

from zands import store
from zands.embed import DOC_PREFIX, QUERY_PREFIX, EmbedError

# Words the fake model treats as meaning the same thing, so tests can show
# vector search finding chunks that share no keywords with the question.
CONCEPTS = {
    "magma": "volcano", "lava": "volcano", "eruption": "volcano", "erupted": "volcano", "volcano": "volcano",
    "chips": "semiconductor", "chipmakers": "semiconductor", "semiconductor": "semiconductor",
    "borrowing": "rates", "rates": "rates", "interest": "rates",
}


class FakeEmbedder:
    """Deterministic bag-of-concepts embeddings, 768-d, unit length."""

    model = "fake-embed"

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        for prefix in (DOC_PREFIX, QUERY_PREFIX):
            if text.startswith(prefix):
                text = text[len(prefix):]
        v = [0.0] * store.EMBED_DIM
        for word in re.findall(r"[a-z]+", text.lower()):
            word = CONCEPTS.get(word, word)
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % store.EMBED_DIM
            v[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]


class DownEmbedder:
    """Behaves like Ollama being unreachable."""

    model = "fake-embed"

    def embed(self, texts):
        raise EmbedError("can't reach Ollama at http://localhost:11434: ConnectError")

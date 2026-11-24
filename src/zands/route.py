"""Query router: decide where an answer should come from before (and right after) retrieval.

Routes:
  news       search the local news index (the default when unsure)
  knowledge  timeless question; the model answers from its own knowledge
  web        needs live data or something the feeds won't carry; use the SearXNG web fallback
  digest     "what's in the news today?": answer from today's top-stories digest
  none       Zands can't help (the user's own private data, requests for advice or predictions)

Fact overrides aren't a separate route: relevant facts are added to every answer.

Step 1 is cheap regex rules (no model call). Step 2, only when no rule
matches, asks a small local model for one word. Step 3 (`gate`) runs after
retrieval: a "knowledge" question with a strong local news match is upgraded
to news, and a "news" question with a weak local match also gets web results.
Every decision carries a human-readable reason that is stored in retrieval_log.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import httpx

from .answer import default_ollama_url

ROUTES = ("news", "knowledge", "web", "digest", "none")
DEFAULT_ROUTER_MODEL = "qwen2.5:latest"  # 24/24 on eval/routes.toml vs 19/24 for qwen3.5:0.8b, 0.14 s/call

# Similarity of the best vector match (1 - cosine distance) at or above which the local index
# is considered to cover a question. Calibrated on 2026-11-24 with nomic-embed-text: 20 covered
# questions scored 0.710-0.879, 20 uncovered ones 0.534-0.689.
STRONG_MATCH = 0.70
# When web results are added because the local match is weak, local hits below this are dropped
# as noise (every uncovered question's best match was <= 0.689; covered ones never get here).
NEAR_MATCH = 0.65

# A topic after the headlines phrase ("latest news on Nvidia") makes it a news search, not a digest.
TOPIC_AFTER = re.compile(
    r"\s+(about|on|regarding|concerning|from|in|for|with|of)\s+(?!(the |today'?s )?(news|headlines|stories|day)\b)\w", re.I
)

# "What's in the news?" questions, answered from the digest.
DIGEST = re.compile(
    r"\b((top|main|biggest|major|latest|today'?s|the day'?s) (news )?(stories|headlines|news)|"
    r"what('?s| is) (in the news|happening in the world|the news)|"
    r"(news|headlines) (today|this morning|right now)|catch me up|news roundup|news summary)\b",
    re.I,
)

# Live values the news feeds don't carry.
LIVE = re.compile(
    r"\b(weather|forecast|temperature (in|today|tomorrow)|exchange rate|"
    r"(current|live|latest) (price|stock price|share price|score)|"
    r"price of .+ (today|now|right now)|what('s| is) the price of|how much is .+ worth (today|now))\b",
    re.I,
)

# Signals that the answer depends on recent events.
FRESH = re.compile(
    r"\b(today|tonight|yesterday|tomorrow|this (morning|afternoon|evening|week|weekend|month|year)|"
    r"last (night|week|weekend|month)|latest|recent|recently|right now|currently|current|so far|"
    r"ongoing|breaking|news|headlines?|updates?|still|just|anymore|who won|what happened|happening|"
    r"20[2-9][0-9])\b",
    re.I,
)

# Timeless phrasings. Only consulted when no FRESH signal matched.
TIMELESS = re.compile(
    r"^\s*(?:"
    r"what (?:is|are) (?:a|an)\b|what is the meaning of|what does .+ mean|define\b|definition of|"
    r"explain (?:how|why|what|the concept|the difference)|explain \w+(?: \w+)?$|"
    r"how (?:does|do) (?:a|an|the)?\s*\w+(?: \w+){0,2} work|how to\b|why (?:is|are|does|do) (?:the sky|water|the sun|ice|planes)\b|"
    r"who (?:wrote|invented|discovered|painted|composed)\b|"
    r"what is the (?:capital|population|square root|formula|boiling point|speed of light|chemical symbol)\b|"
    r"how (?:tall|high|deep|long|old|far|big) is (?:the )?(?:mount|mt\.?|planet|sun|moon|earth|pacific|atlantic|river|nile|amazon river)\b|"
    r"translate\b|convert \d"
    r")",
    re.I,
)

# The user's own private data: not something news or the web can answer.
PERSONAL = re.compile(
    r"\b(my|our) (calendar|schedule|inbox|emails?|bank|account|balance|password|files?|meetings?|appointments?)\b",
    re.I,
)


@dataclass
class RouteDecision:
    route: str
    reason: str
    method: str  # 'rule' | 'model' | 'default' | 'forced'

    def with_gate(self, route: str, note: str) -> "RouteDecision":
        return RouteDecision(route=route, reason=f"{self.reason}; {note}", method=self.method)


def default_router_model() -> str:
    return os.environ.get("ZANDS_ROUTER_MODEL", DEFAULT_ROUTER_MODEL)


def rule_route(question: str) -> RouteDecision | None:
    """Deterministic first pass. None means no rule matched."""
    q = question.strip()
    if m := PERSONAL.search(q):
        return RouteDecision("none", f"rule: personal data ({m.group(0)!r})", "rule")
    if (m := DIGEST.search(q)) and not TOPIC_AFTER.match(q[m.end():]):
        return RouteDecision("digest", f"rule: headlines question ({m.group(0)!r})", "rule")
    if m := LIVE.search(q):
        return RouteDecision("web", f"rule: live data ({m.group(0)!r})", "rule")
    if m := FRESH.search(q):
        return RouteDecision("news", f"rule: time-sensitive ({m.group(0)!r})", "rule")
    if m := TIMELESS.search(q):
        return RouteDecision("knowledge", f"rule: timeless phrasing ({m.group(0).strip()!r})", "rule")
    return None


CLASSIFIER_PROMPT = """Decide where the answer to a question should come from. Reply with exactly one word:
news - about current events, people or organizations in the news, or anything that may have changed in the last two years
knowledge - timeless facts, definitions, science, how things work, history from before 2024
web - needs live data (weather, prices, live scores) or a niche fact a news site would not cover
none - about the user's own private data, or asks for personal advice or a prediction

Question: Who is running Nvidia?
Answer: news
Question: What is photosynthesis?
Answer: knowledge
Question: What's the cheapest flight to Lisbon next week?
Answer: web
Question: Should I quit my job?
Answer: none
Question: What did the EU decide about migrants?
Answer: news
Question: Who was the first person on the moon?
Answer: knowledge
Question: {question}
Answer:"""


class RouterError(RuntimeError):
    pass


def model_route(question: str, model: str | None = None, base_url: str | None = None, timeout: float = 30.0) -> RouteDecision:
    """Ask a small local model for one route word. Raises RouterError on failure."""
    model = model or default_router_model()
    base_url = (base_url or default_ollama_url()).rstrip("/")
    try:
        resp = httpx.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": CLASSIFIER_PROMPT.format(question=question.strip())}],
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_predict": 5},
            },
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        raise RouterError(f"can't reach Ollama: {type(exc).__name__}") from exc
    if resp.status_code != 200:
        raise RouterError(f"Ollama returned {resp.status_code}: {resp.text[:200]}")
    text = resp.json().get("message", {}).get("content", "")
    word = re.findall(r"[a-z]+", text.lower())
    if not word or word[0] not in ROUTES:
        raise RouterError(f"unexpected classifier output {text[:40]!r}")
    return RouteDecision(word[0], f"model {model}: {word[0]}", "model")


def route(question: str, use_model: bool = True, model: str | None = None, base_url: str | None = None, classifier=None) -> RouteDecision:
    """Rules first, then (optionally) the small model, then the news default."""
    decision = rule_route(question)
    if decision:
        return decision
    if use_model:
        try:
            return (classifier or model_route)(question, model=model, base_url=base_url)
        except RouterError as exc:
            return RouteDecision("news", f"default (classifier failed: {exc})", "default")
    return RouteDecision("news", "default (no rule matched)", "default")


def gate(decision: RouteDecision, top_similarity: float | None, has_hits: bool, web_available: bool) -> RouteDecision:
    """Adjust the route once local retrieval has run.

    - knowledge + strong local news match -> news (the news probably matters)
    - news + weak local match -> news+web when SearXNG is reachable (local hits are kept)
    - web without SearXNG -> news (best effort from the local index)
    """
    strong = (top_similarity >= STRONG_MATCH) if top_similarity is not None else has_hits
    sim = f"{top_similarity:.2f}" if top_similarity is not None else ("keyword hits" if has_hits else "no hits")
    if decision.route == "knowledge" and strong:
        return decision.with_gate("news", f"gate: strong local match ({sim})")
    if decision.route == "news" and not strong:
        if web_available:
            return decision.with_gate("news+web", f"gate: weak local match ({sim}), adding web results")
        return decision.with_gate("news", f"gate: weak local match ({sim}), web fallback unavailable")
    if decision.route == "web" and not web_available:
        return decision.with_gate("news", "gate: web fallback unavailable, using local news")
    return decision

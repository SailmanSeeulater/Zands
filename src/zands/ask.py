"""Plan an answer: route -> retrieve -> gate -> facts, digest, web -> prompt.

`plan()` does everything except calling the answer model, so it is testable
offline and reusable (CLI, eval). The decision trail is written to the
question's retrieval_log row (route, route_reason, fact_ids, web_urls).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import digest as digestlib
from . import facts as factlib
from . import route as routelib
from . import store
from . import web as weblib
from .answer import build_messages
from .search import DEFAULT_K, DEFAULT_MODE, RECENCY_WEIGHT, Hit, Retrieval, retrieve

NO_DIGEST_REPLY = "There is no digest of today's stories yet. Run `zands ingest` (or `zands digest --refresh`)."

NONE_REPLY = (
    "Zands can't answer that: it's about your own private data, or asks for advice or a prediction, "
    "and the news index can't help with either."
)
EMPTY_REPLY = (
    "No articles or facts in the local database match that question, so there is nothing to answer from.\n"
    "Try different keywords, or run `zands ingest` to fetch fresh news."
)


@dataclass
class AskPlan:
    question: str
    decision: routelib.RouteDecision
    retrieval: Retrieval | None = None
    facts: list[factlib.Fact] = field(default_factory=list)
    web: weblib.WebContext | None = None
    digest: str | None = None
    messages: list[dict] | None = None  # None -> don't call the model; print `reply`
    reply: str | None = None

    @property
    def route(self) -> str:
        return self.decision.route

    @property
    def hits(self) -> list[Hit]:
        if self.route in ("knowledge", "digest") or self.retrieval is None:
            return []
        return self.retrieval.hits

    @property
    def web_hits(self) -> list[Hit]:
        return self.web.hits if self.web else []


def plan(
    conn: store.Connection,
    question: str,
    k: int = DEFAULT_K,
    mode: str = DEFAULT_MODE,
    embedder=None,
    recency_weight: float = RECENCY_WEIGHT,
    use_router: bool = True,
    forced_route: str | None = None,
    router_model: str | None = None,
    ollama_url: str | None = None,
    web_enabled: bool = True,
    web_save: bool = False,
    use_facts: bool = True,
    use_digest: bool = True,
    classifier=None,  # test seam: replaces routelib.model_route
    web_available=None,  # test seam: () -> bool
    web_fn=None,  # test seam: replaces weblib.web_context
) -> AskPlan:
    # 1. route (before retrieval)
    if forced_route:
        decision = routelib.RouteDecision(forced_route, f"forced: {forced_route}", "forced")
    else:
        decision = routelib.route(question, use_model=use_router, model=router_model, base_url=ollama_url, classifier=classifier)

    if decision.route == "none":
        log_id = store.log_retrieval(conn, question, None, [], k, command="ask", mode=None)
        store.set_log_route(conn, log_id, "none", decision.reason)
        return AskPlan(question, decision, reply=NONE_REPLY)

    if decision.route == "digest":
        row = digestlib.latest(conn)
        log_id = store.log_retrieval(conn, question, None, [], k, command="ask", mode=None)
        p = AskPlan(question, decision, digest=row["text"] if row else None)
        p.facts = factlib.relevant_facts(conn, question) if use_facts else []
        if p.digest:
            p.messages = build_messages(question, [], facts=p.facts, digest=p.digest)
        else:
            p.reply = NO_DIGEST_REPLY
        store.set_log_route(conn, log_id, "digest", decision.reason, fact_ids=[f.id for f in p.facts])
        return p

    # 2. retrieve locally (cheap; also lets the gate catch misrouted questions)
    r = retrieve(conn, question, k=k, mode=mode, embedder=embedder, recency_weight=recency_weight, command="ask")

    # 3. gate: only probe SearXNG when the decision could use it
    strong = (r.top_similarity >= routelib.STRONG_MATCH) if r.top_similarity is not None else bool(r.hits)
    wants_web = web_enabled and (decision.route == "web" or (decision.route == "news" and not strong))
    web_ok = wants_web and (web_available or weblib.available)()
    decision = routelib.gate(decision, r.top_similarity, bool(r.hits), web_ok)

    p = AskPlan(question, decision, retrieval=r)
    p.facts = factlib.relevant_facts(conn, question) if use_facts else []
    if use_digest:
        row = digestlib.latest(conn)
        p.digest = row["text"] if row else None

    # 4. web results
    if decision.route in ("web", "news+web"):
        p.web = (web_fn or weblib.web_context)(question, embedder=embedder, conn=conn, save=web_save)
        if p.web.error and not p.web.hits:
            p.decision = decision.with_gate("news", f"web fallback failed ({p.web.error})")

    # With web results standing in for a weak local match, drop local hits that are just noise.
    if p.route == "web" and not strong:
        r.hits = []
    elif p.route == "news+web" and p.web_hits:
        r.hits = [h for h in r.hits if h.distance is not None and 1.0 - h.distance >= routelib.NEAR_MATCH]

    # 5. prompt (or a canned reply)
    knowledge = p.route == "knowledge"
    if not knowledge and not (p.hits or p.web_hits or p.facts):
        p.reply = EMPTY_REPLY
    else:
        p.messages = build_messages(
            question, p.hits, facts=p.facts, web_hits=p.web_hits, digest=p.digest, knowledge=knowledge
        )
    store.set_log_route(
        conn, r.log_id, p.route, p.decision.reason,
        fact_ids=[f.id for f in p.facts], web_urls=p.web.urls if p.web else None,
    )
    return p

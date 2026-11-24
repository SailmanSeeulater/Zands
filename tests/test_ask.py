"""The ask planner: route -> retrieve -> gate -> facts/digest/web -> prompt. All offline."""

import json
from datetime import datetime, timezone

import pytest

from zands import ask, digest, facts, route, store, web
from zands.embed import embed_pending
from zands.search import Hit

from fakes import FakeEmbedder

DOCS = [
    ("https://n.example/volcano", "Iceland evacuates town", "Lava flowed toward Grindavik after the eruption."),
    ("https://n.example/ecb", "European Central Bank holds", "The bank kept interest rates unchanged."),
]


@pytest.fixture
def db(conn):
    for url, title, body in DOCS:
        store.insert_article(conn, store.Article(url=url, title=title, source="N", body=body, extraction="rss",
                                                 published_at="2026-11-24T09:00:00Z"), [body])
    embed_pending(conn, FakeEmbedder())
    return conn


def classify(route_name):
    return lambda q, model=None, base_url=None: route.RouteDecision(route_name, f"model fake: {route_name}", "model")


def fake_web(hits=None, error=None):
    calls = []

    def fn(question, embedder=None, conn=None, save=False):
        calls.append(question)
        h = hits if hits is not None else [Hit(chunk_id=-1, article_id=-1, title="Web page", source="web: w.example",
                                               url="https://w.example/p", published_at=None, body="web text", snippet="", score=1.0)]
        return web.WebContext(hits=[] if error else h, results=[], error=error)

    fn.calls = calls
    return fn


def run(db, question, **kw):
    kw.setdefault("embedder", FakeEmbedder())
    kw.setdefault("web_available", lambda: True)
    return ask.plan(db, question, **kw)


def last_log(db):
    return db.execute("SELECT * FROM retrieval_log ORDER BY id DESC LIMIT 1").fetchone()


def test_none_route_answers_without_model(db):
    p = run(db, "What's on my calendar tomorrow?")
    assert p.route == "none" and p.messages is None and "private data" in p.reply
    row = last_log(db)
    assert row["route"] == "none" and "personal data" in row["route_reason"]


def test_strong_news_match(db, monkeypatch):
    monkeypatch.setattr(route, "STRONG_MATCH", 0.0)
    wf = fake_web()
    p = run(db, "Iceland lava eruption", classifier=classify("news"), web_fn=wf)
    assert p.route == "news" and p.hits and p.hits[0].url == "https://n.example/volcano"
    assert wf.calls == []  # no web search when the local match is strong
    assert "News sources:" in p.messages[1]["content"]
    row = last_log(db)
    assert row["route"] == "news" and row["mode"] == "hybrid" and row["web_urls"] is None


def test_weak_news_match_adds_web_and_drops_noise(db, monkeypatch):
    monkeypatch.setattr(route, "STRONG_MATCH", 1.01)
    monkeypatch.setattr(route, "NEAR_MATCH", 1.01)
    p = run(db, "Who won the Tour de France?", web_fn=fake_web())
    assert p.route == "news+web" and "weak local match" in p.decision.reason
    assert p.web_hits and p.hits == []  # local hits below NEAR_MATCH are noise
    user = p.messages[1]["content"]
    assert "WEB RESULTS" in user and "[W1] Web page" in user and "News sources" not in user
    assert json.loads(last_log(db)["web_urls"]) == ["https://w.example/p"]


def test_weak_match_without_searxng_keeps_local(db, monkeypatch):
    monkeypatch.setattr(route, "STRONG_MATCH", 1.01)
    wf = fake_web()
    p = run(db, "Who won the Tour de France?", web_available=lambda: False, web_fn=wf)
    assert p.route == "news" and "web fallback unavailable" in p.decision.reason and wf.calls == []


def test_web_disabled_never_probes(db, monkeypatch):
    monkeypatch.setattr(route, "STRONG_MATCH", 1.01)

    def boom():
        raise AssertionError("should not probe SearXNG")

    assert run(db, "Who won the Tour de France?", web_enabled=False, web_available=boom).route == "news"


def test_web_route(db, monkeypatch):
    monkeypatch.setattr(route, "STRONG_MATCH", 1.01)
    p = run(db, "What's the weather in Tokyo tomorrow?", web_fn=fake_web())
    assert p.route == "web" and p.hits == [] and p.web_hits


def test_web_failure_falls_back_to_news(db, monkeypatch):
    monkeypatch.setattr(route, "STRONG_MATCH", 1.01)
    p = run(db, "What's the weather in Tokyo tomorrow?", web_fn=fake_web(error="SearXNG returned no results"))
    assert p.route == "news" and "web fallback failed" in p.decision.reason


def test_knowledge_route_uses_general_prompt(db, monkeypatch):
    monkeypatch.setattr(route, "STRONG_MATCH", 1.01)
    p = run(db, "What is a volcano?")
    assert p.route == "knowledge" and p.hits == []
    system, user = (m["content"] for m in p.messages)
    assert "general knowledge" in system and "News sources" not in user


def test_knowledge_with_strong_local_match_becomes_news(db, monkeypatch):
    monkeypatch.setattr(route, "STRONG_MATCH", 0.0)
    p = run(db, "What is a volcano?")
    assert p.route == "news" and "strong local match" in p.decision.reason and p.hits


def test_forced_route(db, monkeypatch):
    monkeypatch.setattr(route, "STRONG_MATCH", 0.0)
    p = run(db, "What is a volcano?", forced_route="news")
    assert p.decision.method == "forced" and p.route == "news"


def test_digest_route(db):
    p = run(db, "What are today's top stories?")
    assert p.route == "digest" and p.messages is None and "no digest" in p.reply
    digest.refresh(db, now=datetime.now(timezone.utc))
    p = run(db, "What are today's top stories?")
    assert p.route == "digest" and p.hits == []
    system, user = (m["content"] for m in p.messages)
    assert "TODAY'S HEADLINES:" in user and "[D1]" in user and "headlines" in system


def test_facts_and_digest_ride_along(db, monkeypatch):
    monkeypatch.setattr(route, "STRONG_MATCH", 0.0)
    facts.add_fact(db, "European Central Bank", "main rate", "2.0%", valid_from="2026-11-01")
    digest.refresh(db, now=datetime.now(timezone.utc))
    p = run(db, "What did the European Central Bank do with rates?", classifier=classify("news"))
    assert [f.value for f in p.facts] == ["2.0%"]
    user = p.messages[1]["content"]
    assert user.index("VERIFIED FACTS") < user.index("TODAY'S HEADLINES") < user.index("News sources")
    assert json.loads(last_log(db)["fact_ids"]) == [p.facts[0].id]
    p = run(db, "What did the European Central Bank do with rates?", classifier=classify("news"),
            use_facts=False, use_digest=False)
    assert p.facts == [] and p.digest is None


def test_nothing_to_answer_from(db, monkeypatch):
    monkeypatch.setattr(route, "STRONG_MATCH", 1.01)
    p = run(db, "zzyzx qwerty", classifier=classify("news"), web_available=lambda: False, mode="keyword", embedder=None)
    assert p.messages is None and "nothing to answer from" in p.reply

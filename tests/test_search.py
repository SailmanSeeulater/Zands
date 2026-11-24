import json
from datetime import datetime, timezone

import pytest

from zands import store
from zands.search import recency_factor, search

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)

FIXTURE = [
    # (url, title, published_at, chunks)
    (
        "https://news.example/ecb",
        "European Central Bank holds interest rates",
        "2026-11-24T09:00:00Z",
        ["The European Central Bank kept interest rates unchanged on Tuesday, citing slowing inflation."],
    ),
    (
        "https://news.example/fed-old",
        "Federal Reserve cuts interest rates",
        "2026-10-01T09:00:00Z",
        ["The Federal Reserve cut interest rates by a quarter point in October."],
    ),
    (
        "https://news.example/volcano",
        "Volcano erupts in Iceland",
        "2026-11-23T09:00:00Z",
        ["A volcano erupted on the Reykjanes peninsula, forcing evacuations near Grindavik."],
    ),
    (
        "https://news.example/long",
        "Chip export rules tightened",
        "2026-11-22T09:00:00Z",
        [
            "New semiconductor export rules were announced.",
            "The semiconductor rules cover advanced chips.",
            "Analysts said semiconductor makers will adapt.",
        ],
    ),
]


@pytest.fixture
def db(conn):
    for url, title, published, chunks in FIXTURE:
        art = store.Article(
            url=url, title=title, source="Example News", body="\n\n".join(chunks),
            extraction="trafilatura", published_at=published,
        )
        store.insert_article(conn, art, chunks)
    return conn


def test_finds_relevant_chunk(db):
    hits = search(db, "Is there a volcano eruption in Iceland?", now=NOW)
    assert hits[0].title == "Volcano erupts in Iceland"
    assert hits[0].published_at == "2026-11-23T09:00:00Z"
    assert hits[0].source == "Example News"
    assert "[" in hits[0].snippet  # match highlighted


def test_stemming_matches_inflections(db):
    hits = search(db, "erupting volcanoes", now=NOW)
    assert hits and hits[0].url == "https://news.example/volcano"


def test_no_match_and_no_terms(db):
    assert search(db, "cricket world cup", now=NOW) == []
    assert search(db, "what is the latest?", now=NOW) == []


def test_k_limits_results(db):
    assert len(search(db, "interest rates semiconductor volcano", k=2, now=NOW)) == 2


def test_recency_boost_prefers_newer(db):
    # Both rate stories match "interest rates"; the November one should win with recency on.
    hits = search(db, "interest rates", now=NOW, recency_weight=0.3)
    assert hits[0].url == "https://news.example/ecb"
    assert recency_factor(0) == pytest.approx(1.3)
    assert recency_factor(3) == pytest.approx(1.15)
    assert recency_factor(None) == 1.0
    assert recency_factor(5, weight=0) == 1.0


def test_max_per_article(db):
    hits = search(db, "semiconductor", now=NOW, max_per_article=2)
    assert len(hits) == 2
    hits = search(db, "semiconductor", now=NOW, max_per_article=0)
    assert len(hits) == 3


def test_every_retrieval_is_logged(db):
    hits = search(db, "volcano Iceland", now=NOW, command="test")
    search(db, "nothing matches xyzzy", now=NOW)
    search(db, "what?", now=NOW)
    rows = db.execute("SELECT * FROM retrieval_log ORDER BY id").fetchall()
    assert len(rows) == 3
    assert rows[0]["question"] == "volcano Iceland"
    assert rows[0]["query"] == '"volcano" OR "iceland"'
    assert rows[0]["command"] == "test"
    assert json.loads(rows[0]["chunk_ids"]) == [h.chunk_id for h in hits]
    assert json.loads(rows[1]["chunk_ids"]) == []
    assert rows[2]["query"] is None
    search(db, "volcano", now=NOW, log=False)
    assert db.execute("SELECT COUNT(*) FROM retrieval_log").fetchone()[0] == 3

"""Retention: old articles go, with their chunks, keyword index entries and vectors, and aren't
fetched back by the next ingest."""

from datetime import UTC, datetime

import pytest
from fakes import FakeEmbedder
from test_ingest import client

from zands import cli, ingest, store
from zands.embed import embed_pending

NOW = datetime(2026, 11, 27, 12, 0, tzinfo=UTC)


def add(conn, url, published, body):
    store.insert_article(
        conn,
        store.Article(url=url, title=body.split(".")[0], source="N", body=body, extraction="rss",
                      published_at=published),
        [body],
    )


@pytest.fixture
def db(conn):
    add(conn, "https://n.example/old", "2026-08-01T09:00:00Z", "Glacier retreat measured. The ice moved.")
    add(conn, "https://n.example/undated", None, "Undated story about harbours.")  # fetched now: kept
    add(conn, "https://n.example/new", "2026-11-26T09:00:00Z", "Harbour bridge reopens. Traffic flows.")
    embed_pending(conn, FakeEmbedder())
    return conn


def matches(conn, word):
    return conn.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?", (word,)).fetchall()


def test_prune_removes_old_articles_and_everything_built_from_them(db):
    before = ingest.cutoff(30, NOW)
    assert before == "2026-10-28T12:00:00Z"
    assert store.prunable(db, before) == (1, 1)
    assert store.vector_count(db) == 3

    assert store.prune(db, before) == (1, 1)

    urls = [r["url"] for r in db.execute("SELECT url FROM articles ORDER BY url")]
    assert urls == ["https://n.example/new", "https://n.example/undated"]
    assert db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 2
    assert store.vector_count(db) == 2
    assert matches(db, "glacier") == [] and matches(db, "harbour")
    assert store.article_exists(db, "https://n.example/old?utm_source=rss")  # remembered as pruned
    assert store.prune(db, before) == (0, 0)


def test_a_pruned_article_isnt_fetched_again(conn):
    first = ingest.run_ingest(conn, [ingest.Feed("Good", "https://good.example/rss")], client=client(), keep=0)
    assert first.new == 2  # the paywalled one is kept from its RSS text
    store.prune(conn, "2030-01-01T00:00:00Z")  # everything goes
    assert store.stats(conn)["articles"] == 0

    again = ingest.run_ingest(conn, [ingest.Feed("Good", "https://good.example/rss")], client=client(), keep=0)

    assert again.new == 0 and store.stats(conn)["articles"] == 0


def test_ingest_prunes_when_told_how_long_to_keep(conn, monkeypatch):
    add(conn, "https://n.example/ancient", "2020-01-01T00:00:00Z", "Ancient story.")
    feeds = [ingest.Feed("Good", "https://good.example/rss")]

    kept = ingest.run_ingest(conn, feeds, client=client())  # ZANDS_KEEP_DAYS unset: keep all
    assert kept.pruned == 0 and store.article_exists(conn, "https://n.example/ancient")

    monkeypatch.setenv("ZANDS_KEEP_DAYS", "365")
    assert ingest.keep_days() == 365
    report = ingest.run_ingest(conn, feeds, client=client())
    assert report.pruned == 1
    assert conn.execute("SELECT COUNT(*) FROM articles WHERE url LIKE '%ancient%'").fetchone()[0] == 0

    monkeypatch.setenv("ZANDS_KEEP_DAYS", "forever")
    assert ingest.keep_days() == 0


def test_the_prune_command(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("ZANDS_KEEP_DAYS", raising=False)
    path = tmp_path / "news.db"
    conn = store.connect(path)
    add(conn, "https://n.example/old", "2020-01-01T00:00:00Z", "Old story. " * 200)
    add(conn, "https://n.example/new", datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "New story.")
    conn.close()

    assert cli.main(["--db", str(path), "prune"]) == 2
    assert "--keep-days N" in capsys.readouterr().err

    assert cli.main(["--db", str(path), "prune", "--keep-days", "30", "--dry-run"]) == 0
    assert "would remove 1 articles (1 chunks)" in capsys.readouterr().out
    assert store.stats(store.connect(path))["articles"] == 2

    assert cli.main(["--db", str(path), "prune", "--keep-days", "30", "--vacuum"]) == 0
    out = capsys.readouterr().out
    assert "removed 1 articles (1 chunks)" in out and "database:" in out and "MB ->" in out
    assert store.stats(store.connect(path))["articles"] == 1

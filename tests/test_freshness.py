"""News is fetched on demand, only when stale, never twice at once, and a failure never fails
the search."""

import fcntl
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from zands import freshness, store


@pytest.fixture
def db(tmp_path, monkeypatch):
    (tmp_path / "feeds.toml").write_text('[[feed]]\nname = "N"\nurl = "https://n.example/rss"\n')
    data = tmp_path / "data"
    path = data / "news.db"
    conn = store.connect(path)
    monkeypatch.delenv("ZANDS_FEEDS", raising=False)
    monkeypatch.delenv("ZANDS_STALE_HOURS", raising=False)
    return conn, path


def fetched(conn, hours_ago):
    when = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")
    store.insert_article(conn, store.Article(url=f"https://n.example/{hours_ago}", title="t", source="N", body="b",
                                             extraction="rss", published_at=when), ["b"])
    conn.execute("UPDATE articles SET fetched_at = ?", (when,))
    conn.commit()


class Ingest:
    def __init__(self, new=3, fail=None):
        self.calls, self.new, self.fail = [], new, fail

    def __call__(self, conn, feeds, embedder=None):
        self.calls.append([f.name for f in feeds])
        if self.fail:
            raise self.fail
        return SimpleNamespace(new=self.new, embed_error=None)


def test_fresh_news_isnt_fetched_again(db):
    conn, path = db
    fetched(conn, 1)
    ingest = Ingest()
    assert freshness.refresh_if_stale(conn, path, ingest=ingest).ran is False
    assert ingest.calls == []


def test_stale_news_is_fetched_from_the_repos_feeds_file(db):
    conn, path = db
    fetched(conn, 30)
    ingest = Ingest(new=7)
    r = freshness.refresh_if_stale(conn, path, ingest=ingest)
    assert r.ran and ingest.calls == [["N"]]
    assert r.note == "news was 30h old, so feeds were fetched just now: 7 new articles"


def test_an_empty_index_is_fetched_and_zero_turns_it_off(db, monkeypatch):
    conn, path = db
    assert "never fetched" in freshness.refresh_if_stale(conn, path, ingest=Ingest()).note
    monkeypatch.setenv("ZANDS_STALE_HOURS", "0")
    ingest = Ingest()
    assert freshness.refresh_if_stale(conn, path, ingest=ingest).ran is False and ingest.calls == []


def test_a_running_refresh_isnt_doubled(db):
    conn, path = db
    fetched(conn, 30)
    with open(path.parent / ".ingest.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        ingest = Ingest()
        r = freshness.refresh_if_stale(conn, path, ingest=ingest)
    assert not r.ran and "another refresh is running" in r.note and ingest.calls == []


def test_a_failed_refresh_is_a_note_not_an_error(db):
    conn, path = db
    fetched(conn, 30)
    r = freshness.refresh_if_stale(conn, path, ingest=Ingest(fail=OSError("offline")))
    assert not r.ran and "refreshing failed (offline)" in r.note

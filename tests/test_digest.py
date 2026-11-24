import json
from datetime import datetime, timezone

from zands import digest, store
from zands.embed import embed_pending

from fakes import FakeEmbedder

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)

ARTICLES = [
    # (source, title, body, published)
    ("BBC", "Volcano erupts in Iceland", "Lava and magma eruption volcano near Grindavik", "2026-11-24T08:00:00Z"),
    ("DW", "Iceland volcano eruption forces evacuation", "Lava magma eruption volcano near Grindavik town", "2026-11-24T09:00:00Z"),
    ("Guardian", "Live: Iceland eruption", "Volcano lava magma eruption Grindavik", "2026-11-24T10:00:00Z"),
    ("BBC", "Bank holds interest rates", "The central bank held interest rates borrowing", "2026-11-24T07:00:00Z"),
    ("NPR", "Cup final tonight", "The cup final kicks off tonight at the stadium", "2026-11-24T11:00:00Z"),
    ("BBC", "Old story", "Something from last week", "2026-11-20T08:00:00Z"),
    ("web: example.com", "Saved web page", "Lava magma volcano eruption", "2026-11-24T10:30:00Z"),
]


def fill(conn, embed=True):
    for i, (source, title, body, published) in enumerate(ARTICLES):
        store.insert_article(conn, store.Article(url=f"https://n.example/{i}", title=title, source=source, body=body,
                                                 extraction="rss", published_at=published), [body])
    if embed:
        embed_pending(conn, FakeEmbedder())


def test_groups_same_story_and_ranks_by_outlets(conn):
    fill(conn)
    d = digest.build(conn, now=NOW)
    assert d.article_count == 5  # old story and saved web pages are left out
    top = d.stories[0]
    assert top.outlets == 3 and sorted(top.sources) == ["BBC", "DW", "Guardian"]
    assert len(d.stories) == 3
    assert {s.title for s in d.stories[1:]} == {"Bank holds interest rates", "Cup final tonight"}


def test_render_and_save(conn):
    fill(conn)
    d = digest.refresh(conn, now=NOW)
    text = d.text
    assert text.startswith("Top stories, last 24 hours (as of 2026-11-24 12:00 UTC, 5 articles):")
    assert "[D1]" in text and "(BBC, DW, Guardian; 2026-11-24)" in text
    row = digest.latest(conn, now=NOW)
    assert row["day"] == "2026-11-24" and row["text"] == text
    assert json.loads(row["items"])[0]["sources"]
    # one row per day: a rebuild replaces it
    digest.refresh(conn, now=NOW)
    assert conn.execute("SELECT COUNT(*) FROM digests").fetchone()[0] == 1


def test_latest_respects_max_age(conn):
    fill(conn)
    digest.refresh(conn, now=NOW)
    later = datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc)
    assert digest.latest(conn, now=later) is None
    assert digest.latest(conn, max_age_hours=None, now=later) is not None


def test_fallback_without_vectors_groups_by_title(tmp_path):
    conn = store.connect(tmp_path / "novec.db", vector=False)
    for i, (src, title) in enumerate([("A", "Iceland volcano eruption forces evacuation"),
                                      ("B", "Iceland volcano eruption: evacuation ordered"),
                                      ("C", "Cup final tonight")]):
        store.insert_article(conn, store.Article(url=f"https://x/{i}", title=title, source=src, body="b",
                                                 extraction="rss", published_at="2026-11-24T09:00:00Z"), ["b"])
    d = digest.build(conn, now=NOW)
    assert d.stories[0].outlets == 2 and len(d.stories) == 2


def test_empty_digest(conn):
    d = digest.build(conn, now=NOW)
    assert d.stories == [] and d.text == "No articles in the last 24 hours."

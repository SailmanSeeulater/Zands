from zands import store
from zands.chunk import chunk_text


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def make_article(url="https://example.com/a", title="Title", body=None, **kw):
    return store.Article(url=url, title=title, source="Example", body=body or words(50), extraction="rss", **kw)


def test_normalize_url():
    n = store.normalize_url
    assert n("HTTP://Example.COM/News/Story/?utm_source=rss&utm_medium=x#top") == "https://example.com/News/Story"
    assert n("https://example.com/a?b=2&a=1&fbclid=zzz") == "https://example.com/a?a=1&b=2"
    assert n("https://example.com:443/a/") == "https://example.com/a"
    assert n("https://example.com") == "https://example.com/"
    assert n("https://example.com/a?id=5") != n("https://example.com/a?id=6")
    assert n("https://www.dw.com/en/story/a-1?maca=en-rss-en-all-1573-rdf") == "https://www.dw.com/en/story/a-1"
    assert n("https://www.aljazeera.com/news/x?traffic_source=rss") == "https://www.aljazeera.com/news/x"


def test_insert_is_idempotent(conn):
    art = make_article(body=words(1200))
    chunks = chunk_text(art.body)
    first = store.insert_article(conn, art, chunks)
    assert first is not None
    # same article again, and again via a tracking-param / trailing-slash variant
    assert store.insert_article(conn, art, chunks) is None
    variant = make_article(url="http://EXAMPLE.com/a/?utm_campaign=feed", body=words(1200))
    assert store.insert_article(conn, variant, chunk_text(variant.body)) is None

    assert conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == len(chunks)
    assert conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == len(chunks)
    assert store.article_exists(conn, "https://example.com/a?utm_source=x")


def test_chunks_carry_article_metadata(conn):
    art = make_article(title="Big Story", published_at="2026-11-24T08:00:00Z")
    store.insert_article(conn, art, ["one two three"])
    row = conn.execute("SELECT title, source, url, published_at FROM chunks").fetchone()
    assert tuple(row) == ("Big Story", "Example", "https://example.com/a", "2026-11-24T08:00:00Z")


def test_fts_tracks_deletes(conn):
    aid = store.insert_article(conn, make_article(body="zebra crossing"), ["zebra crossing"])
    assert conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'zebra'").fetchone()[0] == 1
    conn.execute("DELETE FROM articles WHERE id = ?", (aid,))
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'zebra'").fetchone()[0] == 0


def test_feed_health(conn):
    fid = store.ensure_feed(conn, "Feed", "https://example.com/rss")
    assert store.ensure_feed(conn, "Feed", "https://example.com/rss") == fid
    store.record_feed_failure(conn, fid, "boom")
    store.record_feed_failure(conn, fid, "boom again")
    row = conn.execute("SELECT * FROM feeds WHERE id = ?", (fid,)).fetchone()
    assert row["failure_count"] == 2 and row["last_error"] == "boom again"
    store.record_feed_success(conn, fid, 7)
    row = conn.execute("SELECT * FROM feeds WHERE id = ?", (fid,)).fetchone()
    assert row["failure_count"] == 0 and row["last_success_at"] and row["last_new_items"] == 7
    assert row["last_error"] == "boom again"  # kept for debugging


def test_connect_is_reentrant(tmp_path):
    path = tmp_path / "sub" / "x.db"
    store.connect(path).close()
    c = store.connect(path)  # schema creation is IF NOT EXISTS
    assert c.execute("PRAGMA user_version").fetchone()[0] == store.SCHEMA_VERSION

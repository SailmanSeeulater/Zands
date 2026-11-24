"""Ingest end to end against a fake HTTP transport: no network."""

import httpx

from zands import ingest, store

RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>Good Feed</title>
<item>
  <title>Harbour bridge reopens</title>
  <link>https://good.example/bridge?utm_source=rss</link>
  <pubDate>Tue, 24 Nov 2026 08:00:00 GMT</pubDate>
  <description>&lt;p&gt;The bridge reopened after repairs.&lt;/p&gt;</description>
</item>
<item>
  <title>Paywalled story</title>
  <link>https://good.example/paywall</link>
  <pubDate>Tue, 24 Nov 2026 07:00:00 GMT</pubDate>
  <description>&lt;p&gt;Summary of the paywalled story about lighthouse keepers.&lt;/p&gt;</description>
</item>
</channel></rss>"""

ARTICLE_TEXT = " ".join(
    f"The harbour bridge reopened to traffic on Tuesday after eighteen months of repairs, sentence {i}."
    for i in range(20)
)
ARTICLE_HTML = f"""<html><head><title>Harbour bridge reopens</title></head>
<body><nav>Home | World</nav><article><h1>Harbour bridge reopens</h1>
<p>{ARTICLE_TEXT}</p><p>Officials said the work finished on schedule and under budget.</p>
</article><footer>Copyright</footer></body></html>"""


def handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if url == "https://good.example/rss":
        return httpx.Response(200, content=RSS.encode(), headers={"content-type": "application/rss+xml"})
    if url.startswith("https://good.example/bridge"):
        return httpx.Response(200, text=ARTICLE_HTML, headers={"content-type": "text/html"})
    if url == "https://good.example/paywall":
        return httpx.Response(403, text="Forbidden")
    if url == "https://broken.example/rss":
        return httpx.Response(500, text="oops")
    if url == "https://garbage.example/rss":
        return httpx.Response(200, text="this is not xml at all", headers={"content-type": "text/plain"})
    raise httpx.ConnectError("no route", request=request)


FEEDS = [
    ingest.Feed("Broken", "https://broken.example/rss"),
    ingest.Feed("Garbage", "https://garbage.example/rss"),
    ingest.Feed("Good", "https://good.example/rss"),
    ingest.Feed("Unreachable", "https://nowhere.example/rss"),
]


def client() -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_broken_feeds_do_not_stop_the_run(conn):
    report = ingest.run_ingest(conn, FEEDS, client=client())
    by_name = {f.name: f for f in report.feeds}
    assert by_name["Good"].ok and by_name["Good"].new == 2
    assert not by_name["Broken"].ok and "500" in by_name["Broken"].error
    assert not by_name["Garbage"].ok
    assert not by_name["Unreachable"].ok

    feeds = {r["name"]: r for r in conn.execute("SELECT * FROM feeds")}
    assert feeds["Good"]["failure_count"] == 0 and feeds["Good"]["last_success_at"]
    assert feeds["Broken"]["failure_count"] == 1 and feeds["Broken"]["last_error"]


def test_extraction_and_rss_fallback(conn):
    ingest.run_ingest(conn, FEEDS, client=client())
    rows = {r["title"]: r for r in conn.execute("SELECT * FROM articles")}
    bridge = rows["Harbour bridge reopens"]
    assert bridge["extraction"] == "trafilatura"
    assert "eighteen months of repairs" in bridge["body"]
    assert "Copyright" not in bridge["body"]
    assert bridge["url"] == "https://good.example/bridge"  # tracking param stripped
    assert bridge["published_at"] == "2026-11-24T08:00:00Z"
    assert bridge["source"] == "Good"

    pay = rows["Paywalled story"]
    assert pay["extraction"] == "rss"
    assert pay["body"] == "Summary of the paywalled story about lighthouse keepers."
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] >= 2


def test_rerun_is_idempotent(conn):
    ingest.run_ingest(conn, FEEDS, client=client())
    counts = conn.execute("SELECT (SELECT COUNT(*) FROM articles), (SELECT COUNT(*) FROM chunks)").fetchone()
    report = ingest.run_ingest(conn, FEEDS, client=client())
    good = next(f for f in report.feeds if f.name == "Good")
    assert good.new == 0 and good.skipped == 2
    assert conn.execute("SELECT (SELECT COUNT(*) FROM articles), (SELECT COUNT(*) FROM chunks)").fetchone() == counts
    broken = conn.execute("SELECT failure_count FROM feeds WHERE name = 'Broken'").fetchone()[0]
    assert broken == 2  # consecutive failures accumulate


def test_load_feeds(tmp_path):
    p = tmp_path / "feeds.toml"
    p.write_text(
        '[[feed]]\nname = "A"\nurl = "https://a.example/rss"\n\n'
        '[[feed]]\nname = "B"\nurl = "https://b.example/rss"\nenabled = false\n'
    )
    assert ingest.load_feeds(p) == [ingest.Feed("A", "https://a.example/rss")]


def test_html_to_text():
    assert ingest.html_to_text("<p>One &amp; two</p><p>Three<br>four</p>") == "One & two\n\nThree\n\nfour"
    assert ingest.html_to_text("") == ""


def test_store_reports_after_ingest(conn):
    ingest.run_ingest(conn, FEEDS, client=client())
    s = store.stats(conn)
    assert s["articles"] == 2 and len(s["feeds"]) == 4

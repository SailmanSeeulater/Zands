"""`zands dashboard`: panel queries on a fixture database, the empty states, and the HTTP server."""

import json
import re
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone

import pytest

from zands import dashboard, digest, facts, store
from zands.search import Hit, hit_sources, retrieve

NOW = datetime(2026, 11, 24, 18, 30, tzinfo=timezone.utc)


def add(conn, i, title, feed_id, source, fetched, published=None):
    store.insert_article(conn, store.Article(
        url=f"https://n.example/{i}", title=title, source=source, body=f"{title}. Body text {i}.", extraction="rss",
        published_at=published or fetched, feed_id=feed_id, fetched_at=fetched,
    ), [f"{title}. Body text {i}."])


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "news.db"
    conn = store.connect(path)
    bbc = store.ensure_feed(conn, "BBC", "https://bbc.example/rss")
    npr = store.ensure_feed(conn, "NPR", "https://npr.example/rss")
    broken = store.ensure_feed(conn, "Broken", "https://broken.example/rss")
    store.record_feed_success(conn, bbc, 2)
    store.record_feed_success(conn, npr, 1)
    store.record_feed_failure(conn, broken, "HTTP 500")
    add(conn, 1, "Nvidia reinstates Huang as CEO", bbc, "BBC", "2026-11-24T13:10:00Z")
    add(conn, 2, "Huang returns to Nvidia amid backlash", npr, "NPR", "2026-11-24T13:40:00Z")
    add(conn, 3, "Iceland volcano erupts near Grindavik", bbc, "BBC", "2026-11-24T16:05:00Z", "2026-11-23T08:00:00Z")
    add(conn, 4, "Web page about Nvidia", None, "web: x.example", "2026-11-24T16:06:00Z")
    digest.refresh(conn, now=NOW)
    facts.add_fact(conn, "Nvidia", "CEO", "Jensen Huang", valid_from="2026-11-23")
    old = facts.add_fact(conn, "Spain", "election", "called", valid_from="2026-11-01")
    facts.expire_fact(conn, old, on="2026-11-20")
    retrieve(conn, "Nvidia CEO", mode="keyword", command="search")
    retrieve(conn, "zebra crossing", mode="keyword", command="ask")  # finds nothing
    retrieve(conn, "Nvidia", mode="keyword", command="eval")  # left out of the retrieval panel
    log_id = store.log_retrieval(conn, "What are today's top stories?", None, [], 8, command="ask")
    store.set_log_route(conn, log_id, "digest", "rule")  # never searches, so it isn't a miss
    conn.close()
    return path


def panels(path):
    conn = dashboard.connect_ro(path)
    try:
        return dashboard.all_panels(conn, path, now=NOW)
    finally:
        conn.close()


def test_coverage(db):
    c = panels(db)["coverage"]
    assert (c["articles"], c["chunks"], c["facts_current"], c["facts_expired"]) == (4, 4, 1, 1)
    assert c["retrievals"] == 3  # eval rows excluded
    assert c["db_bytes"] > 0
    if c["vectors"] is not None:  # sqlite-vec loaded: nothing embedded in this fixture
        assert c["vectors"] == 0 and c["vector_coverage"] == 0


def test_ingested_hourly_per_feed(db):
    d = panels(db)["ingested"]
    assert d["bucket"] == "hour"
    assert d["buckets"][0] == "2026-11-24T13:00:00Z" and d["buckets"][-1] == "2026-11-24T18:00:00Z"
    series = {s["name"]: s["values"] for s in d["series"]}
    assert set(series) == {"BBC", "NPR"}  # web pages aren't feed articles
    assert series["BBC"] == [1, 0, 0, 1, 0, 0] and series["NPR"] == [1, 0, 0, 0, 0, 0]


def test_ingested_goes_daily_and_folds_feeds(tmp_path):
    conn = store.connect(tmp_path / "n.db")
    for i in range(10):
        fid = store.ensure_feed(conn, f"Feed {i:02d}", f"https://f{i}.example")
        for j in range(i + 1):  # Feed 09 has the most articles, Feed 00 the fewest
            add(conn, f"{i}-{j}", f"Story {i} {j}", fid, f"Feed {i:02d}", f"2026-11-{10 + j % 10:02d}T10:00:00Z")
    d = dashboard.ingested(conn, now=NOW)
    assert d["bucket"] == "day" and d["buckets"][0] == "2026-11-10" and d["buckets"][-1] == "2026-11-24"
    names = [s["name"] for s in d["series"]]
    assert len(names) == dashboard.MAX_SERIES + 1 and names[-1] == "Other"
    assert names[:-1] == sorted(names[:-1]) and "Feed 00" not in names
    total = sum(sum(s["values"]) for s in d["series"])
    assert total == sum(range(1, 11))  # nothing lost in the fold


def test_feeds_status_and_collecting_runs(db):
    d = panels(db)["feeds"]
    status = {f["name"]: f["status"] for f in d["feeds"]}
    assert status["Broken"] == "failing"
    assert {f["name"]: f["articles"] for f in d["feeds"]} == {"BBC": 2, "Broken": 0, "NPR": 1}
    assert d["runs"] == [] and d["collecting_since"]  # no ingest runs recorded yet
    stale = dashboard.feeds(dashboard.connect_ro(db), now=datetime(2030, 1, 1, tzinfo=timezone.utc))
    assert {f["name"]: f["status"] for f in stale["feeds"]}["BBC"] == "stale"


def test_digest_counts_distinct_outlets(db):
    d = panels(db)["digest"]
    assert d["day"] == "2026-11-24"
    nvidia = next(s for s in d["stories"] if "Nvidia" in s["title"] or "Huang" in s["title"])
    assert nvidia["sources"] == sorted(set(nvidia["sources"]))
    assert nvidia["outlets"] == len(nvidia["sources"])


def test_topics_from_headlines(db):
    d = panels(db)["topics"]
    assert d["days"][0] == "2026-11-23" and d["days"][-1] == "2026-11-24"
    terms = {t["term"]: t for t in d["terms"]}
    assert terms["Nvidia"]["total"] == 2 and terms["Huang"]["values"] == [0, 2]  # the web page is left out
    assert "Grindavik" not in terms  # one mention isn't a topic


def test_headline_terms_drop_noise():
    assert [w for _, w in dashboard.headline_terms("Live: Trump's new tariffs on China, says report")] == \
        ["Trump", "tariffs", "China"]


def test_retrievals_panel(db):
    d = panels(db)["retrievals"]
    assert d["total"] == 3 and d["days"] == ["2026-11-24"]
    assert {s["name"]: s["values"] for s in d["by_command"]} == {"ask": [2], "search": [1]}
    assert d["routes"] == [{"route": "digest", "n": 1}]
    assert d["empty"] == 1 and d["empty_recent"][0]["question"] == "zebra crossing"
    assert d["hit_sources"]["keyword"] >= 1 and d["hit_sources"]["vector"] == 0
    assert d["hit_sources_logged"] == 2  # the digest row was logged without retrieval


def test_hit_sources_counts_top_k_per_retriever():
    def hit(kw, vec):
        return Hit(chunk_id=1, article_id=1, title="t", source="s", url="u", published_at=None, body="", snippet="",
                   score=1.0, keyword_rank=kw, vector_rank=vec)

    hits = [hit(1, 2), hit(3, None), hit(None, 1), hit(30, 4), hit(25, 40)]
    assert hit_sources(hits, k=8) == {"keyword": 1, "vector": 2, "both": 1, "fusion": 1}


def test_evals_panel_empty_then_filled(db):
    assert panels(db)["evals"]["retrieval"] == [] and panels(db)["evals"]["collecting_since"]
    conn = store.connect(db)
    store.record_eval_run(conn, "retrieval", 20, mode="hybrid", k=8, scored=20, hit1=0.9, hit3=1.0, hitk=1.0,
                          mrr=0.94, fallbacks=0)
    store.record_eval_run(conn, "routes", 27, mode="rules+model", scored=27, accuracy=1.0)
    conn.close()
    d = panels(db)["evals"]
    assert [(r["mode"], r["hit1"], r["mrr"]) for r in d["retrieval"]] == [("hybrid", 0.9, 0.94)]
    assert [(r["correct"], r["accuracy"]) for r in d["routes"]] == [(27, 1.0)]


def test_empty_database_has_collecting_states(tmp_path):
    path = tmp_path / "empty.db"
    store.connect(path).close()
    d = panels(path)
    assert d["ingested"]["buckets"] == [] and d["digest"]["stories"] == [] and d["topics"]["terms"] == []
    assert d["retrievals"]["total"] == 0
    for since in (d["feeds"]["collecting_since"], d["retrievals"]["collecting_since"], d["evals"]["collecting_since"]):
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", since)


def test_collecting_since_is_stamped_once(tmp_path):
    path = tmp_path / "x.db"
    conn = store.connect(path)
    first = store.collecting_since(conn, "ingest_runs")
    store.set_meta(conn, "collecting_since:ingest_runs", "2026-01-01T00:00:00Z")
    conn.close()
    assert first and store.collecting_since(store.connect(path), "ingest_runs") == "2026-01-01T00:00:00Z"


def test_read_only_connection_cannot_write(db):
    conn = dashboard.connect_ro(db)
    with pytest.raises(Exception, match="readonly"):
        conn.execute("DELETE FROM articles")


@pytest.fixture
def server(db):
    srv = dashboard.make_server(db, port=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def get(url):
    with urllib.request.urlopen(url, timeout=5) as res:
        return res.status, res.headers.get("Content-Type"), res.read()


def test_server_serves_page_and_json(server):
    status, ctype, body = get(server + "/")
    assert status == 200 and ctype.startswith("text/html")
    html = body.decode()
    assert "<title>Zands dashboard</title>" in html and "api/all.json" in html
    assert not re.search(r"""(src|href)=["']https?://""", html)  # nothing loaded from the network
    assert "prefers-color-scheme: dark" in html and 'data-theme="dark"' in html
    status, ctype, body = get(server + "/api/all.json")
    data = json.loads(body)
    assert ctype == "application/json" and set(dashboard.PANELS) <= set(data)
    assert data["coverage"]["articles"] == 4
    for name in dashboard.PANELS:  # one endpoint per panel, same shape as in all.json
        assert set(json.loads(get(f"{server}/api/{name}.json")[2])) == set(data[name])


def test_server_unknown_path_is_404(server):
    with pytest.raises(urllib.error.HTTPError) as err:
        get(server + "/api/secrets.json")
    assert err.value.code == 404

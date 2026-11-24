"""`zands dashboard`: one local page of charts over news.db.

A stdlib HTTP server on 127.0.0.1 serves `dashboard.html` (hand-drawn SVG charts, no CDN, so it
works offline) and one read-only JSON endpoint per panel: /api/<panel>.json, plus /api/all.json.
Every request opens the database read-only, so the dashboard can't change it.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path

from . import store
from .search import STOPWORDS

MAX_SERIES = 7  # feeds past this fold into "Other" (8 categorical colours at most)
FEED_STALE_HOURS = 3  # the launchd job runs hourly; three missed runs is worth flagging

# Headline words that say nothing about the topic.
HEADLINE_NOISE = frozenset(
    """says said new news live latest update updates video watch photos pictures week day today year years
    first last after over more could would will can how what why who when where amid against into about
    back just make made get gets one two three four five six seven eight nine ten top big report reports
    people world time man woman calls call takes take set sets via ahead near like also still need""".split()
)


def utc(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def connect_ro(path: str | Path) -> sqlite3.Connection:
    """Read-only connection with sqlite-vec loaded when available (for the vector count)."""
    conn = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        store.load_vec(conn)
    except Exception:  # noqa: BLE001 - the coverage panel shows "unavailable" instead
        pass
    return conn


def _collecting(conn: sqlite3.Connection, name: str) -> str | None:
    try:
        return store.collecting_since(conn, name)
    except sqlite3.Error:
        return None


def _hour_buckets(start: datetime, end: datetime) -> list[datetime]:
    cur = start.replace(minute=0, second=0, microsecond=0)
    out = []
    while cur <= end:
        out.append(cur)
        cur += timedelta(hours=1)
    return out


def _days(start: datetime, end: datetime) -> list[str]:
    cur, out = start.date(), []
    while cur <= end.date():
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


# ---------------------------------------------------------------- panels


def coverage(conn: sqlite3.Connection, db_path: str | Path | None = None, now: datetime | None = None) -> dict:
    one = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    try:
        vectors = one("SELECT COUNT(*) FROM chunk_vec")
    except sqlite3.Error:
        vectors = None
    chunks = one("SELECT COUNT(*) FROM chunks")
    size = None
    if db_path:
        size = sum(p.stat().st_size for p in (Path(f"{db_path}{s}") for s in ("", "-wal")) if p.exists())
    return {
        "articles": one("SELECT COUNT(*) FROM articles"),
        "chunks": chunks,
        "vectors": vectors,
        "vector_coverage": (vectors / chunks) if vectors is not None and chunks else None,
        "db_bytes": size,
        "facts_current": one("SELECT COUNT(*) FROM facts WHERE valid_to IS NULL"),
        "facts_expired": one("SELECT COUNT(*) FROM facts WHERE valid_to IS NOT NULL"),
        "retrievals": one("SELECT COUNT(*) FROM retrieval_log WHERE command IS NOT 'eval'"),
        "newest_article": one("SELECT MAX(published_at) FROM articles"),
        "last_fetch": one("SELECT MAX(fetched_at) FROM articles"),
    }


def ingested(conn: sqlite3.Connection, now: datetime | None = None) -> dict:
    """Articles stored over time, per feed: hourly for the last 48 h while the index is young,
    daily for the last 30 days once it has more than three days of history."""
    now = now or datetime.now(timezone.utc)
    first = utc(conn.execute("SELECT MIN(fetched_at) FROM articles").fetchone()[0])
    if first is None:
        return {"bucket": "hour", "buckets": [], "series": []}
    hourly = (now - first) <= timedelta(days=3)
    if hourly:
        start = max(first, now - timedelta(hours=47))
        keys = [iso(b) for b in _hour_buckets(start, now)]
        key_sql = "strftime('%Y-%m-%dT%H:00:00Z', a.fetched_at)"
    else:
        start = max(first, now - timedelta(days=29))
        keys = _days(start, now)
        key_sql = "substr(a.fetched_at, 1, 10)"
    rows = conn.execute(
        f"""SELECT {key_sql} AS bucket, COALESCE(f.name, a.source) AS feed, COUNT(*) AS n
            FROM articles a LEFT JOIN feeds f ON f.id = a.feed_id
            WHERE a.fetched_at >= ? AND a.source NOT LIKE 'web: %' GROUP BY bucket, feed""",
        (iso(start.replace(minute=0, second=0, microsecond=0)) if hourly else keys[0],),
    ).fetchall()
    totals: Counter[str] = Counter()
    for r in rows:
        totals[r["feed"]] += r["n"]
    # Alphabetical, so a feed keeps its colour as the totals shift; the smallest fold into "Other".
    folds = len(totals) > MAX_SERIES + 1
    keep = sorted(name for name, _ in totals.most_common(MAX_SERIES)) if folds else sorted(totals)
    index = {k: i for i, k in enumerate(keys)}
    series: dict[str, list[int]] = {name: [0] * len(keys) for name in keep}
    other = [0] * len(keys)
    for r in rows:
        if r["bucket"] not in index:
            continue
        target = series.get(r["feed"], other)
        target[index[r["bucket"]]] += r["n"]
    out = [{"name": n, "values": series[n]} for n in keep]
    if any(other):
        out.append({"name": "Other", "values": other})
    return {"bucket": "hour" if hourly else "day", "buckets": keys, "series": out}


def feeds(conn: sqlite3.Connection, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    rows = conn.execute(
        """SELECT f.name, f.last_success_at, f.last_error, f.last_error_at, f.failure_count, f.last_new_items,
                  COUNT(a.id) AS articles, MAX(a.published_at) AS newest
           FROM feeds f LEFT JOIN articles a ON a.feed_id = f.id GROUP BY f.id ORDER BY f.name"""
    ).fetchall()
    out = []
    for r in rows:
        last = utc(r["last_success_at"])
        if r["failure_count"]:
            status = "failing"
        elif last is None:
            status = "never"
        elif now - last > timedelta(hours=FEED_STALE_HOURS):
            status = "stale"
        else:
            status = "ok"
        out.append({**dict(r), "status": status})
    runs = [dict(r) for r in conn.execute(
        """SELECT id, started_at, seconds, new_articles, feeds_ok, feeds_failed, embedded, embed_error
           FROM ingest_runs ORDER BY id DESC LIMIT 48"""
    )][::-1]
    return {"feeds": out, "runs": runs, "collecting_since": _collecting(conn, "ingest_runs")}


def digest(conn: sqlite3.Connection, now: datetime | None = None) -> dict:
    row = conn.execute("SELECT day, created_at, hours, items FROM digests ORDER BY day DESC LIMIT 1").fetchone()
    if row is None:
        return {"day": None, "stories": []}
    items = json.loads(row["items"])
    stories = []
    for i in items:
        outlets = sorted(set(i["sources"]))  # `sources` has one entry per article
        stories.append({"title": i["title"], "sources": outlets, "outlets": len(outlets),
                        "articles": len(i.get("article_ids", i["sources"])), "newest": i.get("newest")})
    return {"day": row["day"], "created_at": row["created_at"], "hours": row["hours"], "stories": stories}


WORD = re.compile(r"[A-Za-z][A-Za-z'’\-]+")


def headline_terms(title: str) -> list[tuple[str, str]]:
    """(key, display) pairs for the topic words in a headline."""
    out = []
    for raw in WORD.findall(title):
        word = raw.strip("'’-")
        if word.lower().endswith(("'s", "’s")):
            word = word[:-2]
        key = word.lower()
        if len(key) < 3 or key in STOPWORDS or key in HEADLINE_NOISE:
            continue
        out.append((key, word))
    return out


def topics(conn: sqlite3.Connection, now: datetime | None = None, days: int = 7, top: int = 12) -> dict:
    """Most frequent headline words per day: a cheap stand-in for entity extraction."""
    now = now or datetime.now(timezone.utc)
    start = now - timedelta(days=days - 1)
    rows = conn.execute(
        """SELECT title, substr(COALESCE(published_at, fetched_at), 1, 10) AS day FROM articles
           WHERE COALESCE(published_at, fetched_at) >= ? AND source NOT LIKE 'web: %'""",
        (start.date().isoformat(),),
    ).fetchall()
    if not rows:
        return {"days": [], "terms": []}
    first_day = min(r["day"] for r in rows)
    day_keys = [d for d in _days(start, now) if d >= first_day]
    per_day: dict[str, Counter[str]] = defaultdict(Counter)
    forms: dict[str, Counter[str]] = defaultdict(Counter)
    totals: Counter[str] = Counter()
    for r in rows:
        for key in {k for k, _ in headline_terms(r["title"])}:  # once per headline
            per_day[key][r["day"]] += 1
            totals[key] += 1
        for key, form in headline_terms(r["title"]):
            forms[key][form] += 1
    terms = [
        {"term": forms[key].most_common(1)[0][0], "total": n, "values": [per_day[key].get(d, 0) for d in day_keys]}
        for key, n in totals.most_common(top) if n >= 2
    ]
    return {"days": day_keys, "terms": terms, "headlines": len(rows)}


def retrievals(conn: sqlite3.Connection, now: datetime | None = None, days: int = 30) -> dict:
    """Questions from search / ask / MCP (eval runs are left out: they'd swamp everything)."""
    now = now or datetime.now(timezone.utc)
    where = "command IS NOT 'eval'"
    first = conn.execute(f"SELECT MIN(created_at) FROM retrieval_log WHERE {where}").fetchone()[0]
    if first is None:
        return {"days": [], "by_command": [], "routes": [], "modes": [], "hit_sources": None, "empty": 0,
                "empty_recent": [], "total": 0, "collecting_since": _collecting(conn, "hit_sources")}
    start = max(utc(first), now - timedelta(days=days - 1))
    day_keys = _days(start, now)
    index = {d: i for i, d in enumerate(day_keys)}
    by_command: dict[str, list[int]] = {}
    for r in conn.execute(
        f"""SELECT substr(created_at, 1, 10) AS day, COALESCE(command, 'other') AS command, COUNT(*) AS n
            FROM retrieval_log WHERE {where} AND created_at >= ? GROUP BY day, command""",
        (day_keys[0],),
    ):
        if r["day"] in index:
            by_command.setdefault(r["command"], [0] * len(day_keys))[index[r["day"]]] += r["n"]
    routes = [dict(r) for r in conn.execute(
        "SELECT route, COUNT(*) AS n FROM retrieval_log WHERE command = 'ask' AND route IS NOT NULL "
        "GROUP BY route ORDER BY n DESC"
    )]
    modes = [dict(r) for r in conn.execute(
        f"SELECT COALESCE(mode, 'unknown') AS mode, COUNT(*) AS n FROM retrieval_log WHERE {where} "
        "GROUP BY mode ORDER BY n DESC"
    )]
    sources = {"keyword": 0, "vector": 0, "both": 0, "fusion": 0, "web": 0}
    logged = 0
    for r in conn.execute(f"SELECT hit_sources, web_urls FROM retrieval_log WHERE {where} AND hit_sources IS NOT NULL"):
        logged += 1
        for k, v in json.loads(r["hit_sources"]).items():
            sources[k] = sources.get(k, 0) + v
        if r["web_urls"]:
            sources["web"] += len(json.loads(r["web_urls"]))
    # Digest and `none` answers never search, so an empty chunk list there isn't a miss.
    empty_where = (f"{where} AND chunk_ids = '[]' AND (web_urls IS NULL OR web_urls = '[]') "
                   "AND COALESCE(route, '') NOT IN ('digest', 'none')")
    empty = conn.execute(f"SELECT COUNT(*) FROM retrieval_log WHERE {empty_where}").fetchone()[0]
    empty_recent = [dict(r) for r in conn.execute(
        f"SELECT created_at, command, route, question FROM retrieval_log WHERE {empty_where} ORDER BY id DESC LIMIT 5"
    )]
    total = conn.execute(f"SELECT COUNT(*) FROM retrieval_log WHERE {where}").fetchone()[0]
    return {
        "days": day_keys,
        "by_command": [{"name": k, "values": v} for k, v in sorted(by_command.items())],
        "routes": routes, "modes": modes,
        "hit_sources": sources if logged else None, "hit_sources_logged": logged,
        "empty": empty, "empty_recent": empty_recent, "total": total,
        "collecting_since": _collecting(conn, "hit_sources"),
    }


def evals(conn: sqlite3.Connection, now: datetime | None = None) -> dict:
    retrieval = [dict(r) for r in conn.execute(
        "SELECT id, created_at, mode, k, questions, scored, hit1, hit3, hitk, mrr, fallbacks, answer_rate, rerank, "
        "question_set "
        "FROM eval_runs WHERE kind = 'retrieval' ORDER BY id"
    )]
    routes = [dict(r) for r in conn.execute(
        "SELECT id, created_at, mode, questions, scored AS correct, accuracy, question_set FROM eval_runs "
        "WHERE kind = 'routes' ORDER BY id"
    )]
    return {"retrieval": retrieval, "routes": routes, "collecting_since": _collecting(conn, "eval_runs")}


PANELS = {
    "coverage": coverage,
    "ingested": ingested,
    "feeds": feeds,
    "digest": digest,
    "topics": topics,
    "retrievals": retrievals,
    "evals": evals,
}


def panel(conn: sqlite3.Connection, name: str, db_path: str | Path | None = None, now: datetime | None = None) -> dict:
    if name == "coverage":
        return coverage(conn, db_path, now)
    return PANELS[name](conn, now=now)


def all_panels(conn: sqlite3.Connection, db_path: str | Path | None = None, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    data = {name: panel(conn, name, db_path, now) for name in PANELS}
    data["generated_at"] = iso(now)
    return data


# ---------------------------------------------------------------- server


def page() -> bytes:
    return resources.files("zands").joinpath("dashboard.html").read_bytes()


def make_handler(db_path: str | Path):
    db_path = Path(db_path).resolve()

    class Handler(BaseHTTPRequestHandler):
        server_version = "zands-dashboard"

        def log_message(self, fmt, *args):  # quiet: one line per request is noise here
            pass

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                return self._send(200, page(), "text/html; charset=utf-8")
            m = re.fullmatch(r"/api/([a-z]+)\.json", path)
            if not m or (m.group(1) != "all" and m.group(1) not in PANELS):
                return self._send(404, b'{"error": "not found"}', "application/json")
            try:
                conn = connect_ro(db_path)
                try:
                    name = m.group(1)
                    data = all_panels(conn, db_path) if name == "all" else panel(conn, name, db_path)
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                return self._send(500, json.dumps({"error": str(exc)}).encode(), "application/json")
            self._send(200, json.dumps(data).encode(), "application/json")

    return Handler


def make_server(db_path: str | Path, host: str = "127.0.0.1", port: int = 8002) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(db_path))

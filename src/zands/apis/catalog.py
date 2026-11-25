"""The public-apis list as a searchable table, so a model can find an API for a need no typed
tool covers, read its docs, and call it with call_api.

Source: the README of github.com/public-apis/public-apis, one markdown table per category:
    ### Animals
    API | Description | Auth | HTTPS | CORS
    | [Cat Facts](https://catfact.ninja/) | Random cat facts | No | Yes | Yes |
The sponsored tables at the top have a different shape and are skipped.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable

from ..search import build_fts_query
from . import http

README_URL = "https://raw.githubusercontent.com/public-apis/public-apis/master/README.md"
ROW = re.compile(r"^\|\s*\[(?P<name>[^\]]+)\]\((?P<url>[^)\s]+)\)\s*\|(?P<rest>.*)\|\s*$")


def parse_readme(text: str) -> list[dict]:
    """Rows of the five-column category tables, under the `### Category` they follow."""
    rows, category = [], None
    for line in text.splitlines():
        if line.startswith("### "):
            category = line[4:].strip()
            continue
        m = ROW.match(line.strip())
        if not m or category is None:
            continue
        cells = [c.strip() for c in m["rest"].split("|")]
        if len(cells) != 4:  # description, auth, https, cors
            continue
        description, auth, https, cors = cells
        auth = auth.strip("`").strip()
        rows.append(
            {
                "name": m["name"].strip(),
                "url": m["url"].strip(),
                "description": description,
                "category": category,
                "auth": "" if auth.lower() == "no" else auth,
                "https": 1 if https.lower() == "yes" else 0,
                "cors": cors.lower(),
            }
        )
    return rows


def update(conn: sqlite3.Connection, fetch: Callable[..., http.Fetched] = http.get) -> int:
    """Replace the catalog with the current list. Returns how many APIs it holds."""
    got = fetch(README_URL, timeout=30)
    if got.status != 200:
        raise http.FetchError(f"GitHub answered HTTP {got.status} for the public-apis list")
    rows = parse_readme(got.text)
    if len(rows) < 100:  # the README changed shape: keep what we have rather than empty it
        raise http.FetchError(f"only {len(rows)} APIs parsed from the public-apis list; its format may have changed")
    with conn:
        conn.execute("DELETE FROM api_catalog")
        conn.executemany(
            """INSERT OR IGNORE INTO api_catalog(name, url, description, category, auth, https, cors)
               VALUES (:name, :url, :description, :category, :auth, :https, :cors)""",
            rows,
        )
    return conn.execute("SELECT COUNT(*) FROM api_catalog").fetchone()[0]


def count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM api_catalog").fetchone()[0]


def find(conn: sqlite3.Connection, need: str, n: int = 8, include_keyed: bool = False) -> list[sqlite3.Row]:
    """Catalog entries matching `need`, best first; by default only no-key HTTPS ones, which
    call_api can reach."""
    any_word = build_fts_query(need)  # each term quoted, so FTS syntax in `need` can't misfire
    if not any_word:
        return []
    every_word = any_word.replace(" OR ", " AND ")
    where = "" if include_keyed else "AND a.auth = '' AND a.https = 1"
    sql = f"""SELECT a.*, bm25(api_catalog_fts, 10.0, 3.0, 2.0) AS score
              FROM api_catalog_fts JOIN api_catalog a ON a.id = api_catalog_fts.rowid
              WHERE api_catalog_fts MATCH ? {where}
              ORDER BY score LIMIT ?"""
    # APIs matching every word first ("flight status" -> flight APIs, not every status page),
    # then any word, to fill up.
    rows = conn.execute(sql, (every_word, n)).fetchall()
    if len(rows) < n and every_word != any_word:
        seen = {r["id"] for r in rows}
        rows += [r for r in conn.execute(sql, (any_word, n * 2)).fetchall() if r["id"] not in seen][: n - len(rows)]
    return rows


def format_found(need: str, rows: list[sqlite3.Row], total: int) -> str:
    if total == 0:
        return "The API catalog is empty. Run `zands apis update` to load the public-apis list."
    if not rows:
        return (
            f'No free, no-key API in the catalog matches "{need}". Try other words for the need '
            "(e.g. 'flight' -> 'aviation', 'stock' -> 'finance'), or say it isn't available."
        )
    lines = [f'APIs for "{need}" (free, no key, HTTPS; from the public-apis list):']
    for r in rows:
        lines.append(f"- {r['name']} [{r['category']}]: {r['description']}\n  docs: {r['url']}")
    lines.append(
        "Next: call_api on the docs URL to read how to use it, then call_api on the endpoint it "
        "describes. Prefer a built-in tool if one fits."
    )
    return "\n".join(lines)

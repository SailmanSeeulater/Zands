"""`zands ui`: Datasette over news.db, with sqlite-vec loaded and a few canned queries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from . import store

METADATA = {
    "title": "Zands news",
    "description": "Local news RAG database: articles, chunks, feed health, fact overrides and the retrieval log.",
    "databases": {
        "news": {
            "tables": {
                "articles": {"description": "One row per article; url is normalized and unique.", "sort_desc": "published_at"},
                "chunks": {"description": "Retrieval units (≤800 words) with copied article metadata."},
                "feeds": {"description": "Feed health: last success, last error, consecutive failures."},
                "facts": {"description": "Fact overrides put first in `zands ask` prompts while current."},
                "retrieval_log": {"description": "Every search/ask/eval retrieval, newest last.", "sort_desc": "id"},
            },
            "queries": {
                "recent_retrievals": {
                    "title": "Recent retrievals with their top hit",
                    "sql": (
                        "SELECT r.id, r.created_at, r.command, r.mode, r.question, r.query, "
                        "json_array_length(r.chunk_ids) AS hits, c.title AS top_title, c.source AS top_source "
                        "FROM retrieval_log r LEFT JOIN chunks c ON c.id = json_extract(r.chunk_ids, '$[0]') "
                        "ORDER BY r.id DESC LIMIT 100"
                    ),
                },
                "empty_retrievals": {
                    "title": "Questions that retrieved nothing",
                    "sql": "SELECT created_at, command, mode, question, query FROM retrieval_log "
                    "WHERE chunk_ids = '[]' ORDER BY id DESC",
                },
                "feed_health": {
                    "title": "Feed health",
                    "sql": (
                        "SELECT f.name, f.last_success_at, f.failure_count, f.last_error, f.last_new_items, "
                        "COUNT(a.id) AS articles, MAX(a.published_at) AS newest "
                        "FROM feeds f LEFT JOIN articles a ON a.feed_id = f.id GROUP BY f.id ORDER BY f.name"
                    ),
                },
                "articles_per_day": {
                    "title": "Articles per day",
                    "sql": "SELECT substr(published_at, 1, 10) AS day, COUNT(*) AS articles FROM articles "
                    "GROUP BY day ORDER BY day DESC LIMIT 60",
                },
                "current_facts": {
                    "title": "Current fact overrides",
                    "sql": "SELECT id, subject, attribute, value, valid_from, source_url FROM facts "
                    "WHERE valid_to IS NULL ORDER BY subject, attribute",
                },
            },
        }
    },
}


def build_command(db: str | Path, host: str = "127.0.0.1", port: int = 8001, metadata_path: Path | None = None) -> list[str]:
    db = Path(db).resolve()
    cmd = [sys.executable, "-m", "datasette", "serve", str(db), "--host", host, "--port", str(port)]
    ext = store.vec_extension_path()
    if ext:
        cmd += ["--load-extension", ext]
    if metadata_path:
        cmd += ["--metadata", str(metadata_path)]
    return cmd


def write_metadata(db: str | Path) -> Path:
    """Write the metadata next to the database (Datasette names the database after the file stem)."""
    db = Path(db).resolve()
    meta = json.loads(json.dumps(METADATA))
    meta["databases"] = {db.stem: METADATA["databases"]["news"]}
    path = db.parent / ".datasette-metadata.json"
    path.write_text(json.dumps(meta, indent=2))
    return path

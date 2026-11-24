"""`zands` command line: ingest, search, ask, stats."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import textwrap

from . import store
from .answer import DEFAULT_NUM_CTX, OllamaError, build_messages, default_model, default_ollama_url, stream_chat
from .ingest import load_feeds, run_ingest
from .search import DEFAULT_K, RECENCY_WEIGHT, build_fts_query, search


def default_db() -> str:
    return os.environ.get("ZANDS_DB", "data/news.db")


def default_feeds() -> str:
    return os.environ.get("ZANDS_FEEDS", "feeds.toml")


def cmd_ingest(args: argparse.Namespace) -> int:
    conn = store.connect(args.db)
    feeds = load_feeds(args.feeds)
    report = run_ingest(conn, feeds, concurrency=args.concurrency, limit=args.limit)
    print(f"\n{'feed':<28} {'status':<6} {'new':>4} {'skip':>5} {'rss':>4}")
    for f in report.feeds:
        status = "ok" if f.ok else "FAIL"
        print(f"{f.name[:28]:<28} {status:<6} {f.new:>4} {f.skipped:>5} {f.rss_fallback:>4}")
        if f.error:
            print(f"  └ {f.error[:200]}")
    s = store.stats(conn)
    print(
        f"\n{report.new} new articles in {report.seconds:.0f}s; "
        f"database now has {s['articles']} articles / {s['chunks']} chunks ({args.db})"
    )
    # Exit non-zero only if every feed failed, so a scheduler can alert on a dead network.
    return 1 if report.feeds and len(report.failed) == len(report.feeds) else 0


def _print_hits(hits) -> None:
    for i, h in enumerate(hits, 1):
        when = (h.published_at or "????-??-??")[:10]
        print(f"[{i}] {h.title}")
        print(f"    {when} · {h.source} · score {h.score:.2f} · chunk {h.chunk_id}")
        print(f"    {h.url}")
        print(textwrap.fill(h.snippet, width=100, initial_indent="    ", subsequent_indent="    "))
        print()


def cmd_search(args: argparse.Namespace) -> int:
    conn = store.connect(args.db)
    hits = search(conn, args.question, k=args.k, recency_weight=args.recency, command="search")
    query = build_fts_query(args.question)
    print(f"query: {query or '(no searchable terms)'}\n")
    if not hits:
        print("No matching chunks.")
        return 1
    _print_hits(hits)
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    conn = store.connect(args.db)
    hits = search(conn, args.question, k=args.k, recency_weight=args.recency, command="ask")
    if not hits:
        print("No articles in the local database match that question, so there is nothing to answer from.")
        print("Try different keywords, or run `zands ingest` to fetch fresh news.")
        return 1
    messages = build_messages(args.question, hits)
    if args.show_prompt:
        for m in messages:
            print(f"--- {m['role']} ---\n{m['content']}\n")
        print("--- answer ---")
    try:
        for piece in stream_chat(messages, model=args.model, base_url=args.ollama_url, num_ctx=args.num_ctx):
            sys.stdout.write(piece)
            sys.stdout.flush()
    except OllamaError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 2
    print("\n\nSources:")
    for i, h in enumerate(hits, 1):
        print(f"  [{i}] {(h.published_at or '????-??-??')[:10]} · {h.source} · {h.title}\n      {h.url}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    conn = store.connect(args.db)
    s = store.stats(conn)
    print(f"database:   {args.db}")
    print(f"articles:   {s['articles']}")
    print(f"chunks:     {s['chunks']}")
    print(f"retrievals: {s['retrievals']}")
    print(f"newest:     {s['newest']}")
    print(f"last fetch: {s['last_fetch']}\n")
    print(f"{'feed':<28} {'last success':<21} {'fails':>5}  last error")
    for f in s["feeds"]:
        err = (f["last_error"] or "")[:60] if f["failure_count"] else ""
        print(f"{f['name'][:28]:<28} {f['last_success_at'] or '-':<21} {f['failure_count']:>5}  {err}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="zands", description="Local news RAG: ingest feeds, search, ask a local model.")
    p.add_argument("--db", default=default_db(), help="SQLite database path (env ZANDS_DB, default data/news.db)")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("ingest", help="fetch feeds and store new articles")
    pi.add_argument("--feeds", default=default_feeds(), help="feeds file (env ZANDS_FEEDS, default feeds.toml)")
    pi.add_argument("--limit", type=int, default=None, help="max entries per feed")
    pi.add_argument("--concurrency", type=int, default=4, help="parallel page fetches (default 4)")
    pi.set_defaults(func=cmd_ingest)

    def retrieval_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("question")
        sp.add_argument("-k", type=int, default=DEFAULT_K, help=f"chunks to retrieve (default {DEFAULT_K})")
        sp.add_argument(
            "--recency", type=float, default=RECENCY_WEIGHT,
            help=f"recency boost weight, 0 to disable (default {RECENCY_WEIGHT})",
        )

    ps = sub.add_parser("search", help="show the chunks a question retrieves")
    retrieval_args(ps)
    ps.set_defaults(func=cmd_search)

    pa = sub.add_parser("ask", help="answer a question from the stored news with a local model")
    retrieval_args(pa)
    pa.add_argument("--model", default=default_model(), help="Ollama model (env ZANDS_MODEL, default qwen2.5:latest)")
    pa.add_argument("--ollama-url", default=default_ollama_url(), help="Ollama base URL (env OLLAMA_HOST)")
    pa.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX, help=f"model context window (default {DEFAULT_NUM_CTX})")
    pa.add_argument("--show-prompt", action="store_true", help="print the full prompt before the answer")
    pa.set_defaults(func=cmd_ask)

    pst = sub.add_parser("stats", help="database counts and feed health")
    pst.set_defaults(func=cmd_stats)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "charset_normalizer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    for chatty in ("trafilatura", "htmldate", "courlan"):  # warn about every odd page; not actionable
        logging.getLogger(chatty).setLevel(logging.ERROR)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())

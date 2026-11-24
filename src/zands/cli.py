"""`zands` command line: ingest, embed, search, ask, fact, eval, ui, schedule, stats."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import textwrap
from datetime import date

from . import facts as factlib
from . import schedule as sched
from . import store
from .answer import DEFAULT_NUM_CTX, OllamaError, build_messages, default_model, default_ollama_url, stream_chat
from .embed import EmbedError, OllamaEmbedder, default_embed_model, embed_pending, rebuild
from .ingest import load_feeds, run_ingest
from .search import DEFAULT_K, DEFAULT_MODE, MODES, RECENCY_WEIGHT, retrieve


def default_db() -> str:
    return os.environ.get("ZANDS_DB", "data/news.db")


def default_feeds() -> str:
    return os.environ.get("ZANDS_FEEDS", "feeds.toml")


def _embedder(args: argparse.Namespace) -> OllamaEmbedder:
    return OllamaEmbedder(model=args.embed_model, base_url=args.ollama_url)


# ---------------------------------------------------------------- ingest / embed


def cmd_ingest(args: argparse.Namespace) -> int:
    conn = store.connect(args.db)
    feeds = load_feeds(args.feeds)
    embedder = None if args.no_embed else _embedder(args)
    report = run_ingest(conn, feeds, concurrency=args.concurrency, limit=args.limit, embedder=embedder)
    print(f"\n{'feed':<28} {'status':<6} {'new':>4} {'skip':>5} {'rss':>4}")
    for f in report.feeds:
        status = "ok" if f.ok else "FAIL"
        print(f"{f.name[:28]:<28} {status:<6} {f.new:>4} {f.skipped:>5} {f.rss_fallback:>4}")
        if f.error:
            print(f"  └ {f.error[:200]}")
    s = store.stats(conn)
    print(
        f"\n{report.new} new articles in {report.seconds:.0f}s; "
        f"database now has {s['articles']} articles / {s['chunks']} chunks / {s['vectors'] or 0} vectors ({args.db})"
    )
    if embedder is not None:
        if report.embed_error:
            print(f"embedding skipped: {report.embed_error} (keyword search still works; next run retries)")
        else:
            print(f"embedded {report.embedded} new chunks")
    # Exit non-zero only if every feed failed, so a scheduler can alert on a dead network.
    return 1 if report.feeds and len(report.failed) == len(report.feeds) else 0


def cmd_embed(args: argparse.Namespace) -> int:
    conn = store.connect(args.db)
    if not conn.vec_enabled:
        print(f"error: sqlite-vec is not available: {conn.vec_error}", file=sys.stderr)
        return 2
    if args.rebuild:
        rebuild(conn)
        print("dropped all stored vectors")
    try:
        n = embed_pending(conn, _embedder(args), progress=True)
    except EmbedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"embedded {n} chunks; {store.vector_count(conn)} of {store.stats(conn)['chunks']} chunks have vectors")
    return 0


# ---------------------------------------------------------------- search / ask


def _retrieve(args: argparse.Namespace, conn, command: str):
    embedder = _embedder(args) if args.mode != "keyword" else None
    r = retrieve(conn, args.question, k=args.k, mode=args.mode, embedder=embedder, recency_weight=args.recency, command=command)
    if r.note:
        print(f"note: {r.note}; using keyword search", file=sys.stderr)
    return r


def _print_hits(hits) -> None:
    for i, h in enumerate(hits, 1):
        when = (h.published_at or "????-??-??")[:10]
        ranks = []
        if h.keyword_rank:
            ranks.append(f"kw #{h.keyword_rank}")
        if h.vector_rank:
            ranks.append(f"vec #{h.vector_rank}")
        print(f"[{i}] {h.title}")
        print(f"    {when} · {h.source} · score {h.score:.4g} ({', '.join(ranks)}) · chunk {h.chunk_id}")
        print(f"    {h.url}")
        print(textwrap.fill(h.snippet, width=100, initial_indent="    ", subsequent_indent="    "))
        print()


def cmd_search(args: argparse.Namespace) -> int:
    conn = store.connect(args.db)
    r = _retrieve(args, conn, "search")
    print(f"mode: {r.mode} · query: {r.query or '(no searchable terms)'}\n")
    if not r.hits:
        print("No matching chunks.")
        return 1
    _print_hits(r.hits)
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    conn = store.connect(args.db)
    r = _retrieve(args, conn, "ask")
    relevant = [] if args.no_facts else factlib.relevant_facts(conn, args.question)
    if r.log_id is not None:
        store.set_log_facts(conn, r.log_id, [f.id for f in relevant])
    if not r.hits and not relevant:
        print("No articles or facts in the local database match that question, so there is nothing to answer from.")
        print("Try different keywords, or run `zands ingest` to fetch fresh news.")
        return 1
    messages = build_messages(args.question, r.hits, facts=relevant)
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
    print("\n")
    if relevant:
        print("Verified facts:")
        for i, f in enumerate(relevant, 1):
            print(f"  [F{i}] {f.subject} | {f.attribute}: {f.value} (since {f.valid_from})")
    if r.hits:
        print(f"Sources ({r.mode}):")
        for i, h in enumerate(r.hits, 1):
            print(f"  [{i}] {(h.published_at or '????-??-??')[:10]} · {h.source} · {h.title}\n      {h.url}")
    return 0


# ---------------------------------------------------------------- facts


def cmd_fact_add(args: argparse.Namespace) -> int:
    conn = store.connect(args.db)
    try:
        fid = factlib.add_fact(
            conn, args.subject, args.attribute, args.value,
            valid_from=args.valid_from, source_url=args.source, supersede=not args.keep_previous,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"added fact {fid}: {args.subject} | {args.attribute}: {args.value}")
    return 0


def cmd_fact_list(args: argparse.Namespace) -> int:
    conn = store.connect(args.db)
    rows = factlib.list_facts(conn, include_expired=args.all)
    if not rows:
        print("no facts" if args.all else "no current facts (use --all to include expired ones)")
        return 0
    print(f"{'id':>4}  {'subject':<20} {'attribute':<16} {'value':<28} {'valid':<23} source")
    for f in rows:
        valid = f"{f.valid_from}..{f.valid_to or ''}"
        print(f"{f.id:>4}  {f.subject[:20]:<20} {f.attribute[:16]:<16} {f.value[:28]:<28} {valid:<23} {f.source_url or ''}")
    return 0


def cmd_fact_expire(args: argparse.Namespace) -> int:
    conn = store.connect(args.db)
    try:
        ok = factlib.expire_fact(conn, args.id, on=args.on)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not ok:
        print(f"no current fact with id {args.id}", file=sys.stderr)
        return 1
    print(f"expired fact {args.id} as of {args.on or date.today().isoformat()}")
    return 0


# ---------------------------------------------------------------- eval


def cmd_eval(args: argparse.Namespace) -> int:
    from .evaluate import load_questions, run_mode

    conn = store.connect(args.db)
    questions = load_questions(args.questions)
    modes = MODES[:2] if args.mode == "compare" else (args.mode,)
    embedder = _embedder(args)

    answer_fn = None
    if args.answers:
        def answer_fn(question, hits):
            return "".join(stream_chat(build_messages(question, hits), model=args.model, base_url=args.ollama_url))

    reports = []
    for mode in modes:
        rep = run_mode(conn, questions, mode, k=args.k, embedder=embedder,
                       answer_fn=answer_fn if (args.answers and mode == modes[0]) else None)
        reports.append(rep)

    if args.verbose_results:
        for rep in reports:
            print(f"\n== {rep.mode}")
            for r in rep.results:
                status = "missing" if r.missing else (f"#{r.rank}" if r.rank else "MISS")
                ans = "" if r.answer_ok is None else ("  answer ok" if r.answer_ok else "  answer WRONG")
                print(f"  {status:>7}  [{r.question.style[:4]}] {r.question.q}{ans}")

    scored = len(reports[0].scored)
    missing = len(questions) - scored
    print(f"\n{len(questions)} questions, {scored} scored" + (f", {missing} missing from the database" if missing else ""))
    print(f"\n{'mode':<20} {'hit@1':>6} {'hit@3':>6} {'hit@' + str(args.k):>6} {'MRR':>6}  {'kw-style':>8} {'para':>6}  fallbacks")
    for rep in reports:
        print(
            f"{rep.mode:<20} {rep.hit_rate(1):>6.0%} {rep.hit_rate(3):>6.0%} {rep.hit_rate():>6.0%} {rep.mrr:>6.2f}"
            f"  {rep.hit_rate(style='keyword'):>8.0%} {rep.hit_rate(style='paraphrase'):>6.0%}  {rep.fallbacks}"
        )
    for rep in reports:
        if rep.answer_rate is not None:
            print(f"\nanswer check ({rep.mode}): {rep.answer_rate:.0%} of answers contain an expected keyword")
    return 0


# ---------------------------------------------------------------- ui / schedule / stats


def cmd_ui(args: argparse.Namespace) -> int:
    from .ui import build_command, write_metadata

    conn = store.connect(args.db)  # creates/migrates the schema before Datasette opens it
    conn.close()
    cmd = build_command(args.db, host=args.host, port=args.port, metadata_path=write_metadata(args.db))
    print(f"Datasette on http://{args.host}:{args.port}/ (Ctrl-C to stop)")
    try:
        return subprocess.run(cmd).returncode
    except KeyboardInterrupt:
        return 0


def cmd_schedule_install(args: argparse.Namespace) -> int:
    plist = sched.build_plist(args.db, args.feeds, interval_minutes=args.interval)
    path = sched.write_plist(plist)
    print(f"wrote {path}")
    print(f"  runs: {' '.join(plist['ProgramArguments'])}")
    print(f"  every {args.interval} min (and once when loaded), logs to {plist['StandardOutPath']}")
    if args.no_load:
        print(f"not loaded; load it with `zands schedule install` (or launchctl bootstrap {sched.domain()} {path})")
        return 0
    try:
        sched.load(path)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"loaded into launchd as {sched.LABEL}; the first ingest is running now")
    return 0


def cmd_schedule_uninstall(args: argparse.Namespace) -> int:
    print("removed the launchd agent" if sched.uninstall() else "nothing to remove")
    return 0


def cmd_schedule_status(args: argparse.Namespace) -> int:
    st = sched.status()
    print(f"plist:     {sched.plist_path()} ({'present' if st.installed else 'not installed'})")
    print(f"launchd:   {'loaded' if st.loaded else 'not loaded'}" + (f", state {st.state}" if st.state else ""))
    if st.interval_minutes:
        print(f"interval:  every {st.interval_minutes} min")
    if st.loaded:
        print(f"runs:      {st.runs if st.runs is not None else '?'} since load, last exit code {st.last_exit or '?'}")
    conn = store.connect(args.db)
    print(f"last article fetched: {store.stats(conn)['last_fetch'] or 'never'}")
    if st.log_tail:
        print(f"\nlast lines of {st.log}:")
        for line in st.log_tail:
            print(f"  {line}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    conn = store.connect(args.db)
    s = store.stats(conn)
    print(f"database:   {args.db}")
    print(f"articles:   {s['articles']}")
    print(f"chunks:     {s['chunks']}")
    vec = s["vectors"] if s["vectors"] is not None else f"n/a ({conn.vec_error})"
    print(f"vectors:    {vec}")
    print(f"facts:      {s['facts']} current")
    print(f"retrievals: {s['retrievals']}")
    print(f"newest:     {s['newest']}")
    print(f"last fetch: {s['last_fetch']}\n")
    print(f"{'feed':<28} {'last success':<21} {'fails':>5}  last error")
    for f in s["feeds"]:
        err = (f["last_error"] or "")[:60] if f["failure_count"] else ""
        print(f"{f['name'][:28]:<28} {f['last_success_at'] or '-':<21} {f['failure_count']:>5}  {err}")
    return 0


# ---------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="zands", description="Local news RAG: ingest feeds, search, ask a local model.")
    p.add_argument("--db", default=default_db(), help="SQLite database path (env ZANDS_DB, default data/news.db)")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="command", required=True)

    def ollama_args(sp: argparse.ArgumentParser, chat: bool = False) -> None:
        sp.add_argument("--ollama-url", default=default_ollama_url(), help="Ollama base URL (env OLLAMA_HOST)")
        sp.add_argument("--embed-model", default=default_embed_model(),
                        help="embedding model (env ZANDS_EMBED_MODEL, default nomic-embed-text)")
        if chat:
            sp.add_argument("--model", default=default_model(), help="Ollama model (env ZANDS_MODEL, default qwen2.5:latest)")

    pi = sub.add_parser("ingest", help="fetch feeds, store new articles, embed new chunks")
    pi.add_argument("--feeds", default=default_feeds(), help="feeds file (env ZANDS_FEEDS, default feeds.toml)")
    pi.add_argument("--limit", type=int, default=None, help="max entries per feed")
    pi.add_argument("--concurrency", type=int, default=4, help="parallel page fetches (default 4)")
    pi.add_argument("--no-embed", action="store_true", help="skip embedding new chunks")
    ollama_args(pi)
    pi.set_defaults(func=cmd_ingest)

    pe = sub.add_parser("embed", help="embed chunks that have no vector yet (backfill)")
    pe.add_argument("--rebuild", action="store_true", help="drop all vectors first (after changing embedding model)")
    ollama_args(pe)
    pe.set_defaults(func=cmd_embed)

    def retrieval_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("question")
        sp.add_argument("-k", type=int, default=DEFAULT_K, help=f"chunks to retrieve (default {DEFAULT_K})")
        sp.add_argument("--mode", choices=MODES, default=DEFAULT_MODE, help=f"retrieval mode (default {DEFAULT_MODE})")
        sp.add_argument(
            "--recency", type=float, default=RECENCY_WEIGHT,
            help=f"recency boost weight, 0 to disable (default {RECENCY_WEIGHT})",
        )

    ps = sub.add_parser("search", help="show the chunks a question retrieves")
    retrieval_args(ps)
    ollama_args(ps)
    ps.set_defaults(func=cmd_search)

    pa = sub.add_parser("ask", help="answer a question from the stored news with a local model")
    retrieval_args(pa)
    ollama_args(pa, chat=True)
    pa.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX, help=f"model context window (default {DEFAULT_NUM_CTX})")
    pa.add_argument("--no-facts", action="store_true", help="leave fact overrides out of the prompt")
    pa.add_argument("--show-prompt", action="store_true", help="print the full prompt before the answer")
    pa.set_defaults(func=cmd_ask)

    pf = sub.add_parser("fact", help="manage fact overrides")
    fsub = pf.add_subparsers(dest="fact_command", required=True)
    fa = fsub.add_parser("add", help="add a fact (closes the current one with the same subject + attribute)")
    fa.add_argument("subject", help='e.g. "Nvidia"')
    fa.add_argument("attribute", help='e.g. "CEO"')
    fa.add_argument("value", help='e.g. "Jensen Huang"')
    fa.add_argument("--from", dest="valid_from", default=None, help="valid from YYYY-MM-DD (default today)")
    fa.add_argument("--source", default=None, help="source URL")
    fa.add_argument("--keep-previous", action="store_true", help="don't close existing facts for this subject + attribute")
    fa.set_defaults(func=cmd_fact_add)
    fl = fsub.add_parser("list", help="list current facts")
    fl.add_argument("--all", action="store_true", help="include expired facts")
    fl.set_defaults(func=cmd_fact_list)
    fe = fsub.add_parser("expire", help="mark a fact as no longer true")
    fe.add_argument("id", type=int)
    fe.add_argument("--on", default=None, help="no longer true from YYYY-MM-DD (default today)")
    fe.set_defaults(func=cmd_fact_expire)

    pv = sub.add_parser("eval", help="retrieval hit rate on the known-answer question set")
    pv.add_argument("--questions", default="eval/questions.toml", help="question file (default eval/questions.toml)")
    pv.add_argument("--mode", choices=(*MODES, "compare"), default="compare",
                    help="mode to score, or 'compare' for keyword vs hybrid (default)")
    pv.add_argument("-k", type=int, default=DEFAULT_K, help=f"chunks retrieved per question (default {DEFAULT_K})")
    pv.add_argument("--answers", action="store_true", help="also ask the model and check answer keywords (slow)")
    pv.add_argument("--list", dest="verbose_results", action="store_true", help="show per-question results")
    ollama_args(pv, chat=True)
    pv.set_defaults(func=cmd_eval)

    pu = sub.add_parser("ui", help="browse the database in Datasette")
    pu.add_argument("--host", default="127.0.0.1")
    pu.add_argument("--port", type=int, default=8001)
    pu.set_defaults(func=cmd_ui)

    psc = sub.add_parser("schedule", help="hourly ingest via launchd (macOS)")
    ssub = psc.add_subparsers(dest="schedule_command", required=True)
    si = ssub.add_parser("install", help="write the launchd agent and load it")
    si.add_argument("--feeds", default=default_feeds(), help="feeds file (default feeds.toml)")
    si.add_argument("--interval", type=int, default=sched.DEFAULT_INTERVAL_MINUTES, help="minutes between runs (default 60)")
    si.add_argument("--no-load", action="store_true", help="write the plist but don't load it")
    si.set_defaults(func=cmd_schedule_install)
    su = ssub.add_parser("uninstall", help="unload and delete the launchd agent")
    su.set_defaults(func=cmd_schedule_uninstall)
    st = ssub.add_parser("status", help="is it loaded, last exit code, recent log lines")
    st.set_defaults(func=cmd_schedule_status)

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

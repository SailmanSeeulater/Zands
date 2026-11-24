"""`zands` command line: ingest, embed, search, ask, fact, digest, eval, ui, schedule, mcp, stats."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import textwrap
from datetime import date

from . import digest as digestlib
from . import facts as factlib
from . import schedule as sched
from . import store
from .answer import DEFAULT_NUM_CTX, OllamaError, build_messages, default_model, default_ollama_url, stream_chat
from .embed import EmbedError, OllamaEmbedder, default_embed_model, embed_pending, rebuild
from .ingest import load_feeds, run_ingest
from .route import ROUTES, default_router_model
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
    try:
        d = digestlib.refresh(conn)
        print(f"digest: {len(d.stories)} top stories from {d.article_count} articles in the last {d.hours}h")
    except Exception as exc:  # noqa: BLE001 - the digest is a nice-to-have
        print(f"digest not rebuilt: {exc}")
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
    r = retrieve(conn, args.question, k=args.k, mode=args.mode, embedder=embedder, recency_weight=args.recency,
                 command=command, reranker=_reranker(args))
    if r.note:
        print(f"note: {r.note}; using keyword search", file=sys.stderr)
    return r


def _reranker(args: argparse.Namespace):
    if not getattr(args, "rerank", False):
        return None
    from .rerank import LLMReranker

    return LLMReranker(model=getattr(args, "model", None), base_url=args.ollama_url)


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


def _print_sources(hits, prefix: str = "") -> None:
    """One line per article; chunks of the same article share it, e.g. [1,2]."""
    groups: dict[str, list[int]] = {}
    first: dict[str, object] = {}
    for i, h in enumerate(hits, 1):
        groups.setdefault(h.url, []).append(i)
        first.setdefault(h.url, h)
    for url, nums in groups.items():
        h = first[url]
        label = ",".join(f"{prefix}{n}" for n in nums)
        when = (h.published_at or "")[:10]
        print(f"  [{label}] {when + ' · ' if when else ''}{h.source} · {h.title}\n      {url}")


def cmd_ask(args: argparse.Namespace) -> int:
    from .ask import plan

    conn = store.connect(args.db)
    p = plan(
        conn, args.question, k=args.k, mode=args.mode,
        embedder=_embedder(args) if args.mode != "keyword" else None,
        recency_weight=args.recency, use_router=not args.no_router,
        forced_route=None if args.route == "auto" else args.route,
        router_model=args.router_model, ollama_url=args.ollama_url,
        web_enabled=not args.no_web, web_save=args.save_web,
        use_facts=not args.no_facts, use_digest=not args.no_digest,
    )
    print(f"route: {p.route} ({p.decision.reason})", file=sys.stderr)
    if p.retrieval and p.retrieval.note:
        print(f"note: {p.retrieval.note}; using keyword search", file=sys.stderr)
    if p.web and p.web.error:
        print(f"note: web fallback: {p.web.error}", file=sys.stderr)
    if p.messages is None:
        print(p.reply)
        return 1
    if args.show_prompt:
        for m in p.messages:
            print(f"--- {m['role']} ---\n{m['content']}\n")
        print("--- answer ---")
    try:
        for piece in stream_chat(p.messages, model=args.model, base_url=args.ollama_url, num_ctx=args.num_ctx):
            sys.stdout.write(piece)
            sys.stdout.flush()
    except OllamaError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 2
    print("\n")
    if p.facts:
        print("Verified facts:")
        for i, f in enumerate(p.facts, 1):
            print(f"  [F{i}] {f.subject} | {f.attribute}: {f.value} (since {f.valid_from})")
    if p.hits:
        print(f"Sources ({p.retrieval.mode}):")
        _print_sources(p.hits)
    if p.web_hits:
        print("Web results (SearXNG):" + (f" saved {p.web.saved} pages to the index" if p.web.saved else ""))
        _print_sources(p.web_hits, prefix="W")
    if p.digest and p.route != "knowledge" and not (p.hits or p.web_hits):
        print("(headlines from today's digest were in the prompt)")
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
    from .evaluate import load_questions, record_mode_report, run_mode

    if args.routes:
        return _eval_routes(args)
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
                       answer_fn=answer_fn if (args.answers and mode == modes[0]) else None,
                       reranker=_reranker(args) if mode != "keyword" else None)
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
    for rep in reports:  # kept for the dashboard's eval history
        record_mode_report(conn, rep, len(questions), rerank=bool(args.rerank and rep.mode != "keyword"))
    return 0


def _eval_routes(args: argparse.Namespace) -> int:
    from .evaluate import load_routes, record_route_report, run_routes

    cases = load_routes(args.routes_file)
    rep = run_routes(cases, use_model=not args.no_router, model=args.router_model, base_url=args.ollama_url)
    record_route_report(store.connect(args.db), rep, use_model=not args.no_router)
    for r in rep.results:
        mark = "ok " if r.ok else "BAD"
        print(f"  {mark} expect {r.case.expect:<9} got {r.got.route:<9} [{r.got.method}] {r.case.q}")
    print(f"\nrouter: {rep.correct}/{len(rep.results)} correct ({rep.accuracy:.0%}); "
          f"{rep.by_method.get('rule', 0)} by rule, {rep.by_method.get('model', 0)} by model, "
          f"{rep.by_method.get('default', 0)} defaulted")
    return 0


# ---------------------------------------------------------------- digest / mcp


def cmd_digest(args: argparse.Namespace) -> int:
    conn = store.connect(args.db)
    if args.refresh or digestlib.latest(conn) is None:
        print(digestlib.refresh(conn, hours=args.hours).text)
    else:
        print(digestlib.latest(conn)["text"])
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from .mcp_server import run

    if not os.path.isabs(args.db):
        print(f"note: --db {args.db!r} is relative to {os.getcwd()}; MCP clients should pass an absolute path",
              file=sys.stderr)
    run(args.db)
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


def cmd_dashboard(args: argparse.Namespace) -> int:
    from .dashboard import make_server

    store.connect(args.db).close()  # creates/migrates the schema; the server itself only reads
    try:
        server = make_server(args.db, host=args.host, port=args.port)
    except OSError as exc:
        print(f"error: can't listen on {args.host}:{args.port}: {exc}", file=sys.stderr)
        return 2
    host, port = server.server_address[:2]
    print(f"Zands dashboard on http://{host}:{port}/ (read-only over {args.db}; Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
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
    d = digestlib.latest(conn, max_age_hours=None)
    print(f"digest:     {d['created_at'] if d else 'never built'}")
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
    ps.add_argument("--rerank", action="store_true", help="rerank candidates with the chat model (slow; see README)")
    ps.add_argument("--model", default=default_model(), help="model used by --rerank")
    ps.set_defaults(func=cmd_search)

    pa = sub.add_parser("ask", help="answer a question from the stored news with a local model")
    retrieval_args(pa)
    ollama_args(pa, chat=True)
    pa.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX, help=f"model context window (default {DEFAULT_NUM_CTX})")
    pa.add_argument("--no-facts", action="store_true", help="leave fact overrides out of the prompt")
    pa.add_argument("--no-digest", action="store_true", help="leave today's headlines out of the prompt")
    pa.add_argument("--route", choices=("auto", *[r for r in ROUTES if r != "none"]), default="auto",
                    help="force a route instead of letting the router decide (default auto)")
    pa.add_argument("--no-router", action="store_true", help="rules only: don't ask the router model")
    pa.add_argument("--router-model", default=default_router_model(),
                    help="router model (env ZANDS_ROUTER_MODEL, default qwen2.5:latest)")
    pa.add_argument("--no-web", action="store_true", help="never use the SearXNG web fallback")
    pa.add_argument("--save-web", action="store_true", help="store fetched web pages in the index")
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
    pv.add_argument("--rerank", action="store_true", help="rerank hybrid/vector candidates with the chat model")
    pv.add_argument("--routes", action="store_true", help="score the router on eval/routes.toml instead")
    pv.add_argument("--routes-file", default="eval/routes.toml", help="router test set (default eval/routes.toml)")
    pv.add_argument("--no-router", action="store_true", help="with --routes: rules only")
    pv.add_argument("--router-model", default=default_router_model(), help="with --routes: router model")
    ollama_args(pv, chat=True)
    pv.set_defaults(func=cmd_eval)

    pdb = sub.add_parser("dashboard", help="charts over the database in the browser (read-only)")
    pdb.add_argument("--host", default="127.0.0.1")
    pdb.add_argument("--port", type=int, default=8002)
    pdb.set_defaults(func=cmd_dashboard)

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

    pd = sub.add_parser("digest", help="show today's top stories (rebuilt after every ingest)")
    pd.add_argument("--refresh", action="store_true", help="rebuild it now")
    pd.add_argument("--hours", type=int, default=digestlib.DEFAULT_HOURS, help="look-back window (default 24)")
    pd.set_defaults(func=cmd_digest)

    pm = sub.add_parser("mcp", help="run the MCP server on stdio (search_news, latest_digest)")
    pm.set_defaults(func=cmd_mcp)

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

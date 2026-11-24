"""`zands eval`: retrieval hit rate (and optional answer checks) on a known-answer question set."""

from __future__ import annotations

import sqlite3
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from . import store
from .search import retrieve

DEFAULT_QUESTIONS = "eval/questions.toml"


@dataclass
class Question:
    q: str
    expect_url: list[str]
    answer_any: list[str] = field(default_factory=list)
    style: str = ""


@dataclass
class QuestionResult:
    question: Question
    missing: bool = False  # none of the expected articles is in the database
    rank: int | None = None  # 1-based rank of the first hit from an expected article
    mode: str | None = None
    answer: str | None = None
    answer_ok: bool | None = None


@dataclass
class ModeReport:
    mode: str
    k: int
    results: list[QuestionResult]

    @property
    def scored(self) -> list[QuestionResult]:
        return [r for r in self.results if not r.missing]

    def hit_rate(self, at: int | None = None, style: str | None = None) -> float:
        at = at or self.k
        rs = [r for r in self.scored if style is None or r.question.style == style]
        return sum(1 for r in rs if r.rank and r.rank <= at) / len(rs) if rs else 0.0

    @property
    def mrr(self) -> float:
        rs = self.scored
        return sum(1 / r.rank for r in rs if r.rank) / len(rs) if rs else 0.0

    @property
    def fallbacks(self) -> int:
        return sum(1 for r in self.results if r.mode and "fallback" in r.mode)

    @property
    def answer_rate(self) -> float | None:
        checked = [r for r in self.scored if r.answer_ok is not None]
        return sum(r.answer_ok for r in checked) / len(checked) if checked else None


def load_questions(path: str | Path) -> list[Question]:
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    out = []
    for item in data.get("question", []):
        urls = item["expect_url"]
        out.append(
            Question(
                q=item["q"],
                expect_url=[urls] if isinstance(urls, str) else list(urls),
                answer_any=list(item.get("answer_any", [])),
                style=item.get("style", ""),
            )
        )
    return out


def _in_db(conn: sqlite3.Connection, patterns: list[str]) -> bool:
    return any(
        conn.execute("SELECT 1 FROM articles WHERE instr(url, ?) > 0 LIMIT 1", (p,)).fetchone() for p in patterns
    )


def answer_matches(answer: str, expected: list[str]) -> bool:
    low = answer.lower()
    return any(e.lower() in low for e in expected)


def run_mode(
    conn: sqlite3.Connection,
    questions: list[Question],
    mode: str,
    k: int = 8,
    embedder=None,
    answer_fn=None,
    reranker=None,
) -> ModeReport:
    """Score one retrieval mode. answer_fn(question, hits) -> str enables answer checks."""
    results = []
    for q in questions:
        res = QuestionResult(question=q)
        if not _in_db(conn, q.expect_url):
            res.missing = True
            results.append(res)
            continue
        r = retrieve(conn, q.q, k=k, mode=mode, embedder=embedder, command="eval", reranker=reranker)
        res.mode = r.mode
        for i, h in enumerate(r.hits, 1):
            if any(p in h.url for p in q.expect_url):
                res.rank = i
                break
        if answer_fn and q.answer_any:
            res.answer = answer_fn(q.q, r.hits)
            res.answer_ok = answer_matches(res.answer, q.answer_any)
        results.append(res)
    return ModeReport(mode=mode, k=k, results=results)


# ---------------------------------------------------------------- router


@dataclass
class RouteCase:
    q: str
    expect: str
    accept: list[str] = field(default_factory=list)


@dataclass
class RouteResult:
    case: RouteCase
    got: object  # route.RouteDecision

    @property
    def ok(self) -> bool:
        return self.got.route == self.case.expect or self.got.route in self.case.accept


@dataclass
class RouteReport:
    results: list[RouteResult]

    @property
    def correct(self) -> int:
        return sum(r.ok for r in self.results)

    @property
    def accuracy(self) -> float:
        return self.correct / len(self.results) if self.results else 0.0

    @property
    def by_method(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.results:
            out[r.got.method] = out.get(r.got.method, 0) + 1
        return out


def load_routes(path: str | Path) -> list[RouteCase]:
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    return [RouteCase(q=r["q"], expect=r["expect"], accept=list(r.get("accept", []))) for r in data.get("route", [])]


def run_routes(cases: list[RouteCase], use_model: bool = True, model: str | None = None, base_url: str | None = None,
               classifier=None) -> RouteReport:
    """Router decisions before the retrieval gate (the gate needs a database and is tested separately)."""
    from .route import route

    return RouteReport([
        RouteResult(c, route(c.q, use_model=use_model, model=model, base_url=base_url, classifier=classifier))
        for c in cases
    ])


def record_mode_report(conn: sqlite3.Connection, rep: ModeReport, questions: int, rerank: bool = False) -> int:
    """Keep one retrieval score in eval_runs (the dashboard's eval history)."""
    return store.record_eval_run(
        conn, "retrieval", questions, mode=rep.mode, k=rep.k, scored=len(rep.scored),
        hit1=rep.hit_rate(1), hit3=rep.hit_rate(3), hitk=rep.hit_rate(), mrr=rep.mrr,
        fallbacks=rep.fallbacks, answer_rate=rep.answer_rate, rerank=int(rerank),
    )


def record_route_report(conn: sqlite3.Connection, rep: RouteReport, use_model: bool = True) -> int:
    return store.record_eval_run(
        conn, "routes", len(rep.results), mode="rules+model" if use_model else "rules",
        scored=rep.correct, accuracy=rep.accuracy,
    )

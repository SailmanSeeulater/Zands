"""Fact overrides: small, dated, hand-curated facts that `zands ask` puts first in the prompt
as authoritative corrections to the model's stale training knowledge.

A fact is (subject, attribute, value) with a validity window [valid_from, valid_to).
Adding a new value for the same subject + attribute closes the previous one, so
history is kept ("Nvidia / CEO" has one current row and any number of past ones).

Relevance is deliberately simple and predictable: a fact is relevant to a
question when every word of its subject appears in the question (plural and
possessive forms tolerated). Write subjects the way you'd ask about them.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import date

from . import store

MAX_FACTS_IN_PROMPT = 10

_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass
class Fact:
    id: int
    subject: str
    attribute: str
    value: str
    valid_from: str
    valid_to: str | None
    source_url: str | None
    added_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Fact":
        return cls(**{k: row[k] for k in row.keys()})


def parse_day(value: str | date | None) -> str:
    """YYYY-MM-DD string for a date, an ISO string, or today (None)."""
    if value is None:
        return date.today().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return date.fromisoformat(value).isoformat()  # raises ValueError on junk


def _norm(word: str) -> str:
    word = word.lower()
    if word.endswith("'s") or word.endswith("’s"):
        word = word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return word


def _words(text: str) -> set[str]:
    return {_norm(w) for w in _WORD.findall(text.replace("’", "'").replace("'s", ""))}


def add_fact(
    conn: sqlite3.Connection,
    subject: str,
    attribute: str,
    value: str,
    valid_from: str | date | None = None,
    source_url: str | None = None,
    supersede: bool = True,
) -> int:
    """Add a fact. With supersede=True (default) any current fact with the same
    subject + attribute (case-insensitive) that started on or before this one
    is closed as of this fact's valid_from."""
    subject, attribute, value = subject.strip(), attribute.strip(), value.strip()
    if not (subject and attribute and value):
        raise ValueError("subject, attribute and value must all be non-empty")
    start = parse_day(valid_from)
    with conn:
        if supersede:
            conn.execute(
                """UPDATE facts SET valid_to = ?
                   WHERE subject = ? COLLATE NOCASE AND attribute = ? COLLATE NOCASE
                     AND valid_to IS NULL AND valid_from <= ?""",
                (start, subject, attribute, start),
            )
        cur = conn.execute(
            """INSERT INTO facts(subject, attribute, value, valid_from, valid_to, source_url, added_at)
               VALUES (?, ?, ?, ?, NULL, ?, ?)""",
            (subject, attribute, value, start, source_url, store.utcnow()),
        )
    return cur.lastrowid


def expire_fact(conn: sqlite3.Connection, fact_id: int, on: str | date | None = None) -> bool:
    """Close a fact as of `on` (default today). Returns False if no such current fact."""
    with conn:
        cur = conn.execute(
            "UPDATE facts SET valid_to = ? WHERE id = ? AND valid_to IS NULL", (parse_day(on), fact_id)
        )
    return cur.rowcount == 1


def list_facts(conn: sqlite3.Connection, include_expired: bool = False, on: str | date | None = None) -> list[Fact]:
    if include_expired:
        rows = conn.execute("SELECT * FROM facts ORDER BY subject COLLATE NOCASE, attribute COLLATE NOCASE, valid_from")
    else:
        rows = conn.execute(
            """SELECT * FROM facts
               WHERE valid_from <= :d AND (valid_to IS NULL OR valid_to > :d)
               ORDER BY subject COLLATE NOCASE, attribute COLLATE NOCASE""",
            {"d": parse_day(on)},
        )
    return [Fact.from_row(r) for r in rows]


def relevant_facts(
    conn: sqlite3.Connection, question: str, on: str | date | None = None, limit: int = MAX_FACTS_IN_PROMPT
) -> list[Fact]:
    """Facts current on `on` whose whole subject appears in the question,
    those that also mention the attribute first."""
    qwords = _words(question)
    scored = []
    for fact in list_facts(conn, on=on):
        subject_words = _words(fact.subject)
        if not subject_words or not subject_words <= qwords:
            continue
        attribute_overlap = len(_words(fact.attribute) & qwords)
        scored.append((attribute_overlap, len(subject_words), fact))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [f for _, _, f in scored[:limit]]

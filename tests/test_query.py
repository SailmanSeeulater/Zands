import sqlite3

import pytest

from zands.search import build_fts_query, query_terms


def test_stopwords_and_question_words_dropped():
    assert query_terms("What is the latest news about the Fed raising interest rates today?") == [
        "fed", "raising", "interest", "rates",
    ]


def test_terms_deduped_and_lowercased():
    assert query_terms("Tesla tesla TESLA earnings") == ["tesla", "earnings"]


def test_or_query_with_quoted_terms():
    assert build_fts_query("Ukraine ceasefire talks") == '"ukraine" OR "ceasefire" OR "talks"'


def test_empty_or_all_stopwords_returns_none():
    assert build_fts_query("") is None
    assert build_fts_query("what is the latest?") is None
    assert build_fts_query("?!*()\"") is None


@pytest.mark.parametrize(
    "question",
    [
        'climate NEAR(summit) AND "COP" OR NOT* -x',
        "title:openai body:gpt",
        '"unbalanced quote',
        "^start (paren",
        "rate* hike* OR*",
        "C++ vs Rust?",
        "émigré café naïve 2026",
        "SELECT * FROM chunks; DROP TABLE chunks;--",
        "ＡＩ 规则 Москва",
    ],
)
def test_hostile_input_yields_valid_fts_query(question):
    query = build_fts_query(question)
    assert query is not None
    for term in query.split(" OR "):
        assert term.startswith('"') and term.endswith('"')
        assert '"' not in term[1:-1]
    # It must actually parse in FTS5.
    db = sqlite3.connect(":memory:")
    db.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
    db.execute("INSERT INTO t VALUES ('climate summit near cop openai rust 2026')")
    db.execute("SELECT * FROM t WHERE t MATCH ?", (query,)).fetchall()


def test_fts_keywords_become_plain_terms():
    # AND/OR/NOT/NEAR are kept only as quoted literals (or dropped as stopwords)
    q = build_fts_query("NEAR OR NOT AND")
    assert q in (None, '"near"')

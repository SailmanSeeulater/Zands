import json

import pytest

from zands import evaluate, store

from fakes import FakeEmbedder
from zands.embed import embed_pending


@pytest.fixture
def db(conn):
    docs = [
        ("https://n.example/volcano-erupts", "Iceland evacuates town", "Lava flowed after the eruption near Grindavik."),
        ("https://n.example/ecb-holds", "European Central Bank holds", "The bank kept interest rates unchanged."),
        ("https://n.example/cup-final", "Cup final tonight", "The cup final kicks off tonight."),
    ]
    for url, title, body in docs:
        store.insert_article(conn, store.Article(url=url, title=title, source="N", body=body, extraction="rss",
                                                 published_at="2026-11-24T09:00:00Z"), [body])
    embed_pending(conn, FakeEmbedder())
    return conn


QUESTIONS = [
    evaluate.Question(q="Is there an eruption in Iceland?", expect_url=["volcano-erupts"], answer_any=["lava"], style="keyword"),
    evaluate.Question(q="magma activity", expect_url=["volcano-erupts"], style="paraphrase"),
    evaluate.Question(q="What did the ECB do with borrowing costs?", expect_url=["ecb-holds"], style="paraphrase"),
    evaluate.Question(q="Who won the election?", expect_url=["not-ingested"], style="keyword"),
]


def test_load_questions(tmp_path):
    p = tmp_path / "q.toml"
    p.write_text('[[question]]\nq = "x?"\nexpect_url = "a"\n\n[[question]]\nq = "y?"\nexpect_url = ["b", "c"]\nanswer_any = ["z"]\nstyle = "paraphrase"\n')
    qs = evaluate.load_questions(p)
    assert qs[0].expect_url == ["a"] and qs[1].expect_url == ["b", "c"] and qs[1].answer_any == ["z"]


def test_shipped_question_set_is_valid():
    qs = evaluate.load_questions("eval/questions.toml")
    assert len(qs) == 32 and len({q.q for q in qs}) == len(qs)
    assert all(q.expect_url and q.answer_any and q.same_story and q.style in ("keyword", "paraphrase") for q in qs)
    assert sum(q.style == "paraphrase" for q in qs) == 17


def test_same_story_terms_match_at_word_starts():
    q = evaluate.Question(q="?", expect_url=["x"], same_story=["Ukrain", "Libya|Benghazi", "drone"])
    assert q.matches_text("Ukrainian sea drones struck a ship off Benghazi")
    assert not q.matches_text("Ukrainian drones struck a ship off Crimea")  # every term is needed
    assert evaluate.Question(q="?", expect_url=[], same_story=["ended"]).matches_text("talks ended") and not \
        evaluate.Question(q="?", expect_url=[], same_story=["ended"]).matches_text("she attended")
    assert not evaluate.Question(q="?", expect_url=["x"]).matches_text("anything")  # no terms: URL only


def test_same_story_hit_counts_and_is_marked(db):
    store.insert_article(db, store.Article(url="https://other.example/eruption-update", title="Grindavik residents flee",
                                           source="O", body="Residents fled Grindavik as lava advanced.",
                                           extraction="rss", published_at="2026-11-24T10:00:00Z"),
                         ["Residents fled Grindavik as lava advanced."])
    q = evaluate.Question(q="Grindavik residents", expect_url=["not-in-db"], same_story=["Grindavik", "lava"])
    [res] = evaluate.run_mode(db, [q], "keyword", k=3).results
    assert not res.missing and res.rank == 1 and res.same_story
    rep = evaluate.run_mode(db, [q, QUESTIONS[0]], "keyword", k=3)
    assert rep.same_story_hits == 1


def test_question_set_id_tracks_content(tmp_path):
    a = tmp_path / "q.toml"
    a.write_text('[[question]]\nq = "x"\nexpect_url = ["y"]\n')
    first = evaluate.question_set_id(a)
    a.write_text('[[question]]\nq = "x?"\nexpect_url = ["y"]\n')
    assert len(first) == 8 and evaluate.question_set_id(a) != first


def test_keyword_vs_hybrid(db):
    kw = evaluate.run_mode(db, QUESTIONS, "keyword", k=3)
    hy = evaluate.run_mode(db, QUESTIONS, "hybrid", k=3, embedder=FakeEmbedder())
    assert [r.missing for r in kw.results] == [False, False, False, True]
    assert len(kw.scored) == 3
    assert kw.results[0].rank == 1 and kw.results[1].rank is None  # "magma" shares no keyword
    assert hy.results[1].rank == 1  # the vector side finds it
    assert hy.hit_rate() > kw.hit_rate()
    assert hy.hit_rate(style="paraphrase") == 1.0
    assert 0 < kw.mrr < hy.mrr <= 1.0
    assert hy.fallbacks == 0
    logged = db.execute("SELECT COUNT(*) FROM retrieval_log WHERE command = 'eval'").fetchone()[0]
    assert logged == 6  # missing questions aren't retrieved


def test_answer_checks(db):
    rep = evaluate.run_mode(db, QUESTIONS[:1], "keyword", answer_fn=lambda q, hits: "Lava flowed [1].")
    assert rep.results[0].answer_ok is True and rep.answer_rate == 1.0
    rep = evaluate.run_mode(db, QUESTIONS[:1], "keyword", answer_fn=lambda q, hits: "No idea.")
    assert rep.answer_rate == 0.0
    assert evaluate.answer_matches("It was on 16 December.", ["December 16", "16 December"])


def test_scores_are_recorded(db):
    rep = evaluate.run_mode(db, QUESTIONS, "keyword", k=3)
    evaluate.record_mode_report(db, rep, len(QUESTIONS), question_set="abcd1234")
    row = db.execute("SELECT * FROM eval_runs").fetchone()
    assert row["question_set"] == "abcd1234"
    assert (row["kind"], row["mode"], row["k"], row["questions"], row["scored"]) == ("retrieval", "keyword", 3, 4, 3)
    assert row["hit1"] == rep.hit_rate(1) and row["mrr"] == rep.mrr and row["rerank"] == 0
    logged = db.execute("SELECT hit_sources FROM retrieval_log WHERE command = 'eval'").fetchall()
    assert all(json.loads(r[0])["vector"] == 0 for r in logged)  # keyword mode: no vector hits

import json
from datetime import datetime, timezone

import pytest

from zands import embed, store
from zands.search import fuse, retrieve

from fakes import DownEmbedder, FakeEmbedder

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)

DOCS = [
    ("https://n.example/volcano", "Iceland evacuates town", "2026-11-23T09:00:00Z",
     "Lava flowed toward Grindavik after the eruption on the Reykjanes peninsula."),
    ("https://n.example/ecb", "European Central Bank holds", "2026-11-24T09:00:00Z",
     "The bank kept interest rates unchanged, citing slowing inflation."),
    ("https://n.example/chips", "Export rules tightened", "2026-11-22T09:00:00Z",
     "New semiconductor export rules cover advanced chips."),
    ("https://n.example/football", "Cup final tonight", "2026-11-24T09:00:00Z",
     "The cup final kicks off tonight at the national stadium."),
]


@pytest.fixture
def db(conn):
    for url, title, published, body in DOCS:
        art = store.Article(url=url, title=title, source="N", body=body, extraction="trafilatura", published_at=published)
        store.insert_article(conn, art, [body])
    return conn


@pytest.fixture
def embedded(db):
    embed.embed_pending(db, FakeEmbedder())
    return db


def test_vec_extension_loads(conn):
    assert conn.vec_enabled, conn.vec_error
    assert conn.execute("SELECT vec_version()").fetchone()[0].startswith("v")


def test_embed_pending_is_incremental_and_uses_document_prefix(db):
    fake = FakeEmbedder()
    assert embed.embed_pending(db, fake) == 4
    assert all(t.startswith("search_document: ") for t in fake.calls[0])
    assert "Iceland evacuates town" in fake.calls[0][0]  # title is embedded with the body
    assert embed.embed_pending(db, fake) == 0
    assert store.vector_count(db) == 4
    assert store.get_meta(db, "embed_model") == "fake-embed"


def test_query_uses_query_prefix(embedded):
    fake = FakeEmbedder()
    retrieve(embedded, "magma activity", embedder=fake, now=NOW)
    assert fake.calls == [["search_query: magma activity"]]


def test_vector_finds_what_keywords_miss(embedded):
    keyword = retrieve(embedded, "magma activity", mode="keyword", now=NOW)
    assert keyword.hits == []
    hybrid = retrieve(embedded, "magma activity", embedder=FakeEmbedder(), now=NOW)
    assert hybrid.mode == "hybrid"
    assert hybrid.hits[0].url == "https://n.example/volcano"
    assert hybrid.hits[0].vector_rank == 1 and hybrid.hits[0].keyword_rank is None


def test_hybrid_combines_both_signals(embedded):
    r = retrieve(embedded, "chipmakers export rules", embedder=FakeEmbedder(), now=NOW)
    top = r.hits[0]
    assert top.url == "https://n.example/chips"
    assert top.keyword_rank == 1 and top.vector_rank == 1
    assert r.candidates["keyword"] >= 1 and r.candidates["vector"] >= 1


def test_vector_mode(embedded):
    r = retrieve(embedded, "borrowing costs", mode="vector", embedder=FakeEmbedder(), now=NOW)
    assert r.mode == "vector"
    assert r.hits[0].url == "https://n.example/ecb"
    assert all(h.bm25 is None for h in r.hits)


def test_ollama_down_degrades_to_keyword(embedded):
    r = retrieve(embedded, "interest rates", embedder=DownEmbedder(), now=NOW)
    assert r.mode == "keyword (fallback)"
    assert "can't reach Ollama" in r.note
    assert r.hits and r.hits[0].url == "https://n.example/ecb"
    row = embedded.execute("SELECT mode, chunk_ids FROM retrieval_log ORDER BY id DESC").fetchone()
    assert row["mode"] == "keyword (fallback)"
    assert json.loads(row["chunk_ids"]) == [h.chunk_id for h in r.hits]


def test_no_vectors_yet_degrades(db):
    r = retrieve(db, "interest rates", embedder=FakeEmbedder(), now=NOW)
    assert r.mode == "keyword (fallback)" and "embedded" in r.note
    assert r.hits


def test_without_sqlite_vec_degrades(tmp_path):
    c = store.connect(tmp_path / "novec.db", vector=False)
    assert not c.vec_enabled
    store.insert_article(c, store.Article(url="https://x/1", title="Rates", source="N", body="interest rates", extraction="rss"), ["interest rates"])
    r = retrieve(c, "interest rates", embedder=FakeEmbedder(), now=NOW)
    assert r.mode == "keyword (fallback)" and "sqlite-vec" in r.note
    assert len(r.hits) == 1


def test_no_embedder_degrades(embedded):
    r = retrieve(embedded, "interest rates", now=NOW)
    assert r.mode == "keyword (fallback)" and r.note == "no embedder configured"


def test_mode_validation(embedded):
    with pytest.raises(ValueError):
        retrieve(embedded, "x", mode="semantic")


def test_rrf_prefers_chunks_found_by_both():
    def row(i, **extra):
        base = {"id": i, "article_id": i, "title": f"t{i}", "source": "s", "url": f"u{i}",
                "published_at": None, "body": "b", "ts": None}
        base.update(extra)
        return base

    kw = [row(1, snip="", rank=-9.0), row(2, snip="", rank=-8.0)]
    vec = [row(3, distance=0.1), row(2, distance=0.2)]
    hits = fuse(kw, vec, NOW)
    assert [h.chunk_id for h in hits] == [2, 1, 3]  # 2 is in both lists; 1 beats 3 on tie-break order
    assert hits[0].score == pytest.approx(1 / 62 + 1 / 62)


def test_model_mismatch_is_refused(embedded):
    other = FakeEmbedder()
    other.model = "another-model"
    with pytest.raises(embed.EmbedError, match="rebuild"):
        embed.embed_pending(embedded, other)
    embed.rebuild(embedded)
    assert store.vector_count(embedded) == 0
    assert embed.embed_pending(embedded, other) == 4


def test_deleting_a_chunk_removes_its_vector(embedded):
    embedded.execute("DELETE FROM articles WHERE url = 'https://n.example/volcano'")
    embedded.commit()
    assert store.vector_count(embedded) == 3


def test_recency_does_not_swamp_rrf():
    # A is clearly best in both lists but 60 days old; B is a weaker match from today.
    # Recency must not flip them (it did when the boost multiplied the fused RRF score).
    def row(i, ts, **extra):
        base = {"id": i, "article_id": i, "title": f"t{i}", "source": "s", "url": f"u{i}",
                "published_at": ts, "body": "b", "ts": ts}
        base.update(extra)
        return base

    old, new = "2026-09-25T12:00:00Z", "2026-11-24T11:00:00Z"
    kw = [row(1, old, snip="", rank=-10.0), row(2, new, snip="", rank=-5.0)]
    vec = [row(1, old, distance=0.1), row(2, new, distance=0.5)]
    assert [h.chunk_id for h in fuse(kw, vec, NOW, recency_weight=0.3)] == [1, 2]
    # ...but between near-equal matches, the newer one wins
    kw = [row(1, old, snip="", rank=-10.0), row(2, new, snip="", rank=-9.5)]
    vec = [row(1, old, distance=0.30), row(2, new, distance=0.31)]
    assert [h.chunk_id for h in fuse(kw, vec, NOW, recency_weight=0.3)] == [2, 1]

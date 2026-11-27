from datetime import UTC, datetime

from zands import rerank, store
from zands.search import Hit, retrieve


def hit(i):
    return Hit(chunk_id=i, article_id=i, title=f"t{i}", source="s", url=f"u{i}", published_at=None,
               body=f"body {i}", snippet="", score=1.0)


def test_parse_order():
    assert rerank.parse_order("3, 1, 3, 9, 0, 2", 4) == [2, 0, 1]
    assert rerank.parse_order("none of them", 4) == []


def test_apply_order_keeps_everything():
    hits = [hit(i) for i in range(4)]
    assert [h.chunk_id for h in rerank.apply_order(hits, [2, 0])] == [2, 0, 1, 3]


class Scripted(rerank.LLMReranker):
    def __init__(self, reply=None, error=False, n=20):
        super().__init__(n=n)
        self.reply, self.error, self.prompts = reply, error, []

    def ask(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise rerank.RerankError("down")
        return self.reply


def test_reranker_reorders_head_only():
    hits = [hit(i) for i in range(5)]
    rr = Scripted("3, 2", n=3)
    out = rr("q?", hits)
    assert [h.chunk_id for h in out] == [2, 1, 0, 3, 4]
    assert "Question: q?" in rr.prompts[0] and "[3] t2" in rr.prompts[0] and "t3" not in rr.prompts[0]


def test_reranker_failure_keeps_order():
    hits = [hit(i) for i in range(3)]
    assert Scripted(error=True)("q", hits) == hits


def test_retrieve_applies_reranker(conn):
    for i, body in enumerate(["interest rates held", "interest rates cut sharply", "rates"]):
        store.insert_article(conn, store.Article(url=f"https://x/{i}", title=f"T{i}", source="N", body=body,
                                                 extraction="rss", published_at="2026-11-24T00:00:00Z"), [body])
    now = datetime(2026, 11, 24, 12, tzinfo=UTC)
    plain = retrieve(conn, "interest rates", mode="keyword", now=now)
    reversed_ = retrieve(conn, "interest rates", mode="keyword", now=now,
                         reranker=lambda q, hits: list(reversed(hits)))
    assert [h.chunk_id for h in reversed_.hits] == [h.chunk_id for h in reversed(plain.hits)]

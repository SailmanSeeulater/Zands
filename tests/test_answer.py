from datetime import date

from zands.answer import build_messages
from zands.search import Hit


def hit(i: int, published: str | None) -> Hit:
    return Hit(
        chunk_id=i, article_id=i, title=f"Title {i}", source=f"Source {i}", url=f"https://x.example/{i}",
        published_at=published, body=f"Body text {i}.", snippet="", bm25=-1.0, score=1.0,
    )


def test_prompt_has_date_sources_and_rules():
    messages = build_messages("What happened?", [hit(1, "2026-11-24T08:00:00Z"), hit(2, None)], today=date(2026, 11, 4))
    system, user = messages[0]["content"], messages[1]["content"]
    assert "Wednesday, November 4, 2026" in system
    assert "ONLY" in system and "cite" in system.lower() and "say so plainly" in system
    assert "[1] Title 1\nSource: Source 1 | Published: 2026-11-24 | https://x.example/1" in user
    assert "[2] Title 2\nSource: Source 2 | Published: date unknown" in user
    assert "Body text 2." in user
    assert user.rstrip().endswith("Question: What happened?")

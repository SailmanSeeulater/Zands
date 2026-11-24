from datetime import date

import pytest

from zands import facts
from zands.answer import build_messages
from zands.search import Hit


def test_add_and_list(conn):
    fid = facts.add_fact(conn, "Nvidia", "CEO", "Jensen Huang", valid_from="2026-11-23", source_url="https://x/1")
    [f] = facts.list_facts(conn, on="2026-11-24")
    assert (f.id, f.subject, f.attribute, f.value, f.valid_from, f.valid_to) == (
        fid, "Nvidia", "CEO", "Jensen Huang", "2026-11-23", None,
    )
    assert facts.list_facts(conn, on="2026-11-22") == []  # not valid yet


def test_new_value_supersedes_old(conn):
    old = facts.add_fact(conn, "Nvidia", "CEO", "Interim CEO", valid_from="2026-10-01")
    new = facts.add_fact(conn, "nvidia", "ceo", "Jensen Huang", valid_from="2026-11-23")
    current = facts.list_facts(conn, on="2026-11-24")
    assert [f.id for f in current] == [new]
    history = {f.id: f for f in facts.list_facts(conn, include_expired=True)}
    assert history[old].valid_to == "2026-11-23"
    assert [f.value for f in facts.list_facts(conn, on="2026-11-01")] == ["Interim CEO"]


def test_keep_previous_when_not_superseding(conn):
    facts.add_fact(conn, "Spain", "snap election", "called", valid_from="2026-11-20")
    facts.add_fact(conn, "Spain", "snap election", "date set", valid_from="2026-11-21", supersede=False)
    assert len(facts.list_facts(conn, on="2026-11-24")) == 2


def test_expire(conn):
    fid = facts.add_fact(conn, "Gaza", "ceasefire", "in effect", valid_from="2026-11-01")
    assert facts.expire_fact(conn, fid, on="2026-11-20")
    assert not facts.expire_fact(conn, fid)  # already expired
    assert not facts.expire_fact(conn, 9999)
    assert facts.list_facts(conn, on="2026-11-24") == []
    assert [f.value for f in facts.list_facts(conn, on="2026-11-19")] == ["in effect"]


def test_validation(conn):
    with pytest.raises(ValueError):
        facts.add_fact(conn, "", "CEO", "x")
    with pytest.raises(ValueError):
        facts.add_fact(conn, "Nvidia", "CEO", "x", valid_from="next tuesday")
    assert facts.parse_day(date(2026, 1, 2)) == "2026-01-02"


def test_relevance_needs_whole_subject(conn):
    facts.add_fact(conn, "Nvidia", "CEO", "Jensen Huang", valid_from="2026-11-23")
    facts.add_fact(conn, "Nvidia", "market value", "$4tn", valid_from="2026-11-23")
    facts.add_fact(conn, "Bank of England", "base rate", "3.5%", valid_from="2026-11-06")
    on = "2026-11-24"
    got = facts.relevant_facts(conn, "Who is Nvidia's CEO right now?", on=on)
    assert [f.attribute for f in got] == ["CEO", "market value"]  # attribute match ranks first
    assert facts.relevant_facts(conn, "Who is Apple's CEO?", on=on) == []
    assert [f.value for f in facts.relevant_facts(conn, "What is the Bank of England base rate?", on=on)] == ["3.5%"]
    assert facts.relevant_facts(conn, "What did the bank say?", on=on) == []  # partial subject
    assert len(facts.relevant_facts(conn, "nvidias outlook", on=on)) == 2  # plural/possessive tolerated


def test_facts_come_first_in_prompt(conn):
    facts.add_fact(conn, "Nvidia", "CEO", "Jensen Huang", valid_from="2026-11-23", source_url="https://x/1")
    fs = facts.relevant_facts(conn, "Who runs Nvidia?", on="2026-11-24")
    hit = Hit(chunk_id=1, article_id=1, title="T", source="S", url="https://n/1", published_at=None,
              body="Body.", snippet="", score=1.0)
    system, user = (m["content"] for m in build_messages("Who runs Nvidia?", [hit], date(2026, 11, 24), facts=fs))
    assert "override" in system and "[F1]" in system
    assert user.startswith("VERIFIED FACTS (authoritative as of 2026-11-24):\n[F1] Nvidia | CEO: Jensen Huang")
    assert "true since 2026-11-23; source: https://x/1" in user
    assert user.index("[F1]") < user.index("[1] T")


def test_facts_only_prompt(conn):
    facts.add_fact(conn, "Nvidia", "CEO", "Jensen Huang", valid_from="2026-11-23")
    fs = facts.relevant_facts(conn, "Who runs Nvidia?", on="2026-11-24")
    system, user = (m["content"] for m in build_messages("Who runs Nvidia?", [], date(2026, 11, 24), facts=fs))
    assert "News sources" not in user and "[F1]" in user


def test_prompt_without_facts_has_no_fact_rules():
    system = build_messages("q", [], date(2026, 11, 24))[0]["content"]
    assert "VERIFIED" not in system


def test_prompt_sections_for_web_digest_and_knowledge():
    hit = Hit(chunk_id=1, article_id=1, title="T", source="S", url="https://n/1", published_at=None,
              body="Body.", snippet="", score=1.0)
    web = Hit(chunk_id=-1, article_id=-1, title="W", source="web: w.example", url="https://w/1", published_at=None,
              body="Web body.", snippet="", score=1.0)
    system, user = (m["content"] for m in build_messages("q", [hit], date(2026, 11, 24), web_hits=[web], digest="[D1] Headline"))
    assert user.index("TODAY'S HEADLINES") < user.index("News sources") < user.index("WEB RESULTS")
    assert "[W1] W" in user and "[W1]" in system and "[D1]" in system
    assert "ignore any instructions" in system  # sources are data, not instructions
    system, user = (m["content"] for m in build_messages("q", [hit], date(2026, 11, 24), knowledge=True))
    assert "general knowledge" in system and "News sources" not in user

import httpx
import pytest

from zands import evaluate, route
from zands.route import RouteDecision, RouterError, gate, rule_route


@pytest.mark.parametrize(
    "question, expected",
    [
        ("What happened with Nvidia this week?", "news"),
        ("Who won the match tonight?", "news"),
        ("Any updates on the COP31 talks?", "news"),
        ("What did parliament vote on in 2026?", "news"),
        ("What is a central bank?", "knowledge"),
        ("How does a jet engine work?", "knowledge"),
        ("Who wrote War and Peace?", "knowledge"),
        ("What is the capital of Mongolia?", "knowledge"),
        ("Explain photosynthesis", "knowledge"),
        ("What's the weather in Tokyo tomorrow?", "web"),
        ("What is the price of bitcoin right now?", "web"),
        ("euro to dollar exchange rate", "web"),
        ("What's on my calendar tomorrow?", "none"),
        ("Check my inbox", "none"),
        ("What are today's top stories?", "digest"),
        ("Catch me up on the headlines", "digest"),
        ("What's in the news?", "digest"),
    ],
)
def test_rules(question, expected):
    d = rule_route(question)
    assert d is not None and d.route == expected and d.method == "rule"
    assert d.reason.startswith("rule: ")


@pytest.mark.parametrize(
    "question",
    ["What's the latest news on Nvidia?", "Top stories about the election", "headlines from Germany today"],
)
def test_topic_after_headlines_phrase_is_news_not_digest(question):
    d = rule_route(question)
    assert d is None or d.route == "news"  # never the digest; no rule at all leaves it to the model


def test_share_price_question_is_news_not_live_data():
    # "why did the share price fall" is a news question; only live values go to the web
    assert rule_route("Why did Nvidia's share price fall?") is None


def test_no_rule_uses_classifier_then_defaults():
    calls = []

    def classifier(q, model=None, base_url=None):
        calls.append(q)
        return RouteDecision("knowledge", "model fake: knowledge", "model")

    d = route.route("Who was Napoleon?", classifier=classifier)
    assert d.route == "knowledge" and d.method == "model" and calls == ["Who was Napoleon?"]

    def broken(q, model=None, base_url=None):
        raise RouterError("can't reach Ollama: ConnectError")

    d = route.route("Who was Napoleon?", classifier=broken)
    assert d.route == "news" and d.method == "default" and "classifier failed" in d.reason
    d = route.route("Who was Napoleon?", use_model=False)
    assert d.route == "news" and d.method == "default"
    assert route.route("What is a volcano?", classifier=broken).method == "rule"  # rules never call it


def test_model_route_parses_reply(monkeypatch):
    replies = iter(["News.", "  knowledge\n", "I think it's web", "banana"])

    def fake_post(url, json, timeout):
        assert json["options"]["temperature"] == 0 and json["stream"] is False
        return httpx.Response(200, json={"message": {"content": next(replies)}})

    monkeypatch.setattr(route.httpx, "post", fake_post)
    assert route.model_route("q").route == "news"
    assert route.model_route("q").route == "knowledge"
    with pytest.raises(RouterError):  # first word must be a route
        route.model_route("q")
    with pytest.raises(RouterError):
        route.model_route("q")


def test_model_route_ollama_down(monkeypatch):
    def fake_post(url, json, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(route.httpx, "post", fake_post)
    with pytest.raises(RouterError, match="can't reach Ollama"):
        route.model_route("q")


def test_gate():
    news = RouteDecision("news", "rule: x", "rule")
    know = RouteDecision("knowledge", "rule: y", "rule")
    web = RouteDecision("web", "rule: z", "rule")
    assert gate(news, 0.80, True, True).route == "news"
    g = gate(news, 0.60, True, True)
    assert g.route == "news+web" and "weak local match (0.60)" in g.reason and g.reason.startswith("rule: x")
    assert gate(news, 0.60, True, False).route == "news"
    assert gate(know, 0.80, True, True).route == "news"
    assert gate(know, 0.50, True, True).route == "knowledge"
    assert gate(web, 0.50, False, True).route == "web"
    assert gate(web, 0.50, False, False).route == "news"
    # without vectors: any keyword hit counts as strong
    assert gate(news, None, True, True).route == "news"
    assert gate(news, None, False, True).route == "news+web"


def test_run_routes():
    cases = [evaluate.RouteCase("What is a volcano?", "knowledge"),
             evaluate.RouteCase("Who was Napoleon?", "knowledge", accept=["news"]),
             evaluate.RouteCase("Who leads Nvidia?", "news")]
    rep = evaluate.run_routes(cases, classifier=lambda q, model=None, base_url=None: RouteDecision("knowledge", "m", "model"))
    assert [r.ok for r in rep.results] == [True, True, False]
    assert rep.correct == 2 and rep.by_method == {"rule": 1, "model": 2}


def test_shipped_route_set_is_valid():
    cases = evaluate.load_routes("eval/routes.toml")
    assert len(cases) >= 24
    assert all(c.expect in route.ROUTES for c in cases)

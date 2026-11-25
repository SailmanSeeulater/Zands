"""Live-data tools, the guarded GET and the public-apis catalog, all without the network."""

import json
from datetime import date

import httpx
import pytest

from zands import store
from zands.apis import catalog, http
from zands.apis import tools as live
from zands.apis.http import FetchError, Fetched

PUBLIC = lambda host: ["93.184.216.34"]  # noqa: E731 - a public address for any host


# ---------------------------------------------------------------- the fence


@pytest.mark.parametrize(
    ("url", "why"),
    [
        ("http://api.example.com/x", "only https"),
        ("https://localhost/x", "local name"),
        ("https://printer.local/x", "local name"),
        ("https://127.0.0.1/x", "non-public"),
        ("https://169.254.169.254/latest/meta-data", "non-public"),  # cloud metadata
        ("https://[::1]/x", "non-public"),
        ("https://user:pw@api.example.com/", "credentials"),
        ("ftp://api.example.com/", "only https"),
    ],
)
def test_the_fence_refuses_non_public_or_non_https(url, why):
    with pytest.raises(FetchError, match=why):
        http.check_url(url, PUBLIC)


def test_a_name_resolving_to_a_private_address_is_refused():
    with pytest.raises(FetchError, match="non-public"):
        http.check_url("https://intranet.example.com/", lambda h: ["10.0.0.5"])
    http.check_url("https://api.example.com/", PUBLIC)  # fine


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


def test_redirects_are_checked_at_every_hop():
    def handler(request):
        return httpx.Response(302, headers={"location": "https://127.0.0.1/admin"})

    with pytest.raises(FetchError, match="non-public"):
        http.get("https://api.example.com/", client=_client(handler), resolver=PUBLIC)


def test_a_redirect_to_a_public_host_is_followed_and_the_body_capped(monkeypatch):
    monkeypatch.setattr(http, "MAX_BYTES", 100)

    def handler(request):
        if request.url.path == "/old":
            return httpx.Response(301, headers={"location": "/new"})
        return httpx.Response(200, content=b"x" * 1000, headers={"content-type": "text/plain"})

    got = http.get("https://api.example.com/old", client=_client(handler), resolver=PUBLIC)
    assert got.url.endswith("/new") and got.status == 200 and len(got.body) == 100


def test_get_json_names_a_failing_status():
    client = _client(lambda r: httpx.Response(404, json={"error": "nope"}))
    with pytest.raises(FetchError, match="HTTP 404"):
        http.get_json("https://api.example.com/x", client=client, resolver=PUBLIC)


def test_readable_compacts_json_extracts_html_and_cuts():
    j = Fetched("u", 200, "application/json", json.dumps({"a": [1, 2], "b": "é"}, indent=4).encode())
    assert http.readable(j) == '{"a":[1,2],"b":"é"}'
    page = "<html><body><nav>menu</nav><article><h1>Docs</h1>" + "<p>Call GET /v1/fact to get a fact. " * 20 + "</p></article></body></html>"
    text = http.readable(Fetched("u", 200, "text/html", page.encode()))
    assert "GET /v1/fact" in text
    cut = http.readable(Fetched("u", 200, "text/plain", b"y" * 50), limit=10)
    assert cut.startswith("y" * 10) and "cut:" in cut


# ---------------------------------------------------------------- typed tools


class FakeFetch:
    """Answers by URL substring; records calls; raises FetchError for a FetchError value."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, url, params=None, timeout=None, raw=False):
        self.calls.append((url, params))
        for key, value in self.routes.items():
            if key in url:
                if isinstance(value, Exception):
                    raise value
                return Fetched(url, 200, "application/xml", value.encode()) if raw else value
        raise FetchError(f"no route for {url}")


GEO = {"results": [{"name": "Tokyo", "admin1": "Tokyo", "country": "Japan", "country_code": "JP",
                    "latitude": 35.69, "longitude": 139.69, "timezone": "Asia/Tokyo"}]}
FORECAST = {
    "current": {"temperature_2m": 13.0, "apparent_temperature": 10.2, "relative_humidity_2m": 36,
                "weather_code": 0, "wind_speed_10m": 3.0},
    "daily": {"time": ["2026-11-25"], "weather_code": [3], "temperature_2m_min": [6.6], "temperature_2m_max": [13.0],
              "precipitation_sum": [0.0], "precipitation_probability_max": [29]},
}
AIR = {"current": {"us_aqi": 63, "pm2_5": 15.9, "pm10": 18.0}}


def test_weather_from_open_meteo_with_air_quality():
    fetch = FakeFetch({"geocoding-api": GEO, "api.open-meteo.com/v1/forecast": FORECAST, "air-quality": AIR})
    out = live.weather("Tokyo", fetch=fetch)
    assert "Weather for Tokyo, Tokyo, Japan" in out
    assert "Now: 13.0°C (feels 10.2°C), clear sky" in out
    assert "2026-11-25: 6.6–13.0°C, overcast" in out and "US AQI 63" in out
    assert "Source: Open-Meteo" in out and "(Asia/Tokyo)" in out


def test_weather_falls_back_to_wttr_when_open_meteo_is_down():
    wttr = {"current_condition": [{"temp_C": "12", "FeelsLikeC": "10", "humidity": "50", "windspeedKmph": "5",
                                   "weatherDesc": [{"value": "Sunny "}]}],
            "weather": [{"date": "2026-11-25", "mintempC": "6", "maxtempC": "14", "hourly": [{"chanceofrain": "20"}]}]}
    fetch = FakeFetch({"geocoding-api": GEO, "api.open-meteo.com": FetchError("didn't answer"), "wttr.in": wttr,
                       "air-quality": FetchError("down")})
    out = live.weather("Tokyo", fetch=fetch)
    assert "Now: 12°C (feels 10°C), sunny" in out and "rain chance up to 20%" in out
    assert "wttr.in (Open-Meteo unavailable)" in out and "Air quality" not in out


def test_weather_for_several_places_in_one_call():
    # Asked "warmer in Bangkok or Berlin?", a 7B model looked up Bangkok and made up Berlin.
    # One call can now cover both.
    geo = {"Bangkok": {"results": [{"name": "Bangkok", "country": "Thailand", "latitude": 13.75, "longitude": 100.5,
                                    "timezone": "Asia/Bangkok"}]},
           "Berlin": {"results": [{"name": "Berlin", "country": "Germany", "latitude": 52.52, "longitude": 13.41,
                                   "timezone": "Europe/Berlin"}]}}

    def fetch(url, params=None, timeout=None):
        if "geocoding" in url:
            return geo[params["name"]]
        if "forecast" in url:
            return FORECAST
        raise FetchError("no air quality")

    for asked in ("Bangkok; Berlin", "Bangkok vs Berlin", "Bangkok or Berlin"):
        out = live.weather(asked, fetch=fetch)
        assert "Weather for Bangkok, Thailand" in out and "Weather for Berlin, Germany" in out, asked
    one = FakeFetch({"geocoding-api": GEO, "open-meteo": FORECAST})
    live.weather("Paris, France", fetch=one)  # a comma is "City, Country", not two places
    assert [p["name"] for u, p in one.calls if "geocoding" in u] == ["Paris"]


def test_weather_for_an_unknown_place_says_so():
    assert "No place called" in live.weather("Xyzzyville", fetch=FakeFetch({"geocoding-api": {}}))


def test_geocode_prefers_the_country_or_state_given():
    paris = {"results": [{"name": "Paris", "country": "France", "country_code": "FR", "admin1": "Île-de-France"},
                         {"name": "Paris", "country": "United States", "country_code": "US", "admin1": "Texas"}]}
    assert live.geocode("Paris, Texas", FakeFetch({"geocoding": paris}))["admin1"] == "Texas"
    assert live.geocode("Paris", FakeFetch({"geocoding": paris}))["country"] == "France"


def test_convert_currency_fiat_and_crypto():
    fetch = FakeFetch({"frankfurter": {"base": "USD", "date": "2026-11-24", "rates": {"THB": 33.93}},
                       "coingecko": {"bitcoin": {"eur": 50000, "last_updated_at": 1795000000}}})
    assert live.convert_currency(100, "usd", "thb", fetch=fetch).startswith("100 USD = 3,393.00 THB (rate 33.93)")
    btc = live.convert_currency(0.5, "BTC", "EUR", fetch=fetch)
    assert "0.5 BTC = 25,000.00 EUR" in btc and "CoinGecko" in btc
    eur_to_btc = live.convert_currency(25000, "EUR", "BTC", fetch=fetch)
    assert "= 0.5 BTC" in eur_to_btc


def test_an_unknown_currency_code_gets_a_plain_answer():
    fetch = FakeFetch({"frankfurter": FetchError("api.frankfurter.dev answered HTTP 404")})
    assert "isn't a currency code" in live.convert_currency(1, "USD", "XYZ", fetch=fetch)


def test_failures_come_back_as_text_suggesting_another_way():
    out = live.wikipedia("anything", fetch=FakeFetch({"wikipedia": FetchError("timed out")}))
    assert out.startswith("Couldn't get") and "another tool" in out


def test_wikipedia_summary():
    fetch = FakeFetch({
        "w/api.php": ["canberra", ["Canberra", "Canberra Raiders"], [], []],
        "page/summary/Canberra": {"title": "Canberra", "description": "Capital city of Australia",
                                  "extract": "Canberra is the capital city of Australia.", "timestamp": "2026-11-01T00:00:00Z",
                                  "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Canberra"}}},
    })
    out = live.wikipedia("capital of australia", fetch=fetch)
    assert out.startswith("Wikipedia: Canberra — Capital city of Australia")
    assert "last edited 2026-11-01" in out and "Other matches: Canberra Raiders" in out


def test_public_holidays_are_national_deduped_with_the_next_marked():
    year = date.today().year
    days = [
        {"date": f"{year}-01-01", "name": "New Year's Day", "localName": "New Year's Day", "global": True},
        {"date": f"{year}-04-03", "name": "Good Friday", "localName": "Good Friday", "global": True},
        {"date": f"{year}-04-03", "name": "Good Friday", "localName": "Good Friday", "global": True},
        {"date": f"{year}-05-08", "name": "Truman Day", "localName": "Truman Day", "global": False},
        {"date": f"{year}-12-25", "name": "Christmas Day", "localName": "Christmas Day", "global": True},
        {"date": f"{year}-12-31", "name": "New Year's Eve", "localName": "New Year's Eve", "global": True},
    ]
    fetch = FakeFetch({"AvailableCountries": [{"countryCode": "US", "name": "United States"}], "PublicHolidays": days})
    out = live.public_holidays("united states", year, fetch=fetch, today=date(year, 11, 24))
    assert out.count("Good Friday") == 1 and "Truman Day" not in out and "plus 1 regional" in out
    # The next one is worked out on the first line, not left to be found in the list.
    assert out.splitlines()[0] == f"Today is {year}-11-24. Next national holiday in US: {year}-12-25, Christmas Day (in 31 days)."
    assert out.count("<- next") == 1 and f"{year}-12-25: Christmas Day  <- next" in out
    late = live.public_holidays("US", year, fetch=fetch, today=date(year, 12, 31))
    assert late.startswith(f"Today is {year}-12-31. Next national holiday in US: {year}-12-31, New Year's Eve (today).")
    over = live.public_holidays("US", year, fetch=FakeFetch({"PublicHolidays": days[:2]}), today=date(year, 11, 24))
    assert f"No national holidays left in US this year; ask for year {year + 1}." in over and "<- next" not in over
    assert 'no holiday data for "Narnia"' in live.public_holidays("Narnia", year, fetch=fetch)


def test_earthquakes_filtered_and_strongest_first():
    feed = {"features": [{"properties": {"mag": m, "place": f"place {m}", "time": 1795000000000, "tsunami": int(m > 6)}}
                         for m in (4.6, 6.4, 5.1)]}
    fetch = FakeFetch({"4.5_day": feed})
    out = live.earthquakes(5, "day", fetch=fetch)
    assert out.splitlines()[1].startswith("M6.4 place 6.4") and "tsunami alert" in out
    assert "place 4.6" not in out and ": 2 (USGS" in out


ATOM = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/abs/2411.1</id><published>2024-11-27T00:00:00Z</published>
<title>Retrieval-Augmented
 Generation for Reviews</title><summary>We study RAG.</summary>
<author><name>A</name></author><author><name>B</name></author></entry></feed>"""


def test_arxiv_requires_every_word_and_parses_entries():
    fetch = FakeFetch({"arxiv": ATOM})
    out = live.arxiv("retrieval augmented generation", 3, fetch_text=lambda u, p, **kw: fetch(u, p, raw=True))
    assert "- Retrieval-Augmented Generation for Reviews (2024-11-27; A, B)" in out
    assert fetch.calls[0][1]["search_query"] == "all:retrieval AND all:augmented AND all:generation"


def test_books():
    fetch = FakeFetch({"openlibrary": {"numFound": 2, "docs": [
        {"title": "Dune", "author_name": ["Frank Herbert"], "first_publish_year": 1965, "edition_count": 155,
         "key": "/works/OL893414W"}]}})
    assert "- Dune by Frank Herbert, first published 1965, 155 editions" in live.books("dune", fetch=fetch)


# ---------------------------------------------------------------- the catalog

README = """# Public APIs
### APIs Covered Under APILayer Suite!
| API | Description | Call this API |
|:---|:---|:---|
| [IPstack](https://ipstack.com/) | Locate visitors | [run](https://x) |
### Animals
API | Description | Auth | HTTPS | CORS
|:---|:---|:---|:---|:---|
| [Cat Facts](https://catfact.ninja/) | Daily cat facts | No | Yes | Yes |
| [Cats](https://docs.thecatapi.com/) | Pictures of cats | `apiKey` | Yes | No |
### Transportation
API | Description | Auth | HTTPS | CORS
|:---|:---|:---|:---|:---|
| [Transport for Berlin, Germany](https://github.com/derhuerst/vbb-rest) | Third-party VBB API | No | Yes | Unknown |
| [Old Trains](http://trains.example) | Train times | No | No | No |
| [Status Page](https://status.example) | Status of every service | No | Yes | Yes |
"""


def test_parse_readme_takes_category_tables_only():
    rows = catalog.parse_readme(README)
    assert [r["name"] for r in rows] == ["Cat Facts", "Cats", "Transport for Berlin, Germany", "Old Trains", "Status Page"]
    cats = rows[1]
    assert cats["auth"] == "apiKey" and cats["category"] == "Animals" and cats["https"] == 1
    assert rows[0]["auth"] == "" and rows[3]["https"] == 0


@pytest.fixture
def conn(tmp_path):
    c = store.connect(tmp_path / "news.db")
    rows = catalog.parse_readme(README)
    c.executemany(
        "INSERT INTO api_catalog(name, url, description, category, auth, https, cors) "
        "VALUES (:name, :url, :description, :category, :auth, :https, :cors)", rows)
    c.commit()
    return c


def test_find_offers_free_https_apis_and_every_word_first(conn):
    names = [r["name"] for r in catalog.find(conn, "train berlin")]
    assert names[0] == "Transport for Berlin, Germany"
    assert "Old Trains" not in names  # plain http: call_api can't reach it
    assert "Cats" not in [r["name"] for r in catalog.find(conn, "cats")]  # needs a key
    assert "Cats" in [r["name"] for r in catalog.find(conn, "cats", include_keyed=True)]
    assert catalog.find(conn, "AND OR NOT (") == []  # FTS syntax can't break the query


def test_format_found_points_to_the_next_step(conn):
    out = catalog.format_found("cat facts", catalog.find(conn, "cat facts"), catalog.count(conn))
    assert "Cat Facts [Animals]" in out and "docs: https://catfact.ninja/" in out and "call_api" in out
    assert "zands apis update" in catalog.format_found("x", [], 0)
    assert "No free, no-key API" in catalog.format_found("x", [], 5)


def test_update_refuses_a_list_it_cant_parse(conn):
    with pytest.raises(FetchError, match="format may have changed"):
        catalog.update(conn, fetch=lambda url, timeout=None: Fetched(url, 200, "text/plain", b"# nothing"))
    assert catalog.count(conn) == 5  # kept

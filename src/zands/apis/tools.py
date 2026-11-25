"""Typed live-data tools over no-key public APIs, each returning short plain text.

Small models call a tool with an obvious name and simple arguments far more reliably than they
build a URL, so the common needs get one tool each. Every function takes `fetch` (http.get_json
by default) so tests run without the network, and returns text that names its source. Failures
come back as text too ("couldn't reach ..."), never an exception, so the model can try
another way.

APIs (all free, no key): Open-Meteo (geocoding, forecast, air quality) with wttr.in as the
weather fallback, Frankfurter (ECB exchange rates), CoinGecko (crypto prices), Wikipedia,
Nager.Date (public holidays), USGS (earthquakes), arXiv, Open Library, Free Dictionary.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import date, datetime, timezone
from urllib.parse import quote
from zoneinfo import ZoneInfo

from . import http
from .http import FetchError

Fetch = Callable[..., object]

WMO = {  # Open-Meteo weather codes
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light showers", 81: "showers",
    82: "violent showers", 85: "snow showers", 86: "heavy snow showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}

CRYPTO = {  # symbol -> CoinGecko id
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "XRP": "ripple", "DOGE": "dogecoin",
    "ADA": "cardano", "BNB": "binancecoin", "USDT": "tether", "USDC": "usd-coin", "LTC": "litecoin",
    "DOT": "polkadot", "TRX": "tron", "AVAX": "avalanche-2", "LINK": "chainlink", "XLM": "stellar",
}


def _fail(what: str, exc: Exception) -> str:
    return f"Couldn't get {what}: {exc}. Try different wording, another tool, or say it's unavailable."


# ---------------------------------------------------------------- places


def geocode(place: str, fetch: Fetch = http.get_json) -> dict | None:
    data = fetch("https://geocoding-api.open-meteo.com/v1/search", {"name": place.split(",")[0].strip(), "count": 5})
    results = (data or {}).get("results") or []
    if not results:
        return None
    # "Paris, US" / "Springfield, Illinois": prefer a result whose country or region matches the rest.
    hint = ",".join(place.split(",")[1:]).strip().lower()
    if hint:
        for r in results:
            if hint in " ".join(str(r.get(k, "")) for k in ("country", "country_code", "admin1")).lower():
                return r
    return results[0]


def _place_name(g: dict) -> str:
    return ", ".join(x for x in (g.get("name"), g.get("admin1"), g.get("country")) if x)


# "Bangkok; Berlin", "Bangkok vs Berlin", "Bangkok or Berlin": several places in one call, since
# small models asked to compare two cities tend to look up one and invent the other.
SEVERAL_PLACES = re.compile(r"\s*;\s*|\s*\|\s*|\s+(?:vs\.?|versus|or|and)\s+", re.I)


def weather(place: str, days: int = 3, fetch: Fetch = http.get_json) -> str:
    """Current weather, a short forecast, local time and air quality for a place, or for each of
    several places separated by ';'."""
    places = [p for p in SEVERAL_PLACES.split(place.strip()) if p][:5]
    if len(places) > 1:
        return "\n\n".join(_weather_one(p, days, fetch) for p in places)
    return _weather_one(place, days, fetch)


def _weather_one(place: str, days: int, fetch: Fetch) -> str:
    days = max(1, min(int(days), 7))
    try:
        g = geocode(place, fetch)
    except FetchError as exc:
        return _fail(f"the location of {place!r}", exc)
    if not g:
        return f'No place called "{place}" was found. Try the city name, or "City, Country".'
    lat, lon, tz = g["latitude"], g["longitude"], g.get("timezone") or "UTC"
    lines = [f"Weather for {_place_name(g)} ({lat:.2f}, {lon:.2f})"]
    try:
        lines.append(f"Local time: {datetime.now(ZoneInfo(tz)):%Y-%m-%d %H:%M} ({tz})")
    except Exception:  # noqa: BLE001 - an unknown zone name only loses this line
        pass
    try:
        f = fetch(
            "https://api.open-meteo.com/v1/forecast",
            {
                "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": days,
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
            },
            timeout=6,
        )
        c = f["current"]
        lines.append(
            f"Now: {c['temperature_2m']}°C (feels {c['apparent_temperature']}°C), {WMO.get(c['weather_code'], 'code ' + str(c['weather_code']))}, "
            f"humidity {c['relative_humidity_2m']}%, wind {c['wind_speed_10m']} km/h"
        )
        d = f["daily"]
        for i, day in enumerate(d["time"]):
            lines.append(
                f"{day}: {d['temperature_2m_min'][i]}–{d['temperature_2m_max'][i]}°C, {WMO.get(d['weather_code'][i], '?')}, "
                f"rain {d['precipitation_sum'][i]} mm ({d['precipitation_probability_max'][i]}% chance)"
            )
        source = "Open-Meteo"
    except (FetchError, KeyError, TypeError) as exc:
        try:
            w = fetch(f"https://wttr.in/{lat:.3f},{lon:.3f}", {"format": "j1"})
            c = w["current_condition"][0]
            lines.append(
                f"Now: {c['temp_C']}°C (feels {c['FeelsLikeC']}°C), {c['weatherDesc'][0]['value'].strip().lower()}, "
                f"humidity {c['humidity']}%, wind {c['windspeedKmph']} km/h"
            )
            for day in w.get("weather", [])[:days]:
                rain = max((int(h.get("chanceofrain", 0)) for h in day.get("hourly", [])), default=0)
                lines.append(f"{day['date']}: {day['mintempC']}–{day['maxtempC']}°C, rain chance up to {rain}%")
            source = "wttr.in (Open-Meteo unavailable)"
        except (FetchError, KeyError, IndexError, TypeError) as exc2:
            return _fail(f"weather for {place!r}", exc2 if isinstance(exc2, FetchError) else exc)
    try:
        a = fetch(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            {"latitude": lat, "longitude": lon, "current": "us_aqi,pm2_5,pm10"},
            timeout=6,
        )["current"]
        lines.append(f"Air quality: US AQI {a['us_aqi']}, PM2.5 {a['pm2_5']} µg/m³, PM10 {a['pm10']} µg/m³")
    except (FetchError, KeyError, TypeError):
        pass
    lines.append(f"Source: {source}, fetched {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC.")
    return "\n".join(lines)


# ---------------------------------------------------------------- money


def _money(value: float) -> str:
    """Cents for ordinary amounts, significant digits for tiny ones (0.000012 BTC)."""
    return f"{value:,.2f}" if abs(value) >= 1 else f"{value:.6g}"


def convert_currency(amount: float, from_currency: str, to_currency: str, fetch: Fetch = http.get_json) -> str:
    """Convert between currencies (ECB rates) or price a crypto coin (CoinGecko)."""
    src, dst = from_currency.strip().upper(), to_currency.strip().upper()
    amount = float(amount)
    try:
        if src in CRYPTO or dst in CRYPTO:
            coin, fiat, invert = (src, dst, False) if src in CRYPTO else (dst, src, True)
            data = fetch(
                "https://api.coingecko.com/api/v3/simple/price",
                {"ids": CRYPTO[coin], "vs_currencies": fiat.lower(), "include_last_updated_at": "true"},
            )
            entry = data.get(CRYPTO[coin]) or {}
            price = entry.get(fiat.lower())
            if price is None:
                return f"CoinGecko has no {coin} price in {fiat}."
            when = datetime.fromtimestamp(entry.get("last_updated_at", 0), timezone.utc)
            value = amount / price if invert else amount * price
            return (
                f"{amount:g} {src} = {_money(value)} {dst} (1 {coin} = {price:,} {fiat}). "
                f"Source: CoinGecko, updated {when:%Y-%m-%d %H:%M} UTC. Prices move by the minute."
            )
        data = fetch("https://api.frankfurter.dev/v1/latest", {"base": src, "symbols": dst})
        rate = (data.get("rates") or {}).get(dst)
        if rate is None:
            return f"No ECB rate from {src} to {dst}. Frankfurter covers about 30 major currencies."
        return (
            f"{amount:g} {src} = {_money(amount * rate)} {dst} (rate {rate}). "
            f"Source: European Central Bank via Frankfurter, rates of {data.get('date')} (updated on working days)."
        )
    except FetchError as exc:
        msg = str(exc)
        if "HTTP 404" in msg or "HTTP 422" in msg:
            return f"{src} or {dst} isn't a currency code Frankfurter knows (use ISO codes like USD, EUR, THB)."
        return _fail(f"the {src}→{dst} rate", exc)


# ---------------------------------------------------------------- reference


def wikipedia(topic: str, fetch: Fetch = http.get_json) -> str:
    """The Wikipedia summary of the best-matching article."""
    try:
        found = fetch(
            "https://en.wikipedia.org/w/api.php",
            {"action": "opensearch", "search": topic, "limit": 3, "namespace": 0, "format": "json"},
        )
        titles = found[1] if isinstance(found, list) and len(found) > 1 else []
        if not titles:
            return f'Wikipedia has no article matching "{topic}".'
        s = fetch(f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(titles[0].replace(' ', '_'), safe='')}")
    except FetchError as exc:
        return _fail(f"Wikipedia on {topic!r}", exc)
    out = [f"Wikipedia: {s.get('title')}" + (f" — {s['description']}" if s.get("description") else "")]
    out.append((s.get("extract") or "(no summary)").strip())
    if s.get("timestamp"):
        out.append(f"(article last edited {s['timestamp'][:10]})")
    out.append((s.get("content_urls") or {}).get("desktop", {}).get("page", ""))
    if len(titles) > 1:
        out.append("Other matches: " + "; ".join(titles[1:]))
    return "\n".join(x for x in out if x)


def define_word(word: str, fetch: Fetch = http.get_json) -> str:
    """English definitions of a word."""
    try:
        data = fetch(f"https://api.dictionaryapi.dev/api/v2/entries/en/{quote(word.strip())}")
    except FetchError as exc:
        if "HTTP 404" in str(exc):
            return f'No dictionary entry for "{word}".'
        return _fail(f"a definition of {word!r}", exc)
    out = []
    for entry in data[:1]:
        out.append(f"{entry.get('word')} {entry.get('phonetic') or ''}".strip())
        for meaning in entry.get("meanings", [])[:3]:
            for d in meaning.get("definitions", [])[:2]:
                out.append(f"- ({meaning.get('partOfSpeech')}) {d.get('definition')}")
    return "\n".join(out + ["Source: Free Dictionary API (Wiktionary)."])


# ---------------------------------------------------------------- calendars and events


def public_holidays(country: str, year: int | None = None, fetch: Fetch = http.get_json, today: date | None = None) -> str:
    """Public holidays of a country in a year."""
    now = today or date.today()
    year = int(year or now.year)
    try:
        code = country.strip().upper()
        if len(code) != 2:
            countries = fetch("https://date.nager.at/api/v3/AvailableCountries")
            want = country.strip().lower()
            match = next((c for c in countries if c["name"].lower() == want), None) or next(
                (c for c in countries if want in c["name"].lower()), None
            )
            if not match:
                return f'Nager.Date has no holiday data for "{country}".'
            code = match["countryCode"]
        days = fetch(f"https://date.nager.at/api/v3/PublicHolidays/{year}/{code}")
    except FetchError as exc:
        return _fail(f"holidays for {country!r}", exc)
    if not days:
        return f"No public holidays listed for {code} in {year}."
    today = now.isoformat()
    national, seen = [], set()
    for h in days:
        if h.get("global", True) and (h["date"], h["name"]) not in seen:
            seen.add((h["date"], h["name"]))
            national.append(h)
    regional = len({(h["date"], h["name"]) for h in days if not h.get("global", True)})
    upcoming = next((h for h in national if h["date"] >= today), None)
    # The answer to "when is the next holiday" goes first, worked out: small models misread a
    # marker in the middle of a list.
    if upcoming:
        days_left = (date.fromisoformat(upcoming["date"]) - now).days
        when = "today" if days_left == 0 else "tomorrow" if days_left == 1 else f"in {days_left} days"
        lines = [f"Today is {today}. Next national holiday in {code}: {upcoming['date']}, {upcoming['name']} ({when})."]
    elif year == now.year:
        lines = [f"Today is {today}. No national holidays left in {code} this year; ask for year {year + 1}."]
    else:
        lines = [f"Today is {today}."]
    lines.append(f"All national public holidays in {code}, {year} (Source: Nager.Date):")
    for h in national:
        mark = "  <- next" if h is upcoming else ""
        local = f" ({h['localName']})" if h["localName"] != h["name"] else ""
        lines.append(f"{h['date']}: {h['name']}{local}{mark}")
    if regional:
        lines.append(f"(plus {regional} regional holidays observed only in some states or provinces)")
    return "\n".join(lines)


def earthquakes(min_magnitude: float = 4.5, period: str = "day", fetch: Fetch = http.get_json) -> str:
    """Recent earthquakes worldwide from the USGS."""
    period = period if period in ("hour", "day", "week", "month") else "day"
    m = float(min_magnitude)
    feed = "significant" if m >= 6.5 else "4.5" if m >= 4.5 else "2.5" if m >= 2.5 else "1.0" if m >= 1 else "all"
    try:
        data = fetch(f"https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/{feed}_{period}.geojson")
    except FetchError as exc:
        return _fail("earthquake data", exc)
    quakes = [f["properties"] for f in data.get("features", []) if (f["properties"].get("mag") or 0) >= m]
    quakes.sort(key=lambda p: p.get("mag") or 0, reverse=True)
    if not quakes:
        return f"No magnitude {m:g}+ earthquakes in the past {period} (USGS)."
    lines = [f"Magnitude {m:g}+ earthquakes in the past {period}: {len(quakes)} (USGS, strongest first)"]
    for p in quakes[:10]:
        when = datetime.fromtimestamp(p["time"] / 1000, timezone.utc)
        tsunami = ", tsunami alert" if p.get("tsunami") else ""
        lines.append(f"M{p['mag']} {p.get('place')} at {when:%Y-%m-%d %H:%M} UTC{tsunami}")
    return "\n".join(lines)


# ---------------------------------------------------------------- papers and books


def arxiv(query: str, max_results: int = 5, fetch_text: Callable[..., http.Fetched] = http.get) -> str:
    """arXiv papers matching all the words of a query, most relevant first."""
    n = max(1, min(int(max_results), 10))
    # Every word must match ("all:a AND all:b"): arXiv ORs bare words, which for a phrase
    # like "retrieval augmented generation" brings back papers on anything generative.
    terms = [w for w in "".join(ch if ch.isalnum() or ch in "-." else " " for ch in query).split() if len(w) > 1]
    if not terms:
        return "Give arXiv a few keywords to search for."
    try:
        got = fetch_text(
            "https://export.arxiv.org/api/query",
            {"search_query": " AND ".join(f"all:{w}" for w in terms[:8]), "max_results": n,
             "sortBy": "relevance", "sortOrder": "descending"},
            timeout=15,
        )
        root = ET.fromstring(got.body)
    except (FetchError, ET.ParseError) as exc:
        return _fail(f"arXiv results for {query!r}", exc)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entries = root.findall("a:entry", ns)
    if not entries:
        return f'No arXiv papers match "{query}".'
    lines = [f'arXiv papers for "{query}", most relevant first:']
    for e in entries:
        title = " ".join((e.findtext("a:title", "", ns)).split())
        authors = [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)]
        summary = " ".join((e.findtext("a:summary", "", ns)).split())
        lines.append(
            f"- {title} ({e.findtext('a:published', '', ns)[:10]}; {', '.join(authors[:3])}{' et al.' if len(authors) > 3 else ''})\n"
            f"  {e.findtext('a:id', '', ns)}\n  {summary[:240]}{'…' if len(summary) > 240 else ''}"
        )
    return "\n".join(lines)


def books(query: str, max_results: int = 5, fetch: Fetch = http.get_json) -> str:
    """Books matching a title, author or subject (Open Library)."""
    n = max(1, min(int(max_results), 10))
    try:
        data = fetch(
            "https://openlibrary.org/search.json",
            {"q": query, "limit": n, "fields": "title,author_name,first_publish_year,edition_count,key"},
        )
    except FetchError as exc:
        return _fail(f"books for {query!r}", exc)
    docs = data.get("docs") or []
    if not docs:
        return f'Open Library has no books matching "{query}".'
    lines = [f'Books matching "{query}" ({data.get("numFound", len(docs)):,} found; Open Library):']
    for d in docs:
        authors = ", ".join((d.get("author_name") or ["unknown author"])[:2])
        lines.append(
            f"- {d.get('title')} by {authors}, first published {d.get('first_publish_year', '?')}, "
            f"{d.get('edition_count', '?')} editions: https://openlibrary.org{d.get('key', '')}"
        )
    return "\n".join(lines)


TOOLS = {
    "weather": weather,
    "convert_currency": convert_currency,
    "wikipedia": wikipedia,
    "define_word": define_word,
    "public_holidays": public_holidays,
    "earthquakes": earthquakes,
    "arxiv": arxiv,
    "books": books,
}

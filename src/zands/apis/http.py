"""A guarded GET for public web APIs, used by every live-data tool.

A model decides what URL to call (call_api takes any), so the request is fenced in:
- HTTPS and GET only, no cookies, a generic User-Agent, no credentials ever sent.
- The host must resolve only to public addresses: loopback, private, link-local (cloud
  metadata), multicast and reserved ranges are refused, as are `localhost` and `.local`
  names. Redirects are followed by hand, at most 3, and each hop is checked the same way.
  (A host that resolves differently between the check and the connection could slip past;
  that needs a DNS server under someone's control, and the request is still GET-only.)
- At most MAX_BYTES of the body is read, with a timeout.
HTML is turned into readable text (an API's docs page), JSON is compacted, and both are cut
to a size a small model can read.
"""

from __future__ import annotations

import ipaddress
import json
import socket
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx

warnings.filterwarnings("ignore", message="SOCKS support in urllib3")
import trafilatura  # noqa: E402

USER_AGENT = "Zands/0.4 (local assistant; +https://github.com/SailmanSeeulater/Zands)"
TIMEOUT = 10.0
MAX_BYTES = 400_000
MAX_REDIRECTS = 3
MAX_CHARS = 6_000  # what a tool hands back to the model

Resolver = Callable[[str], list[str]]


class FetchError(RuntimeError):
    """The request was refused or failed; the message is safe to show the model."""


@dataclass
class Fetched:
    url: str
    status: int
    content_type: str
    body: bytes

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self):
        try:
            return json.loads(self.body)
        except ValueError as exc:
            raise FetchError(f"{self.url} didn't return JSON ({self.content_type or 'no content type'})") from exc


def resolve(host: str) -> list[str]:
    try:
        return sorted({info[4][0] for info in socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)})
    except socket.gaierror as exc:
        raise FetchError(f"can't resolve {host}: {exc.strerror or exc}") from exc


def check_url(url: str, resolver: Resolver = resolve) -> None:
    """Raise FetchError unless `url` is https on a host with only public addresses."""
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise FetchError(f"only https URLs are allowed, not {parts.scheme or 'a bare path'}: {url}")
    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        raise FetchError(f"no host in {url}")
    if parts.username or parts.password:
        raise FetchError("URLs with credentials in them aren't allowed")
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal", ".lan", ".home.arpa")):
        raise FetchError(f"{host} is a local name; only public hosts are allowed")
    try:
        addresses = [host] if ipaddress.ip_address(host) else []
    except ValueError:
        addresses = resolver(host)
    if not addresses:
        raise FetchError(f"{host} has no addresses")
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global or ip.is_multicast:
            raise FetchError(f"{host} resolves to a non-public address ({address}); refused")


def make_client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT, "Accept": "application/json, text/plain;q=0.9, */*;q=0.5"},
        timeout=TIMEOUT,
        follow_redirects=False,
    )


def get(
    url: str,
    params: dict | None = None,
    *,
    client: httpx.Client | None = None,
    resolver: Resolver = resolve,
    timeout: float = TIMEOUT,
) -> Fetched:
    """GET `url` within the fence above. Raises FetchError."""
    own = client is None
    client = client or make_client()
    try:
        if params:
            url = str(httpx.URL(url, params={k: v for k, v in params.items() if v is not None}))
        for _ in range(MAX_REDIRECTS + 1):
            check_url(url, resolver)
            try:
                with client.stream("GET", url, timeout=timeout) as resp:
                    if resp.is_redirect and resp.headers.get("location"):
                        url = urljoin(url, resp.headers["location"])
                        continue
                    body = bytearray()
                    for chunk in resp.iter_bytes():
                        body += chunk
                        if len(body) > MAX_BYTES:
                            body = body[:MAX_BYTES]
                            break
                    return Fetched(str(resp.url), resp.status_code, resp.headers.get("content-type", ""), bytes(body))
            except httpx.TimeoutException as exc:
                raise FetchError(f"{urlsplit(url).hostname} didn't answer within {timeout:.0f}s") from exc
            except httpx.HTTPError as exc:
                raise FetchError(f"request to {urlsplit(url).hostname} failed: {exc}") from exc
        raise FetchError(f"more than {MAX_REDIRECTS} redirects")
    finally:
        if own:
            client.close()


def get_json(url: str, params: dict | None = None, **kw):
    """GET and parse JSON; a non-2xx status is a FetchError naming it."""
    fetched = get(url, params, **kw)
    if not 200 <= fetched.status < 300:
        raise FetchError(f"{urlsplit(fetched.url).hostname} answered HTTP {fetched.status}")
    return fetched.json()


def readable(fetched: Fetched, limit: int = MAX_CHARS) -> str:
    """The body as text a model can read: JSON compacted, HTML as its main text, cut to `limit`."""
    ctype = fetched.content_type.lower()
    text = fetched.text
    if "json" in ctype or text.lstrip()[:1] in ("{", "["):
        try:
            text = json.dumps(json.loads(fetched.body), ensure_ascii=False, separators=(",", ":"))
        except ValueError:
            pass
    elif "html" in ctype or text.lstrip()[:1] == "<" and "<html" in text[:2000].lower():
        text = trafilatura.extract(text, include_links=True, include_tables=True, favor_recall=True) or ""
        if not text.strip():
            text = "(the page had no readable text; it may need JavaScript)"
    text = text.strip()
    if len(text) > limit:
        text = text[:limit] + f"\n… (cut: {len(fetched.body):,} bytes in all; ask for a narrower URL or fewer results)"
    return text

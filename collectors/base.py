"""Base collector primitives shared across ATS sources.

Every source ends up producing a stream of RawPosting records that the upsert
layer consumes. The collector itself knows nothing about skills, salaries, or
title families — only how to fetch, parse the source's JSON, apply the India
location filter, and hand off structured records.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

log = structlog.get_logger()

# Per-host politeness: one request every ~2.5s.
_MIN_INTERVAL_SEC = 2.5

# Contact email advertised in the User-Agent so anyone rate-limiting us can
# reach out instead of just blocking. Falls back to a placeholder in dev.
_CONTACT = os.environ.get("COLLECTOR_CONTACT_EMAIL", "unknown@example.com")
USER_AGENT = (
    f"IndianTechJobPanel/0.1 (+contact: {_CONTACT}; "
    "research; https://github.com/rohit-raj/indian-tech-job-panel)"
)


@dataclass
class RawPosting:
    """One posting as returned by a source, pre-upsert.

    payload is the source's original record verbatim — this is the "keep the
    raw" rule. Everything downstream can be re-derived from it.
    """

    source: str
    source_id: str | None
    company: str
    title_raw: str
    location_raw: str
    url: str
    description_raw: str
    payload: dict[str, Any] = field(default_factory=dict)

    def content_hash(self) -> str:
        """Hash of the fields that would count as an edit if they changed."""
        blob = "|".join([
            self.title_raw or "",
            self.location_raw or "",
            self.description_raw or "",
        ])
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def payload_gzip(self) -> bytes:
        return gzip.compress(json.dumps(self.payload, ensure_ascii=False).encode("utf-8"))


# --- India location filter ---------------------------------------------------

# Canonical India cities + a few common misspellings/variants. All lowercase.
# NCR members collapse to "NCR" per Phase 0; we still match them individually
# so the filter fires, but geography.py in the extractor does the collapsing.
_INDIA_CITIES = {
    "bangalore", "bengaluru", "blr",
    "mumbai", "bombay",
    "delhi", "new delhi", "ncr",
    "gurgaon", "gurugram",
    "noida",
    "ghaziabad",
    "faridabad",
    "hyderabad", "hyd", "secunderabad",
    "pune",
    "chennai", "madras",
    "kolkata", "calcutta",
    "ahmedabad",
    "jaipur",
    "kochi", "cochin", "ernakulam",
    "chandigarh",
    "trivandrum", "thiruvananthapuram",
    "coimbatore",
    "indore",
    "bhubaneswar",
    "nagpur",
    "vadodara", "baroda",
    "mysore", "mysuru",
    "vizag", "visakhapatnam",
    "surat",
    "lucknow",
    "kanpur",
    "goa",
    "mohali",
    "gandhinagar",
}

# Regex for standalone India-country signals. Word-bounded so "Indianapolis"
# doesn't match.
_INDIA_SIGNAL = re.compile(
    r"\b(?:india|bharat|in-?remote|remote\s*[-,]?\s*india|apac\s*[-,]?\s*india)\b",
    re.IGNORECASE,
)

# Non-India signals that would false-positive on a bare city name (e.g. "Delhi,
# NY" is not the Delhi we mean). If the location string contains one of these
# country/state anchors and no India anchor, we drop it.
_NON_INDIA_ANCHORS = re.compile(
    r"\b(?:usa|united states|u\.s\.a|u\.s\.|canada|uk|united kingdom|"
    r"germany|france|spain|italy|netherlands|ireland|australia|"
    r"singapore|dubai|uae|japan|korea|philippines|vietnam|indonesia|"
    r"brazil|mexico|argentina|"
    r"new york|california|texas|london|berlin|toronto|sydney)\b",
    re.IGNORECASE,
)

_CITY_TOKEN = re.compile(r"[a-zA-Z]+")


def is_india_location(location: str | None) -> bool:
    """True if the location string looks like an India-hiring posting.

    Heuristic (deliberately conservative — we'd rather miss an India posting
    than include a US one and pollute the panel):

    1. If it contains a strong non-India country/state anchor and no India
       anchor, drop it. This handles "Delhi, NY" and "Chennai Consulting, USA".
    2. If it contains an India country signal ("india", "remote - india"),
       accept.
    3. If any tokenised word matches a known India city, accept.
    4. Otherwise, drop.
    """
    if not location:
        return False

    loc = location.strip().lower()
    if not loc:
        return False

    india_signal = bool(_INDIA_SIGNAL.search(loc))
    non_india_signal = bool(_NON_INDIA_ANCHORS.search(loc))

    if non_india_signal and not india_signal:
        return False

    if india_signal:
        return True

    tokens = {t.lower() for t in _CITY_TOKEN.findall(loc)}
    return bool(tokens & _INDIA_CITIES)


# --- HTTP client with polite rate limiting ----------------------------------


class RateLimitedClient:
    """httpx.Client wrapper that keeps at least _MIN_INTERVAL_SEC between calls
    to the same host, sets a truthful User-Agent, and retries transient errors.
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self._client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            follow_redirects=True,
        )
        self._last_hit: dict[str, float] = {}

    def _sleep_if_needed(self, host: str) -> None:
        now = time.monotonic()
        last = self._last_hit.get(host, 0.0)
        wait = _MIN_INTERVAL_SEC - (now - last)
        if wait > 0:
            time.sleep(wait)
        self._last_hit[host] = time.monotonic()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        reraise=True,
    )
    def get_json(self, url: str) -> Any:
        host = httpx.URL(url).host
        self._sleep_if_needed(host)
        resp = self._client.get(url)
        # 404 on ATS boards means the token is stale/wrong — not a retriable error.
        if resp.status_code == 404:
            resp.raise_for_status()
        # Retry on 5xx and 429.
        if resp.status_code >= 500 or resp.status_code == 429:
            resp.raise_for_status()
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> RateLimitedClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# --- Collector base class ---------------------------------------------------


class BaseCollector:
    """Interface every ATS collector implements.

    Subclasses override fetch_for_token to yield RawPosting objects from one
    company token; the base run() method handles orchestration, applies the
    India filter, and returns a flat list ready for the upsert layer.
    """

    source_name: str = "base"

    def __init__(self, tokens: Iterable[str], client: RateLimitedClient | None = None) -> None:
        self.tokens = list(tokens)
        self.client = client or RateLimitedClient()
        self._owns_client = client is None

    def fetch_for_token(self, token: str) -> Iterator[RawPosting]:
        raise NotImplementedError

    def run(self) -> list[RawPosting]:
        out: list[RawPosting] = []
        for token in self.tokens:
            token_log = log.bind(source=self.source_name, token=token)
            try:
                fetched = 0
                kept = 0
                for posting in self.fetch_for_token(token):
                    fetched += 1
                    if is_india_location(posting.location_raw):
                        out.append(posting)
                        kept += 1
                token_log.debug("token_done", fetched=fetched, kept=kept)
            except httpx.HTTPStatusError as exc:
                # 404 is common: tokens go stale as companies switch ATS or shut down.
                if exc.response.status_code == 404:
                    token_log.info("token_404_dropping")
                else:
                    token_log.warning("http_error", status=exc.response.status_code)
            except Exception as exc:  # noqa: BLE001 - collector-level catch-all
                token_log.error("token_failed", err=str(exc))
        return out

    def __del__(self) -> None:
        if getattr(self, "_owns_client", False):
            try:
                self.client.close()
            except Exception:  # noqa: BLE001
                pass

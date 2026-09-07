"""Greenhouse ATS collector.

Endpoint: https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true
No auth. Returns all live postings for a company board with full HTML descriptions.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from html import unescape

from collectors.base import BaseCollector, RawPosting

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def _strip_html(html: str | None) -> str:
    if not html:
        return ""
    text = _TAG.sub(" ", html)
    text = unescape(text)
    return _WS.sub(" ", text).strip()


class GreenhouseCollector(BaseCollector):
    source_name = "greenhouse"

    def fetch_for_token(self, token: str) -> Iterator[RawPosting]:
        url = f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true"
        data = self.client.get_json(url)

        # Greenhouse response shape: {"jobs": [{"id": ..., "title": ..., ...}]}
        # Some boards nest company name; others don't. Fall back to the token.
        company_hint = (
            data.get("meta", {}).get("board_name")
            or token.replace("-", " ").title()
        )

        for job in data.get("jobs", []):
            location_raw = ""
            if isinstance(job.get("location"), dict):
                location_raw = job["location"].get("name", "") or ""
            elif isinstance(job.get("offices"), list) and job["offices"]:
                location_raw = ", ".join(o.get("name", "") for o in job["offices"] if o)

            yield RawPosting(
                source=self.source_name,
                source_id=str(job.get("id", "")),
                company=company_hint,
                title_raw=(job.get("title") or "").strip(),
                location_raw=location_raw.strip(),
                url=(job.get("absolute_url") or "").strip(),
                description_raw=_strip_html(job.get("content")),
                payload=job,
            )

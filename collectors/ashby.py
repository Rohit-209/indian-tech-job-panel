"""Ashby ATS collector.

Endpoint: https://api.ashbyhq.com/posting-api/job-board/{name}?includeCompensation=true
No auth. Compensation flag is important — Ashby is one of the few ATSes that
surfaces structured salary bands.
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


class AshbyCollector(BaseCollector):
    source_name = "ashby"

    def fetch_for_token(self, token: str) -> Iterator[RawPosting]:
        url = (
            f"https://api.ashbyhq.com/posting-api/job-board/{token}"
            "?includeCompensation=true"
        )
        data = self.client.get_json(url)

        # Ashby response: {"jobs": [...], "apiVersion": "..."}
        company_name = token.replace("-", " ").title()

        for job in data.get("jobs", []):
            location_raw = job.get("location") or ""
            if not location_raw and job.get("secondaryLocations"):
                secondary = job["secondaryLocations"]
                if isinstance(secondary, list):
                    location_raw = ", ".join(
                        loc.get("location", "") for loc in secondary if isinstance(loc, dict)
                    )

            description_raw = _strip_html(
                job.get("descriptionHtml") or job.get("descriptionPlain") or ""
            )

            yield RawPosting(
                source=self.source_name,
                source_id=str(job.get("id", "")),
                company=company_name,
                title_raw=(job.get("title") or "").strip(),
                location_raw=location_raw.strip(),
                url=(job.get("jobUrl") or job.get("applyUrl") or "").strip(),
                description_raw=description_raw,
                payload=job,
            )

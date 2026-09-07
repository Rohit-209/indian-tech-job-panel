"""Lever ATS collector.

Endpoint: https://api.lever.co/v0/postings/{company}?mode=json
No auth. Returns live postings with rich structured metadata (categories,
descriptionPlain, workplaceType).
"""

from __future__ import annotations

from collections.abc import Iterator

from collectors.base import BaseCollector, RawPosting


def _combine_description(job: dict) -> str:
    """Lever splits descriptions into descriptionPlain plus lists. Concatenate
    them so downstream skill extraction sees the whole thing.
    """
    parts: list[str] = []
    if job.get("descriptionPlain"):
        parts.append(job["descriptionPlain"])
    for section in job.get("lists", []) or []:
        text = section.get("text") or ""
        content = section.get("content") or ""
        # content is HTML; a naive strip is fine for skill matching.
        content_clean = " ".join(content.replace("<", " <").split())
        parts.append(f"{text} {content_clean}")
    if job.get("additionalPlain"):
        parts.append(job["additionalPlain"])
    return " ".join(p for p in parts if p).strip()


class LeverCollector(BaseCollector):
    source_name = "lever"

    def fetch_for_token(self, token: str) -> Iterator[RawPosting]:
        url = f"https://api.lever.co/v0/postings/{token}?mode=json"
        data = self.client.get_json(url)

        # Lever returns a flat list of postings.
        for job in data:
            categories = job.get("categories") or {}
            location_raw = categories.get("location", "") or ""
            # Some postings expose allLocations for multi-location roles.
            if not location_raw and job.get("allLocations"):
                location_raw = ", ".join(job["allLocations"])

            # workplaceType is 'on-site' | 'remote' | 'hybrid' — worth capturing
            # in the raw payload so the extractor can use it directly.
            company_name = token.replace("-", " ").title()

            yield RawPosting(
                source=self.source_name,
                source_id=str(job.get("id", "")),
                company=company_name,
                title_raw=(job.get("text") or "").strip(),
                location_raw=location_raw.strip(),
                url=(job.get("hostedUrl") or job.get("applyUrl") or "").strip(),
                description_raw=_combine_description(job),
                payload=job,
            )

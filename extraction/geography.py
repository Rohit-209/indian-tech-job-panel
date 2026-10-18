"""Geography: canonical city + workplace model (remote / hybrid / onsite).

City
  location_raw is split into segments on ";" ("Bengaluru; Gurugram; Mumbai"),
  each India segment is mapped to a canonical city, and Delhi / Gurugram /
  Noida / Ghaziabad / Faridabad collapse into "NCR" (Phase 0 decision — the
  postings don't distinguish them consistently). `city` is the first India
  city listed, `cities` all of them. A posting that also lists non-India
  segments ("Bangalore, India; Remote, Canada; ...") is `is_multi_country`:
  it's a global role India happens to be eligible for, not an India-market
  posting in the same sense.

Workplace
  Ordered by how much the signal can be trusted; first hit wins and the
  source is recorded in workplace_source:
    1. ats_field  — Ashby workplaceType; Greenhouse custom fields
                    (Airbnb "Workplace Type", MongoDB "Working Model Eligibility")
    2. li_tag     — LinkedIn tags in the description (#LI-Hybrid, #LI-Remote)
    3. location   — "Remote - India" etc., India segments only
    4. phrase     — explicit statements ("hybrid model of 3 days in office",
                    "all-remote"); bare words like "remote teams" don't count
    5. company_policy — employer-wide fact, only where the panel itself shows
                    it (GitLab: every posting with a signal says all-remote)
  No signal -> is_remote / is_hybrid stay NULL. Absence of a remote signal
  is not evidence of onsite.

Run:
  python -m extraction.geography --dry-run
  python -m extraction.geography
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from collections import Counter
from dataclasses import dataclass

from collectors.base import is_india_location
from extraction.text import clean_text

# alias (lowercase, word-bounded) -> canonical city
_CITY_ALIASES: dict[str, str] = {
    "bengaluru": "Bengaluru", "bangalore": "Bengaluru",
    "gurugram": "NCR", "gurgaon": "NCR", "new delhi": "NCR", "delhi": "NCR",
    "noida": "NCR", "greater noida": "NCR", "ghaziabad": "NCR", "faridabad": "NCR",
    "mumbai": "Mumbai", "bombay": "Mumbai", "navi mumbai": "Mumbai", "thane": "Mumbai",
    "hyderabad": "Hyderabad", "secunderabad": "Hyderabad",
    "chennai": "Chennai", "madras": "Chennai",
    "pune": "Pune",
    "kolkata": "Kolkata", "calcutta": "Kolkata",
    "ahmedabad": "Ahmedabad", "gandhinagar": "Ahmedabad",
    "jaipur": "Jaipur",
    "kochi": "Kochi", "cochin": "Kochi",
    "thiruvananthapuram": "Thiruvananthapuram", "trivandrum": "Thiruvananthapuram",
    "chandigarh": "Chandigarh", "mohali": "Chandigarh",
    "indore": "Indore", "coimbatore": "Coimbatore", "mysore": "Mysuru", "mysuru": "Mysuru",
    "nagpur": "Nagpur", "surat": "Surat", "lucknow": "Lucknow", "goa": "Goa",
}
# Longest alias first so "new delhi" / "navi mumbai" win over "delhi" / "mumbai".
_CITY_RX = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in sorted(_CITY_ALIASES, key=len, reverse=True)) + r")\b",
    re.I,
)
_REMOTE_RX = re.compile(r"\bremote\b", re.I)

# Values seen in the panel's ATS custom fields.
_ASHBY_WORKPLACE = {"remote": "remote", "hybrid": "hybrid", "onsite": "onsite"}
_GH_WORKPLACE_FIELDS = {
    "workplace type": {"remote": "remote", "hybrid": "hybrid", "onsite": "onsite"},
    # MongoDB: "1-In Office", "2-Flexible", "3-Fully Remote", "3CF-Customer Facing Remote"
    "working model eligibility": {"in office": "onsite", "flexible": "hybrid",
                                  "remote": "remote"},
}

# Employer-wide workplace model, used only when the posting itself is silent.
# Keep this tiny and evidenced: GitLab has no offices, and in the panel
# every GitLab posting with any workplace signal is remote (29/29 on
# 2026-10-18), none hybrid or onsite.
_COMPANY_POLICY = {"gitlab": "remote"}

_LI_TAG = re.compile(r"#LI-\s*(Remote|Hybrid|Onsite|On-site|Office)\b", re.I)
_LI_LEVEL = {"remote": "remote", "hybrid": "hybrid", "onsite": "onsite",
             "on-site": "onsite", "office": "onsite"}

_PHRASES: list[tuple[str, re.Pattern[str]]] = [
    ("hybrid", re.compile(
        r"\bhybrid (?:work(?:ing)? )?(?:model|position|policy|arrangement|schedule)\b"
        r"|\bthis (?:is a |role is (?:a )?)hybrid\b"
        r"|\b(?:\d|two|three|four) days? (?:a |per |each )?week in (?:the |our )?office\b"
        r"|\b(?:\d|two|three|four) days? in (?:the )?office\b"
        r"|\bin the office at least (?:\d|two|three|four) days\b",
        re.I)),
    ("remote", re.compile(
        r"\ball[- ]remote\b|\bfully remote\b|\bremote[- ]first\b"
        r"|\bthis (?:is a |role is (?:a )?|position is (?:a )?)(?:fully )?remote\b",
        re.I)),
    ("onsite", re.compile(
        # Not Okta's "you can work from the office, or home" flexibility policy.
        r"\bwork from (?:the )?office\b(?!,? or (?:from )?(?:home|remote))"
        r"|\bfully (?:on-?site|in[- ]office)\b"
        r"|\bfive days (?:a |per )?week in (?:the )?office\b"
        r"|\bthis (?:is an |role is (?:an )?)(?:on-?site|in[- ]office)\b",
        re.I)),
]


@dataclass(frozen=True)
class Geography:
    city: str | None
    cities: list[str]
    is_multi_country: bool
    workplace: str | None          # 'remote' | 'hybrid' | 'onsite' | None
    workplace_source: str | None   # ats_field | li_tag | location | phrase | company_policy

    @property
    def is_remote(self) -> bool | None:
        return None if self.workplace is None else self.workplace == "remote"

    @property
    def is_hybrid(self) -> bool | None:
        return None if self.workplace is None else self.workplace == "hybrid"


def _segments(location: str) -> list[str]:
    return [s.strip() for s in re.split(r"[;|]", location or "") if s.strip()]


def parse_location(location: str | None) -> tuple[list[str], bool, bool]:
    """-> (canonical India cities in order, any India segment remote, multi-country)."""
    cities: list[str] = []
    remote = False
    multi = False
    for seg in _segments(location or ""):
        if not is_india_location(seg):
            multi = True
            continue
        for m in _CITY_RX.finditer(seg):
            city = _CITY_ALIASES[m.group(1).lower()]
            if city not in cities:
                cities.append(city)
        remote = remote or bool(_REMOTE_RX.search(seg))
    return cities, remote, multi


def _ats_workplace(source: str, payload: dict) -> str | None:
    if source == "ashby":
        wt = (payload.get("workplaceType") or "").lower()
        return _ASHBY_WORKPLACE.get(wt)
    if source == "greenhouse":
        for field in payload.get("metadata") or []:
            mapping = _GH_WORKPLACE_FIELDS.get((field.get("name") or "").strip().lower())
            if not mapping:
                continue
            value = field.get("value")
            value = " ".join(value) if isinstance(value, list) else str(value or "")
            value = value.lower()
            for needle, level in mapping.items():
                if needle in value:
                    return level
    return None


def _ashby_secondary(payload: dict) -> str:
    sec = payload.get("secondaryLocations") or []
    return "; ".join(s.get("location", "") for s in sec if isinstance(s, dict))


def extract_geography(
    source: str,
    location_raw: str | None,
    description_raw: str | None,
    payload: dict | None = None,
    company: str | None = None,
) -> Geography:
    payload = payload or {}
    location = location_raw or ""
    if source == "ashby" and (secondary := _ashby_secondary(payload)):
        location = f"{location}; {secondary}" if location else secondary
    cities, loc_remote, multi = parse_location(location)

    workplace, src = _ats_workplace(source, payload), "ats_field"
    text = None
    if workplace is None:
        text = clean_text(description_raw)
        if m := _LI_TAG.search(text):
            workplace, src = _LI_LEVEL[m.group(1).lower()], "li_tag"
    if workplace is None and loc_remote:
        workplace, src = "remote", "location"
    if workplace is None:
        for level, rx in _PHRASES:
            if rx.search(text or ""):
                workplace, src = level, "phrase"
                break
    if workplace is None and (policy := _COMPANY_POLICY.get((company or "").lower())):
        workplace, src = policy, "company_policy"

    return Geography(
        city=cities[0] if cities else None,
        cities=cities,
        is_multi_country=multi,
        workplace=workplace,
        workplace_source=src if workplace else None,
    )


# --- DB replay --------------------------------------------------------------


def _fetch_postings(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT posting_key, source, company, location_raw, description_raw, payload_raw"
            " FROM postings"
        )
        for key, source, company, loc, desc, payload_raw in cur:
            payload = json.loads(gzip.decompress(payload_raw)) if payload_raw else {}
            yield key, source, company, loc, desc, payload


def _write(conn, rows: list[tuple]) -> None:
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(
            """
            CREATE TEMP TABLE geo_assign (
                posting_key TEXT, city TEXT, cities TEXT[], is_multi_country BOOLEAN,
                is_remote BOOLEAN, is_hybrid BOOLEAN, workplace_source TEXT
            ) ON COMMIT DROP
            """
        )
        with cur.copy("COPY geo_assign FROM STDIN") as copy:
            for row in rows:
                copy.write_row(row)
        cur.execute(
            """
            UPDATE postings p
            SET city = g.city, cities = g.cities, is_multi_country = g.is_multi_country,
                is_remote = g.is_remote, is_hybrid = g.is_hybrid,
                workplace_source = g.workplace_source
            FROM geo_assign g
            WHERE p.posting_key = g.posting_key
            """
        )


def _report(results: dict[str, Geography]) -> None:
    n = len(results)
    print(f"=== geography ({n} postings) ===")
    print("\n  city:")
    for city, c in Counter(g.city for g in results.values()).most_common():
        print(f"    {c:>5}  {c / n:>4.0%}  {city}")
    multi_city = sum(len(g.cities) > 1 for g in results.values())
    multi_country = sum(g.is_multi_country for g in results.values())
    print(f"\n  multi-city: {multi_city}   multi-country: {multi_country}")
    print("\n  workplace (source):")
    for (w, s), c in Counter(
        (g.workplace, g.workplace_source) for g in results.values()
    ).most_common():
        print(f"    {c:>5}  {c / n:>4.0%}  {w} ({s})")


def replay(conn, dry_run: bool = False) -> None:
    """Recompute city / workplace for every posting."""
    results = {
        key: extract_geography(source, loc, desc, payload, company)
        for key, source, company, loc, desc, payload in _fetch_postings(conn)
    }
    _report(results)
    if dry_run:
        print("\n(dry run — postings unchanged)")
        return
    _write(conn, [
        (k, g.city, g.cities, g.is_multi_country, g.is_remote, g.is_hybrid,
         g.workplace_source)
        for k, g in results.items()
    ])
    print(f"\nwrote geography for {len(results)} postings")


def main() -> int:
    import psycopg
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="report only, no writes")
    args = parser.parse_args()

    load_dotenv()
    with psycopg.connect(os.environ["SUPABASE_DB_URL"]) as conn:
        replay(conn, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

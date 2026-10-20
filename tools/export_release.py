"""Export the public dataset release: derived tables, no employer text.

Released (CC BY 4.0, see LICENSE-DATA):
  postings.csv.gz        one row per posting: identity, URL, title, location,
                         lifecycle dates, every extracted field, plus two
                         fields derived from the withheld text so the analyses
                         still reproduce — published_on (the ATS's own publish
                         date, needed for A2's delayed entry) and
                         description_words (A5)
  observations.csv.gz    one row per posting per day it was seen
  posting_skills.csv.gz  skills matched in each description
  MANIFEST.json          row counts, sha256 per file, panel dates, export time

Withheld: description_raw and payload_raw. That text belongs to the employers;
the URL lets anyone read a posting while it's live. A7's similarity scores
need the text, so its output ships as report/a7_duplicate_pairs.csv instead.

Run: python -m tools.export_release            # -> dist/release/
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from datetime import UTC, datetime

import pandas as pd
import psycopg
from dotenv import load_dotenv

from analysis.common import REPO, _query, load_lifecycle
from extraction.text import clean_text

OUT = REPO / "dist" / "release"

# Whitelist, not blacklist: a column added to the schema later stays private
# until someone decides to release it.
POSTING_COLUMNS = [
    "posting_key", "source", "source_id", "company", "title_raw", "location_raw", "url",
    "first_seen", "last_seen", "is_active", "closed_on", "content_hash",
    "title_family", "seniority", "city", "cities", "is_multi_country", "is_remote",
    "is_hybrid", "workplace_source", "salary_min", "salary_max", "salary_currency",
    "salary_period", "salary_from_monthly", "extracted_at",
]
DERIVED_COLUMNS = ["published_on", "description_words"]
WITHHELD = {"description_raw", "payload_raw"}


def _write(df: pd.DataFrame, name: str) -> dict:
    path = OUT / name
    # mtime=0 so the same data gives the same bytes, and the same sha256.
    with gzip.GzipFile(path, "wb", mtime=0) as raw, \
            io.TextIOWrapper(raw, encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)
    return {"rows": len(df), "columns": list(df.columns),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def export() -> dict:
    load_dotenv(REPO / ".env")
    with psycopg.connect(os.environ["SUPABASE_DB_URL"]) as conn:
        postings = _query(conn, f"SELECT {', '.join(POSTING_COLUMNS)}, description_raw"
                                " FROM postings ORDER BY posting_key")
        obs = _query(conn, "SELECT posting_key, observed_on, content_hash FROM observations"
                           " ORDER BY posting_key, observed_on")
        skills = _query(conn, "SELECT posting_key, skill FROM posting_skills"
                              " ORDER BY posting_key, skill")
    published, _ = load_lifecycle()
    postings["description_words"] = [len(clean_text(d).split())
                                     for d in postings.description_raw]
    postings = postings.drop(columns="description_raw").merge(published, on="posting_key",
                                                              how="left")
    postings["cities"] = postings.cities.map(lambda c: "|".join(c or []))
    postings = postings[POSTING_COLUMNS + DERIVED_COLUMNS]
    assert not WITHHELD & set(postings.columns)

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "panel_start": str(obs.observed_on.min()), "panel_end": str(obs.observed_on.max()),
        "licence": "CC-BY-4.0 (data); see LICENSE-DATA",
        "dictionary": "DATA_DICTIONARY.md",
        "withheld": sorted(WITHHELD),
        "files": {
            "postings.csv.gz": _write(postings, "postings.csv.gz"),
            "observations.csv.gz": _write(obs, "observations.csv.gz"),
            "posting_skills.csv.gz": _write(skills, "posting_skills.csv.gz"),
        },
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    m = export()
    print(f"panel {m['panel_start']} → {m['panel_end']}; wrote {OUT.relative_to(REPO)}/")
    for name, f in m["files"].items():
        print(f"  {name:<24} {f['rows']:>7} rows  sha256 {f['sha256'][:12]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

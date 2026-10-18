"""Shared data access for the analyses.

Every analysis starts from the same frame so their samples are comparable:
one row per posting with extracted fields, plus the in-scope flag (technical
IC roles, see extraction/titles.py). Analyses filter from here and say how.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv

from extraction.titles import in_scope

REPO = Path(__file__).resolve().parent.parent
FIGURES = REPO / "report" / "figures"


def _query(conn, sql: str) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql)
        cols = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def load_lifecycle() -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (per-posting ATS publish date, observations).

    published_on is the employer's own publish timestamp from the raw payload
    (Greenhouse first_published, Ashby publishedAt), as an IST date. It gives
    postings already live on panel day 1 a known age at entry, which is what
    lets survival analysis handle left truncation instead of dropping them.
    """
    import gzip
    import json

    load_dotenv(REPO / ".env")
    with psycopg.connect(os.environ["SUPABASE_DB_URL"]) as conn:
        raw = _query(conn, "SELECT posting_key, source, payload_raw FROM postings")
        obs = _query(conn, "SELECT posting_key, observed_on FROM observations")
    published = []
    for key, source, payload_raw in raw.itertuples(index=False):
        p = json.loads(gzip.decompress(payload_raw))
        ts = p.get("first_published") if source == "greenhouse" else p.get("publishedAt")
        published.append((
            key,
            pd.Timestamp(ts).tz_convert("Asia/Kolkata").date() if ts else None,
        ))
    return pd.DataFrame(published, columns=["posting_key", "published_on"]), obs


def load_postings() -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (postings, posting_skills). postings carries an `in_scope` column."""
    load_dotenv(REPO / ".env")
    with psycopg.connect(os.environ["SUPABASE_DB_URL"]) as conn:
        postings = _query(conn, """
            SELECT posting_key, source, company, title_raw, title_family, seniority,
                   city, is_remote, is_hybrid, is_multi_country,
                   first_seen, last_seen, is_active, closed_on
            FROM postings
        """)
        skills = _query(conn, "SELECT posting_key, skill FROM posting_skills")
    postings["in_scope"] = [
        in_scope(f, s) for f, s in zip(postings.title_family, postings.seniority, strict=True)
    ]
    return postings, skills

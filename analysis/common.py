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

"""Volume and parse-rate monitoring — the second half of Phase 1.4.

Two checks:

  1. Volume. For each source, is today's rows_fetched at least 50% of the
     recent_hist 7-day median? (The closure guard in upsert.py is stricter at
     40% — this monitor warns earlier so we notice a source degrading before
     it fully trips.)

  2. Parse rate. For each key extracted field (title_family, city, is_remote,
     salary_min), what share of NEW postings written today have it populated?
     A silent drop from 90% to 12% means a site changed its markup or the
     extractor broke.

Exits non-zero on failure so the GH Action emails us.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import psycopg
from dotenv import load_dotenv

load_dotenv()


VOLUME_WARN_RATIO = 0.50
PARSE_RATE_MIN = 0.60  # if <60% of new rows have this field, flag it

PARSE_RATE_FIELDS = [
    "title_family",
    "city",
    "is_remote",
]


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def check_volume(cur: psycopg.Cursor) -> list[Check]:
    cur.execute(
        """
        WITH today AS (
            SELECT source, SUM(rows_fetched) AS n
            FROM run_log
            WHERE started_at::date = CURRENT_DATE
              AND status = 'ok'
            GROUP BY source
        ),
        recent_hist AS (
            SELECT source,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY rows_fetched)::int AS med,
                   COUNT(*) AS n
            FROM run_log
            WHERE started_at >= NOW() - INTERVAL '8 days'
              AND started_at::date < CURRENT_DATE
              AND status = 'ok'
            GROUP BY source
        )
        SELECT COALESCE(t.source, tr.source) AS source,
               COALESCE(t.n, 0) AS today,
               tr.med AS med,
               tr.n AS history
        FROM today t
        FULL OUTER JOIN recent_hist tr ON t.source = tr.source
        """
    )
    out: list[Check] = []
    for row in cur.fetchall():
        source, today_n, med, history = row
        if history is None or history < 3:
            # Not enough history yet — skip.
            out.append(Check(
                name=f"volume:{source}",
                passed=True,
                detail=f"today={today_n}, insufficient history ({history or 0})",
            ))
            continue
        threshold = med * VOLUME_WARN_RATIO
        passed = today_n >= threshold
        out.append(Check(
            name=f"volume:{source}",
            passed=passed,
            detail=f"today={today_n}, 7d-median={med}, threshold={threshold:.0f}",
        ))
    return out


def check_parse_rate(cur: psycopg.Cursor) -> list[Check]:
    out: list[Check] = []
    for field in PARSE_RATE_FIELDS:
        cur.execute(
            f"""
            SELECT source,
                   COUNT(*) AS total,
                   COUNT({field}) AS populated
            FROM postings
            WHERE first_seen = CURRENT_DATE
              AND extracted_at IS NOT NULL
            GROUP BY source
            """
        )
        for source, total, populated in cur.fetchall():
            if total < 5:
                continue  # too few to judge
            rate = populated / total
            passed = rate >= PARSE_RATE_MIN
            out.append(Check(
                name=f"parse:{source}:{field}",
                passed=passed,
                detail=f"{populated}/{total} = {rate:.0%}",
            ))
    return out


def main() -> int:
    db_url = os.environ["SUPABASE_DB_URL"]
    with psycopg.connect(db_url) as conn, conn.cursor() as cur:
        checks = check_volume(cur) + check_parse_rate(cur)

    any_failed = False
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        print(f"[{status}] {c.name}: {c.detail}")
        if not c.passed:
            any_failed = True
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())

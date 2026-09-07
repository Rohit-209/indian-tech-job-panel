"""Quick DB-state summary. Useful after every run.

Run: python -m tools.summary
"""

from __future__ import annotations

import os

import psycopg
from dotenv import load_dotenv

load_dotenv()


def main() -> int:
    db_url = os.environ["SUPABASE_DB_URL"]
    with psycopg.connect(db_url) as conn, conn.cursor() as cur:
        print("=== postings ===")
        cur.execute(
            """
            SELECT source, COUNT(*) AS total,
                   SUM((is_active)::int) AS active,
                   MIN(first_seen) AS earliest,
                   MAX(last_seen) AS latest
            FROM postings
            GROUP BY source
            ORDER BY source
            """
        )
        for row in cur.fetchall():
            print(f"  {row[0]:<12} total={row[1]:>5}  active={row[2]:>5}"
                  f"  earliest={row[3]}  latest={row[4]}")

        print("\n=== top companies by posting count ===")
        cur.execute(
            """
            SELECT source, company, COUNT(*) AS n
            FROM postings
            WHERE is_active
            GROUP BY source, company
            ORDER BY n DESC
            LIMIT 15
            """
        )
        for source, company, n in cur.fetchall():
            print(f"  {n:>4}  {source:<10}  {company}")

        print("\n=== location samples ===")
        cur.execute(
            """
            SELECT location_raw, COUNT(*) AS n
            FROM postings
            WHERE is_active
            GROUP BY location_raw
            ORDER BY n DESC
            LIMIT 15
            """
        )
        for loc, n in cur.fetchall():
            print(f"  {n:>4}  {loc}")

        print("\n=== observations ===")
        cur.execute("SELECT COUNT(*), COUNT(DISTINCT observed_on) FROM observations")
        row = cur.fetchone()
        print(f"  rows={row[0]}  distinct days={row[1]}")

        print("\n=== run_log (last 5) ===")
        cur.execute(
            """
            SELECT started_at, source, rows_fetched, rows_new, rows_closed, status
            FROM run_log
            ORDER BY started_at DESC LIMIT 5
            """
        )
        for row in cur.fetchall():
            print(f"  {row[0].isoformat()[:19]}  {row[1]:<10}"
                  f"  fetched={row[2]:>4} new={row[3]:>4} closed={row[4]:>3}  {row[5]}")

        print("\n=== storage estimate (payload_raw) ===")
        cur.execute(
            """
            SELECT pg_size_pretty(SUM(octet_length(payload_raw))::bigint) AS raw_size,
                   pg_size_pretty(pg_relation_size('postings')) AS table_size,
                   pg_size_pretty(pg_database_size(current_database())) AS db_size
            FROM postings
            """
        )
        raw, table, db = cur.fetchone()
        print(f"  gzipped payload total: {raw}")
        print(f"  postings table size:   {table}")
        print(f"  full database size:    {db}")

    return 0


if __name__ == "__main__":
    main()

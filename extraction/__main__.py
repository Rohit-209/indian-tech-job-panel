"""Run every extractor (skills, titles, geography, salary) over the full history.

Each extractor is a replay (recompute everything, replace its outputs), so
this is safe to run any number of times and is what the daily cron calls
after collection. Titles use the rules classifier until --evaluate shows
the LLM beats it; switch with TITLE_METHOD=llm (needs GEMINI_API_KEY, and
new titles must be cached first with `python -m extraction.titles --label-llm`).

Run: python -m extraction [--dry-run]
"""

from __future__ import annotations

import argparse
import os

import psycopg
from dotenv import load_dotenv

from extraction import geography, salary, skills, titles


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay all extractors")
    parser.add_argument("--dry-run", action="store_true", help="report only, no writes")
    args = parser.parse_args()

    load_dotenv()
    method = os.environ.get("TITLE_METHOD", "rules")
    with psycopg.connect(os.environ["SUPABASE_DB_URL"]) as conn:
        skills.replay(conn, dry_run=args.dry_run)
        print()
        titles.replay(conn, method=method, dry_run=args.dry_run)
        print()
        geography.replay(conn, dry_run=args.dry_run)
        print()
        salary.replay(conn, dry_run=args.dry_run)
        if not args.dry_run:
            with conn.transaction(), conn.cursor() as cur:
                cur.execute("UPDATE postings SET extracted_at = NOW()")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

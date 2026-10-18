"""Apply db/schema.sql to Supabase.

Run: python -m tools.apply_schema
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv()


def main() -> int:
    db_url = os.environ["SUPABASE_DB_URL"]
    schema_path = Path(__file__).parent.parent / "db" / "schema.sql"
    sql = schema_path.read_text()

    with psycopg.connect(db_url) as conn, conn.cursor() as cur:
        cur.execute(sql)
        conn.commit()

        # Sanity check — list what we just created.
        cur.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name IN ('postings', 'posting_skills', 'observations', 'run_log',
                                 'title_llm_cache')
            ORDER BY table_name
            """
        )
        tables = [r[0] for r in cur.fetchall()]

    print("Applied schema. Tables in public schema:")
    for t in tables:
        print(f"  - {t}")
    expected = {"postings", "posting_skills", "observations", "run_log", "title_llm_cache"}
    missing = expected - set(tables)
    if missing:
        print(f"MISSING: {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

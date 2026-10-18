"""Coverage statistics for DATA_DICTIONARY.md.

Prints the coverage section as markdown, or with --write replaces the block
between the COVERAGE markers in DATA_DICTIONARY.md, so the numbers there are
always regenerated, never hand-edited.

Run: python -m tools.coverage [--write]
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import psycopg
from dotenv import load_dotenv

from extraction.skills import load_dictionary
from extraction.titles import in_scope

REPO = Path(__file__).resolve().parent.parent
DICTIONARY = REPO / "DATA_DICTIONARY.md"
START, END = "<!-- COVERAGE:START -->", "<!-- COVERAGE:END -->"


def _rows(cur, sql: str) -> list[tuple]:
    cur.execute(sql)
    return cur.fetchall()


def _dist(rows: list[tuple], n: int) -> str:
    return ", ".join(f"{k if k is not None else 'NULL'} {c} ({c / n:.0%})" for k, c in rows)


def build() -> str:
    load_dotenv(REPO / ".env")
    with psycopg.connect(os.environ["SUPABASE_DB_URL"]) as conn, conn.cursor() as cur:
        (n, active, first, last, days) = _rows(cur, """
            SELECT COUNT(*), SUM(is_active::int), MIN(first_seen), MAX(last_seen),
                   (SELECT COUNT(DISTINCT observed_on) FROM observations)
            FROM postings""")[0]
        n_obs = _rows(cur, "SELECT COUNT(*) FROM observations")[0][0]
        by_source = _rows(cur, "SELECT source, COUNT(*) FROM postings GROUP BY 1 ORDER BY 2 DESC")
        by_company = _rows(cur, "SELECT company, COUNT(*) FROM postings GROUP BY 1 ORDER BY 2 DESC")
        fam_sen = _rows(cur, "SELECT title_family, seniority FROM postings")
        k_scope = sum(in_scope(f, s) for f, s in fam_sen)
        def counts(col: str) -> list[tuple]:
            return _rows(cur, f"SELECT {col}, COUNT(*) FROM postings GROUP BY 1 ORDER BY 2 DESC")

        family, seniority, city = counts("title_family"), counts("seniority"), counts("city")
        workplace = _rows(cur, """
            SELECT CASE WHEN is_remote THEN 'remote' WHEN is_hybrid THEN 'hybrid'
                        WHEN is_remote IS NOT NULL THEN 'onsite' END, COUNT(*)
            FROM postings GROUP BY 1 ORDER BY 2 DESC""")
        wsource = _rows(cur, "SELECT workplace_source, COUNT(*) FROM postings"
                             " GROUP BY 1 ORDER BY 2 DESC")
        skills_any, skill_rows, distinct_skills = _rows(cur, """
            SELECT COUNT(DISTINCT posting_key), COUNT(*), COUNT(DISTINCT skill)
            FROM posting_skills""")[0]
        salary = _rows(cur, "SELECT salary_currency, COUNT(*) FROM postings"
                            " WHERE salary_max IS NOT NULL GROUP BY 1 ORDER BY 2 DESC")
        multi_city = _rows(cur, "SELECT COUNT(*) FROM postings WHERE cardinality(cities) > 1")[0][0]
        multi_country = _rows(cur, "SELECT COUNT(*) FROM postings WHERE is_multi_country")[0][0]

    now = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    d = load_dictionary()
    lines = [
        f"*Generated {now} by `python -m tools.coverage --write`. Do not edit by hand.*",
        "",
        "| | |",
        "|---|---|",
        f"| Panel window | {first} → {last} ({days} collection days) |",
        f"| Postings ever seen | {n} ({active} active, {n - active} closed) |",
        f"| Observations (posting × day) | {n_obs} |",
        f"| By source | {_dist(by_source, n)} |",
        f"| In scope (technical IC) | {k_scope} ({k_scope / n:.0%}) |",
        f"| Employers | {len(by_company)}: {_dist(by_company, n)} |",
        "",
        "| Field | Coverage / distribution |",
        "|---|---|",
        f"| `title_family` | {_dist(family, n)} |",
        f"| `seniority` | {_dist(seniority, n)} |",
        f"| `city` | {_dist(city, n)}; multi-city {multi_city}, multi-country {multi_country} |",
        f"| workplace (`is_remote`/`is_hybrid`) | {_dist(workplace, n)} |",
        f"| `workplace_source` | {_dist(wsource, n)} |",
        f"| skills | {skills_any} postings with ≥ 1 skill ({skills_any / n:.0%}); "
        f"{skill_rows} posting–skill rows; {distinct_skills} of {len(d.skills)} dictionary "
        f"skills observed (dictionary v{d.version}) |",
        f"| salary | {sum(c for _, c in salary)} postings ({sum(c for _, c in salary) / n:.1%}): "
        f"{_dist(salary, n)} |",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write", action="store_true", help="update DATA_DICTIONARY.md")
    args = parser.parse_args()
    block = build()
    if not args.write:
        print(block)
        return 0
    text = DICTIONARY.read_text()
    new = re.sub(re.escape(START) + r".*?" + re.escape(END),
                 f"{START}\n{block}\n{END}", text, flags=re.S)
    if new == text and START not in text:
        raise SystemExit(f"markers {START} / {END} not found in {DICTIONARY.name}")
    DICTIONARY.write_text(new)
    print(f"updated coverage section in {DICTIONARY.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

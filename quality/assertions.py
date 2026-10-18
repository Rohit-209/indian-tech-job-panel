"""Data-quality assertions that run after each collector + extraction cycle.

Fail loudly (non-zero exit) so the GitHub Action emails us. Many of these
duplicate DB-level CHECK constraints — that's deliberate. The CHECKs prevent
bad writes; these SELECTs catch drift we introduced by loosening a constraint
later or by extractor bugs that write into an unconstrained column.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import psycopg
import structlog
from dotenv import load_dotenv

from extraction.titles import FAMILIES, SENIORITIES

load_dotenv()
log = structlog.get_logger()


@dataclass
class AssertionResult:
    name: str
    passed: bool
    detail: str = ""


ASSERTIONS: list[tuple[str, str]] = [
    (
        "no_negative_salaries",
        """
        SELECT COUNT(*) FROM postings
        WHERE (salary_min IS NOT NULL AND salary_min < 0)
           OR (salary_max IS NOT NULL AND salary_max < 0)
        """,
    ),
    (
        "first_seen_le_last_seen",
        "SELECT COUNT(*) FROM postings WHERE first_seen > last_seen",
    ),
    (
        "closed_iff_closed_on",
        """
        SELECT COUNT(*) FROM postings
        WHERE (is_active = FALSE AND closed_on IS NULL)
           OR (is_active = TRUE  AND closed_on IS NOT NULL)
        """,
    ),
    (
        "observation_never_before_first_seen",
        """
        SELECT COUNT(*) FROM observations o
        JOIN postings p USING (posting_key)
        WHERE o.observed_on < p.first_seen
        """,
    ),
    (
        "skill_count_plausible",
        # A posting with >80 skills is almost certainly a false-positive
        # explosion in the dictionary (e.g. matching every 2-letter token).
        # Catch it before it distorts A1 clustering.
        """
        SELECT COUNT(*) FROM (
            SELECT posting_key, COUNT(*) AS n
            FROM posting_skills
            GROUP BY posting_key
            HAVING COUNT(*) > 80
        ) x
        """,
    ),
    (
        "salary_min_le_max",
        """
        SELECT COUNT(*) FROM postings
        WHERE salary_min IS NOT NULL
          AND salary_max IS NOT NULL
          AND salary_min > salary_max
        """,
    ),
    # --- Extracted fields (python -m extraction runs before these) ----------
    # Each one backs a promise made in DATA_DICTIONARY.md.
    (
        "closure_after_last_sighting",
        # closed_on == last_seen is tolerated: panel day 1 had two runs, and
        # the second "closed" five postings the first had just seen.
        "SELECT COUNT(*) FROM postings WHERE closed_on < last_seen",
    ),
    (
        "title_family_in_vocabulary",
        "SELECT COUNT(*) FROM postings WHERE extracted_at IS NOT NULL"
        " AND (title_family IS NULL OR title_family NOT IN ("
        + ", ".join(f"'{f}'" for f in FAMILIES) + "))",
    ),
    (
        "seniority_in_vocabulary",
        "SELECT COUNT(*) FROM postings WHERE extracted_at IS NOT NULL"
        " AND (seniority IS NULL OR seniority NOT IN ("
        + ", ".join(f"'{s}'" for s in SENIORITIES) + "))",
    ),
    (
        "workplace_fields_consistent",
        # Unknown workplace is NULL in all three columns together; known is
        # exactly one of remote / hybrid / onsite (both flags false).
        """
        SELECT COUNT(*) FROM postings
        WHERE (is_remote IS NULL) <> (workplace_source IS NULL)
           OR (is_remote IS NULL) <> (is_hybrid IS NULL)
           OR (is_remote AND is_hybrid)
        """,
    ),
    (
        "city_is_first_of_cities",
        """
        SELECT COUNT(*) FROM postings
        WHERE extracted_at IS NOT NULL
          AND ((city IS NULL) <> (cardinality(cities) = 0)
               OR (city IS NOT NULL AND city <> cities[1]))
        """,
    ),
    (
        "salary_has_currency_and_period",
        """
        SELECT COUNT(*) FROM postings
        WHERE salary_max IS NOT NULL
          AND (salary_currency IS NULL OR salary_period IS DISTINCT FROM 'annual')
        """,
    ),
]


def run() -> list[AssertionResult]:
    db_url = os.environ["SUPABASE_DB_URL"]
    results: list[AssertionResult] = []
    with psycopg.connect(db_url) as conn, conn.cursor() as cur:
        for name, sql in ASSERTIONS:
            cur.execute(sql)
            row = cur.fetchone()
            bad = row[0] if row else 0
            results.append(
                AssertionResult(name=name, passed=(bad == 0), detail=f"{bad} rows failed")
            )
    return results


def main() -> int:
    results = run()
    any_failed = False
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.name}: {r.detail}")
        if not r.passed:
            any_failed = True
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())

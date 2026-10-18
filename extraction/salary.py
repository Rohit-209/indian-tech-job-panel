"""Salary: disclosed pay range -> annual min/max with currency.

Disclosure in this panel is rare (see report/A4_disclosure.md), so this is a
small parser, not a model. What it handles:

  "India Annual Pay Range ₹1,370,000 — ₹1,790,000 INR"   (Airbnb)
  "INR 12,00,000 - 18,00,000"                              (Indian digit grouping)
  "12-18 LPA", "₹15 lakh", "up to 25 LPA"                  (lakh notation)
  "₹80,000 per month"                                      (monthly -> annual, flagged)
  "$150,000 — $200,000 USD"                                (parsed, currency USD)

What it deliberately ignores: ATS salary custom fields. Okta and MongoDB
carry "Minimum Salary" / "Job Post Range" fields on every posting, but the
values are placeholders (0.0, USD 0.01). A zero is "not disclosed", never a
salary of zero.

Run:
  python -m extraction.salary --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from dataclasses import dataclass

from extraction.text import clean_text

_LAKH = 100_000
_NUM = r"\d[\d,]*(?:\.\d+)?"
_DASH = r"\s*(?:-|–|—|to)\s*"
_CUR = r"(?:₹|INR|Rs\.?|\$|USD)"

# A range with an explicit currency on at least one side.
_RANGE = re.compile(
    rf"(?P<c1>{_CUR})?\s*(?P<lo>{_NUM})\s*(?P<u1>lpa|lakhs?|l\b|k\b)?{_DASH}"
    rf"(?P<c2>{_CUR})?\s*(?P<hi>{_NUM})\s*(?P<u2>lpa|lakhs?|l\b|k\b)?\s*(?P<c3>INR|USD)?"
    rf"(?P<per>\s*(?:per|/|a)\s*(?:month|annum|year|yr|mo)\b|\s*monthly|\s*annually)?",
    re.I,
)
# Single value: "up to 25 LPA", "₹15 lakh".
_SINGLE = re.compile(
    rf"(?P<cap>up to\s*)?(?P<c1>{_CUR})?\s*(?P<v>{_NUM})\s*(?P<u>lpa|lakhs?)\b", re.I)
_PAY_CONTEXT = re.compile(r"\b(?:pay|salary|compensation|ctc|package|stipend|base)\b", re.I)


@dataclass(frozen=True)
class Salary:
    min: float | None
    max: float | None
    currency: str | None
    period: str | None             # always 'annual' after conversion
    from_monthly: bool | None


NONE = Salary(None, None, None, None, None)


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _currency(*tokens: str | None) -> str | None:
    for t in tokens:
        if not t:
            continue
        t = t.upper().rstrip(".")
        if t in ("₹", "INR", "RS"):
            return "INR"
        if t in ("$", "USD"):
            return "USD"
    return None


def _scale(v: float, unit: str | None) -> float:
    u = (unit or "").lower()
    if u in ("lpa", "lakh", "lakhs", "l"):
        return v * _LAKH
    if u == "k":
        return v * 1_000
    return v


def parse_salary(text: str | None) -> Salary:
    t = clean_text(text)
    for m in _RANGE.finditer(t):
        unit_lo, unit_hi = m["u1"] or m["u2"], m["u2"] or m["u1"]
        currency = _currency(m["c1"], m["c2"], m["c3"])
        # Lakh units imply INR; otherwise demand an explicit currency so
        # "5-7 years" and "3-5 days a week" never parse as pay.
        if currency is None and (unit_lo or "").lower() in ("lpa", "lakh", "lakhs"):
            currency = "INR"
        if currency is None:
            continue
        lo, hi = _scale(_num(m["lo"]), unit_lo), _scale(_num(m["hi"]), unit_hi)
        if lo <= 0 or hi <= 0:  # placeholders
            continue
        if lo > hi:
            lo, hi = hi, lo
        monthly = bool(m["per"]) and re.search(r"month|mo\b", m["per"], re.I) is not None
        if monthly:
            lo, hi = lo * 12, hi * 12
        if currency == "INR" and hi < 50_000:  # too small to be annual pay
            continue
        return Salary(lo, hi, currency, "annual", monthly)

    for m in _SINGLE.finditer(t):
        window = t[max(m.start() - 80, 0):m.end() + 20]
        if not _PAY_CONTEXT.search(window):
            continue
        v = _scale(_num(m["v"]), m["u"])
        if v <= 0:
            continue
        lo = None if m["cap"] else v
        return Salary(lo, v, "INR", "annual", False)
    return NONE


# --- DB replay --------------------------------------------------------------


def replay(conn, dry_run: bool = False) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT posting_key, company, description_raw FROM postings")
        rows = cur.fetchall()
    parsed = {k: parse_salary(d) for k, _c, d in rows}
    company = {k: c for k, c, _d in rows}

    disclosed = {k: s for k, s in parsed.items() if s.max is not None}
    print(f"=== salary ({len(rows)} postings) ===")
    print(f"  disclosed: {len(disclosed)} ({len(disclosed) / max(len(rows), 1):.1%})")
    by = Counter((company[k], s.currency) for k, s in disclosed.items())
    for (co, cur_), n in by.most_common():
        print(f"    {n:>4}  {co} ({cur_})")
    if dry_run:
        print("\n(dry run — postings unchanged)")
        return

    with conn.transaction(), conn.cursor() as cur:
        cur.execute(
            "CREATE TEMP TABLE salary_assign (posting_key TEXT, salary_min NUMERIC,"
            " salary_max NUMERIC, salary_currency TEXT, salary_period TEXT,"
            " salary_from_monthly BOOLEAN) ON COMMIT DROP"
        )
        with cur.copy("COPY salary_assign FROM STDIN") as copy:
            for k, s in parsed.items():
                copy.write_row((k, s.min, s.max, s.currency, s.period, s.from_monthly))
        cur.execute(
            """
            UPDATE postings p
            SET salary_min = a.salary_min, salary_max = a.salary_max,
                salary_currency = a.salary_currency, salary_period = a.salary_period,
                salary_from_monthly = a.salary_from_monthly
            FROM salary_assign a
            WHERE p.posting_key = a.posting_key
            """
        )
    print(f"\nwrote salary fields for {len(parsed)} postings")


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

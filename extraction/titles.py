"""Title normalisation: raw title -> (title_family, seniority).

Two classifiers share one taxonomy:

  rules  — ordered keyword patterns below. Free, deterministic, auditable.
  llm    — Gemini over (company, title), cached in title_llm_cache
           (extraction/titles_llm.py).

Neither is trusted on faith. Both are scored against a blind hand-labelled
sample (data/title_labels.csv) with `--evaluate`, and the replay uses
whichever wins. If the LLM doesn't clearly beat 60 lines of regex, it
doesn't earn its place in the pipeline.

Family is *function*, seniority is *level*, and they're independent: an
Engineering Manager is (software_engineering, manager). The project's scope
filter — technical IC roles — is then `in_scope()`, not a family of its own.

Run:
  python -m extraction.titles --dry-run               # rules over all postings, report
  python -m extraction.titles --sample-labels 200     # write blind labelling sheet
  python -m extraction.titles --label-llm             # fill the Gemini cache
  python -m extraction.titles --evaluate              # score rules + llm vs hand labels
  python -m extraction.titles --method rules          # replay into postings
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
from collections import Counter
from pathlib import Path

# --- Taxonomy ---------------------------------------------------------------
# These descriptions are the labelling guide AND the LLM prompt — one source
# of truth, so the model and the human labeller are scored on the same rules.

FAMILIES: dict[str, str] = {
    "software_engineering": "Builds software: backend, frontend, full-stack, mobile, "
        "embedded, architects, business-systems developers (Salesforce/ServiceNow devs).",
    "sre_infra": "Site reliability, DevOps, platform, cloud, infrastructure, network and "
        "production/release engineering.",
    "data_engineering": "Data pipelines, data platforms, analytics engineering, big data.",
    "data_science": "Data scientists, applied/research scientists, statisticians, "
        "decision scientists.",
    "ml_engineering": "Machine learning / AI engineers who build and ship models or "
        "LLM systems, ML platform, research engineers.",
    "data_analytics": "Data, product, business, BI and people analysts; analytics leads.",
    "qa_testing": "QA, test automation, SDET, software engineer in test.",
    "security_engineering": "Security engineers, AppSec, detection & response, "
        "security architects and analysts.",
    "solutions_support": "Customer-facing technical roles: solutions/sales/forward-deployed "
        "engineers and architects, technical support and services engineers, technical "
        "account managers, implementation and professional services.",
    "product_design": "Product and program managers, designers, UX researchers, "
        "technical writers.",
    "non_tech": "Everything else: sales, marketing, finance, operations, recruiting, HR, "
        "legal, admin, account executives.",
}

SENIORITIES: dict[str, str] = {
    "intern": "Interns, apprentices, trainees.",
    "junior": "Junior, associate, entry-level, graduate, or level I / 1 / L1.",
    "mid": "Level II / 2 / L2 with no other marker.",
    "senior": "Senior / Sr, or level III / 3 / L3.",
    "staff": "Staff, lead / tech lead, architect, or level IV / 4 / L4.",
    "principal": "Principal, distinguished, fellow, or level V / 5 / L5.",
    "manager": "People managers: engineering manager, 'Manager, X', team lead of a "
        "non-IC team. NOT product/program/account/project managers, who are ICs.",
    "director": "Director, head of, VP, chief.",
    "unspecified": "No level marker at all (e.g. plain 'Software Engineer'). Do not guess.",
}

TECH_FAMILIES = frozenset({
    "software_engineering", "sre_infra", "data_engineering", "data_science",
    "ml_engineering", "data_analytics", "qa_testing", "security_engineering",
})
MANAGEMENT = frozenset({"manager", "director"})


def in_scope(family: str | None, seniority: str | None) -> bool:
    """Project scope: full-time technical IC roles."""
    return family in TECH_FAMILIES and seniority not in MANAGEMENT and seniority != "intern"


# --- Rules baseline ---------------------------------------------------------
# First match wins, so order is the logic: specific functions before the
# generic "engineer" catch-all, and "software engineer in test" before SWE.

def _rx(*alts: str) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(alts) + r")\b", re.I)


_FAMILY_RULES: list[tuple[str, re.Pattern[str]]] = [
    # Job names that contain technical words but aren't technical roles
    # ("Senior Product Manager, Data Platform", "Director of Recruiting, Engineering").
    ("product_design", _rx(r"product manag\w+", r"program manager", r"product owner",
                           r"technical writer", r"designer", r"ux researcher")),
    ("non_tech", _rx(r"recruit\w*", r"sourcer", r"talent")),
    ("qa_testing", _rx(r"engineer in test", r"sdet", r"qa", r"quality assurance",
                       r"test(?:ing)? (?:engineer|automation|lead)", r"automation test\w*")),
    ("security_engineering", _rx(r"security (?:engineer|architect|analyst|researcher)\w*",
                                 r"appsec", r"application security", r"infosec",
                                 r"detection (?:and|&) response", r"detection engineer\w*",
                                 r"penetration", r"threat (?:research|intel\w*|hunt\w*)")),
    ("ml_engineering", _rx(r"machine learning", r"ml", r"ai engineer\w*", r"ai/ml",
                           r"llm", r"research engineer\w*", r"mlops")),
    ("data_science", _rx(r"data scien\w+", r"applied scientist", r"research scientist",
                         r"statistician", r"decision scien\w+")),
    ("data_engineering", _rx(r"data engineer\w*", r"analytics engineer\w*", r"etl",
                             r"big data", r"data platform")),
    ("data_analytics", _rx(r"(?:data|product|business|bi|people|marketing) analyst\w*",
                           r"business intelligence", r"analytics")),
    ("sre_infra", _rx(r"site reliability", r"sre", r"devops", r"platform engineer\w*",
                      r"infrastructure", r"cloud engineer\w*", r"network engineer\w*",
                      r"systems engineer\w*", r"production engineer\w*",
                      r"release engineer\w*", r"build engineer\w*",
                      r"operations engineer\w*")),
    # Before SWE so "Developer Support Engineer" isn't a developer.
    ("solutions_support", _rx(r"support engineer\w*", r"tse",
                              r"technical (?:support|services)")),
    # Unambiguous SWE titles beat customer-facing context
    # ("Software Engineer, Professional Services R&D" is a SWE).
    ("software_engineering", _rx(r"software engineer\w*", r"software developer\w*",
                                 r"developer", r"sde", r"swe", r"programmer",
                                 r"(?:back|front)[- ]?end", r"full[- ]?stack",
                                 r"member of technical staff", r"tech lead")),
    ("solutions_support", _rx(r"solutions?", r"customer engineer\w*", r"consult\w*",
                              r"field engineer\w*", r"forward deployed\w*",
                              r"sales engineer\w*", r"pre-?sales", r"implementation",
                              r"professional services", r"technical account manager",
                              r"deployment (?:engineer|manager)\w*")),
    ("software_engineering", _rx(r"engineer\w*", r"engineering", r"architect\w*",
                                 r"mobile", r"ios", r"android")),
    ("product_design", _rx(r"product manager", r"program manager", r"tpm",
                           r"product owner", r"designer", r"design", r"ux", r"ui",
                           r"technical writer", r"documentation")),
]

# Titles where "manager" is the job name, not a people-management level.
_IC_MANAGER = re.compile(
    r"\b(?:product|program|project|account|success|renewals?|revenue|office|"
    r"partner|marketing|operations|services|solutions|deployment|tools)\s+managers?\b",
    re.I,
)

_SENIORITY_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("intern", _rx(r"intern", r"internship", r"apprentice\w*", r"trainee")),
    ("director", _rx(r"director", r"head of", r"vp", r"vice president", r"chief", r"cto")),
    ("principal", _rx(r"principal", r"distinguished", r"fellow")),
    ("staff", _rx(r"staff")),
    ("senior", _rx(r"senior", r"sr")),
    ("junior", _rx(r"junior", r"jr", r"associate", r"entry[- ]level", r"graduate",
                   r"new grad", r"fresher")),
    ("mid", _rx(r"intermediate", r"mid[- ]level")),
]

# Company ladders: "Software Engineer 3, Atlas", "TSE II", "Software Engineer (L2)".
# A bare number only counts right after a role noun; "L<n>" counts anywhere.
_LADDER = re.compile(
    r"\b(?:engineer|tse|analyst|specialist|developer|executive|scientist|architect)"
    r"\s*\(?(i{1,3}|iv|v|[1-5])\b"
    r"|\(?\bl([1-6])\b\)?",
    re.I,
)
_LADDER_LEVEL = {
    "i": "junior", "1": "junior",
    "ii": "mid", "2": "mid",
    "iii": "senior", "3": "senior",
    "iv": "staff", "4": "staff",
    "v": "principal", "5": "principal", "6": "principal",
}
_LEAD = _rx(r"lead", r"tech lead", r"architect\w*")
_MANAGER = _rx(r"managers?", r"mgr")


def _strip_qualifiers(title: str) -> str:
    # Drop location/team suffixes that carry no level information but can
    # end in digits or roman numerals ("... - India", "(Mandarin Speaking)").
    return re.sub(r"\s*[-–|]\s*(?:india|bengaluru|bangalore|gurugram|gurgaon|mumbai|"
                  r"pune|hyderabad|delhi|remote)\b.*$", "", title, flags=re.I).strip()


def rules_family(title: str) -> str:
    for family, rx in _FAMILY_RULES:
        if rx.search(title):
            return family
    return "non_tech"


def rules_seniority(title: str) -> str:
    t = _strip_qualifiers(title)
    for level, rx in _SENIORITY_RULES[:2]:  # intern, director
        if rx.search(t):
            return level
    if _MANAGER.search(t) and not _IC_MANAGER.search(t):
        return "manager"
    for level, rx in _SENIORITY_RULES[2:]:
        if rx.search(t):
            return level
    if m := _LADDER.search(t):
        return _LADDER_LEVEL[(m.group(1) or m.group(2)).lower()]
    if _LEAD.search(t):
        return "staff"
    return "unspecified"


def classify_rules(title: str | None) -> tuple[str, str]:
    t = (title or "").strip()
    return rules_family(t), rules_seniority(t)


# --- Hand labels + evaluation ------------------------------------------------

# Rules were tuned by reading every title seen up to this date, so a hand
# label on one of those titles scores the rules in-sample. Labels on titles
# first seen after it ("holdout") are the fair rules-vs-LLM comparison.
RULES_FROZEN_ON = "2026-10-18"
LABELS_PATH = Path(__file__).resolve().parent.parent / "data" / "title_labels.csv"
_LABEL_FIELDS = ["split", "company", "title", "title_family", "seniority", "notes"]


def _distinct_titles(conn, since: str | None = None) -> list[tuple[str, str]]:
    sql = "SELECT DISTINCT company, title_raw FROM postings"
    params: tuple = ()
    if since:
        sql += " WHERE first_seen > %s"
        params = (since,)
    with conn.cursor() as cur:
        cur.execute(sql + " ORDER BY company, title_raw", params)
        return cur.fetchall()


def _read_labels() -> list[dict[str, str]]:
    if not LABELS_PATH.exists():
        return []
    with LABELS_PATH.open(newline="") as f:
        return list(csv.DictReader(f))


def sample_labels(conn, n: int, since: str | None, seed: int = 20261018) -> int:
    """Append n unlabelled titles to the labelling sheet (blind: no predictions)."""
    from extraction.titles_llm import title_key

    existing = _read_labels()
    taken = {title_key(r["company"], r["title"]) for r in existing}
    pool = [p for p in _distinct_titles(conn, since) if title_key(*p) not in taken]
    picked = random.Random(seed).sample(pool, min(n, len(pool)))
    split = "holdout" if since and since >= RULES_FROZEN_ON else "dev"

    LABELS_PATH.parent.mkdir(exist_ok=True)
    write_header = not LABELS_PATH.exists()
    with LABELS_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_LABEL_FIELDS)
        if write_header:
            w.writeheader()
        for company, title in sorted(picked):
            w.writerow({"split": split, "company": company, "title": title,
                        "title_family": "", "seniority": "", "notes": ""})
    return len(picked)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def _mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: b = only A right, c = only B right."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def _score(name: str, gold: list[tuple[str, str]], pred: list[tuple[str, str]]) -> list[bool]:
    n = len(gold)
    fam = [g[0] == p[0] for g, p in zip(gold, pred, strict=True)]
    sen = [g[1] == p[1] for g, p in zip(gold, pred, strict=True)]
    both = [f and s for f, s in zip(fam, sen, strict=True)]
    g_in = [in_scope(*g) for g in gold]
    p_in = [in_scope(*p) for p in pred]
    tp = sum(a and b for a, b in zip(g_in, p_in, strict=True))
    prec = tp / max(sum(p_in), 1)
    rec = tp / max(sum(g_in), 1)

    print(f"  {name}")
    for label, hits in (("family", fam), ("seniority", sen), ("both", both)):
        k = sum(hits)
        lo, hi = _wilson(k, n)
        print(f"    {label:<10} {k:>3}/{n}  {k / n:.0%}  (95% CI {lo:.0%}–{hi:.0%})")
    print(f"    in_scope   precision {prec:.0%}  recall {rec:.0%}")
    errors = Counter((g[0], p[0]) for g, p in zip(gold, pred, strict=True) if g[0] != p[0])
    if errors:
        top = ", ".join(f"{g}→{p} ×{c}" for (g, p), c in errors.most_common(5))
        print(f"    top family errors: {top}")
    return both


def evaluate(conn) -> int:
    from extraction.titles_llm import load_cache, model_name, title_key

    labels = [r for r in _read_labels() if r["title_family"] and r["seniority"]]
    if not labels:
        print(f"no completed labels in {LABELS_PATH}")
        return 1
    bad = [r for r in labels
           if r["title_family"] not in FAMILIES or r["seniority"] not in SENIORITIES]
    if bad:
        for r in bad:
            print(f"  invalid label: {r}")
        return 1

    cache = load_cache(conn)
    for split in ("dev", "holdout"):
        rows = [r for r in labels if r["split"] == split]
        if not rows:
            continue
        note = "rules scored in-sample" if split == "dev" else "fair comparison"
        print(f"=== {split} (n={len(rows)}, {note}) ===")
        gold = [(r["title_family"], r["seniority"]) for r in rows]
        rules_ok = _score("rules", gold, [classify_rules(r["title"]) for r in rows])

        keys = [title_key(r["company"], r["title"]) for r in rows]
        if all(k in cache for k in keys):
            llm_ok = _score(f"llm ({model_name()})", gold, [cache[k] for k in keys])
            b = sum(r and not m for r, m in zip(rules_ok, llm_ok, strict=True))
            c = sum(m and not r for r, m in zip(rules_ok, llm_ok, strict=True))
            print(f"  paired: only rules right {b}, only llm right {c},"
                  f" exact McNemar p={_mcnemar_exact(b, c):.3f}")
        else:
            print("  llm: not every labelled title is cached — run --label-llm")
    return 0


# --- Replay -----------------------------------------------------------------


def _fetch_postings(conn) -> list[tuple[str, str, str]]:
    with conn.cursor() as cur:
        cur.execute("SELECT posting_key, company, title_raw FROM postings")
        return cur.fetchall()


def _write_titles(conn, rows: list[tuple[str, str, str]]) -> None:
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(
            "CREATE TEMP TABLE title_assign"
            " (posting_key TEXT, title_family TEXT, seniority TEXT) ON COMMIT DROP"
        )
        with cur.copy("COPY title_assign FROM STDIN") as copy:
            for row in rows:
                copy.write_row(row)
        cur.execute(
            """
            UPDATE postings p
            SET title_family = a.title_family, seniority = a.seniority
            FROM title_assign a
            WHERE p.posting_key = a.posting_key
            """
        )


def _report(assigned: dict[str, tuple[str, str]]) -> None:
    n = len(assigned)
    print(f"=== titles ({n} postings) ===")
    for label, idx in (("title_family", 0), ("seniority", 1)):
        print(f"\n  {label}:")
        for value, c in Counter(v[idx] for v in assigned.values()).most_common():
            print(f"    {c:>5}  {c / n:>4.0%}  {value}")
    k = sum(in_scope(*v) for v in assigned.values())
    print(f"\n  in scope (technical IC): {k} ({k / n:.0%})")


def main() -> int:
    import psycopg
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="report only, no writes")
    mode.add_argument("--guide", action="store_true", help="print the labelling guide")
    mode.add_argument("--sample-labels", type=int, metavar="N",
                      help="append N blind titles to data/title_labels.csv")
    mode.add_argument("--label-llm", action="store_true", help="fill the Gemini cache")
    mode.add_argument("--evaluate", action="store_true", help="score vs hand labels")
    parser.add_argument("--since", help="with --sample-labels: only titles first seen after")
    parser.add_argument("--method", choices=["rules", "llm"], default="rules",
                        help="classifier for the replay / dry run")
    args = parser.parse_args()

    if args.guide:
        print("title_family:")
        for k, v in FAMILIES.items():
            print(f"  {k:<22} {v}")
        print("\nseniority:")
        for k, v in SENIORITIES.items():
            print(f"  {k:<22} {v}")
        return 0

    load_dotenv()
    with psycopg.connect(os.environ["SUPABASE_DB_URL"]) as conn:
        if args.sample_labels:
            n = sample_labels(conn, args.sample_labels, args.since)
            print(f"appended {n} titles to {LABELS_PATH}")
            return 0
        if args.evaluate:
            return evaluate(conn)

        from extraction import titles_llm

        postings = _fetch_postings(conn)
        if args.label_llm:
            added = titles_llm.fill_cache(conn, [(c, t) for _k, c, t in postings])
            print(f"cached {added} new titles ({titles_llm.model_name()},"
                  f" prompt v{titles_llm.PROMPT_VERSION})")
            return 0

        if args.method == "llm":
            cache = titles_llm.load_cache(conn)
            missing = sum(titles_llm.title_key(c, t) not in cache for _k, c, t in postings)
            if missing:
                raise SystemExit(f"{missing} postings have uncached titles — run --label-llm")
            assigned = {k: cache[titles_llm.title_key(c, t)] for k, c, t in postings}
        else:
            assigned = {k: classify_rules(t) for k, _c, t in postings}

        _report(assigned)
        if args.dry_run:
            print("\n(dry run — postings unchanged)")
            return 0
        _write_titles(conn, [(k, f, s) for k, (f, s) in assigned.items()])
        print(f"\nwrote title_family/seniority for {len(assigned)} postings ({args.method})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Skill extraction: regex dictionary over title + description.

Deterministic, auditable, free. Every skill in extraction/skills.yaml compiles
to one regex; a posting gets a skill if that regex matches anywhere in its
title + description, unless the posting's employer is on the skill's
suppress_for list. Company boilerplate (intros, EEO, legal notices) is removed
first — see extraction/text.py — so skills reflect the role, not the employer.

Replay, not increment: each run recomputes skills for every posting and
replaces posting_skills in a single transaction. The table therefore always
reflects exactly one dictionary version, and a dictionary fix is applied to
the full history by re-running — never by hand-editing rows. Because
boilerplate is learned from the whole corpus, a posting's skills can shift
slightly as the corpus grows; replay keeps every posting on the same footing.

Run:
  python -m extraction.skills                  # replay into posting_skills
  python -m extraction.skills --dry-run        # compute + report, no writes
  python -m extraction.skills --explain go     # show match contexts for audit
  python -m extraction.skills --boilerplate    # show segments dropped per company
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from extraction.text import clean_text, find_boilerplate, strip_boilerplate

DICTIONARY_PATH = Path(__file__).with_name("skills.yaml")

# A match can't sit inside a larger word. The right boundary also rejects
# + and # so `c` doesn't fire on C++ / C#, and `java` doesn't on JavaScript.
_LEFT = r"(?<![A-Za-z0-9_])"
_RIGHT = r"(?![A-Za-z0-9_+#])"

# `listed` tokens (Go, R, C) only count inside an enumeration: right after a
# separator / connective, or right before one.
_LIST_BEFORE = (
    r"(?:(?<=[,/(])|(?<=[,/(] )|(?<=: )|(?<= and )|(?<= or )|(?<= in )"
    r"|(?<= with )|(?<= using )|(?<= like )|(?<=e\.g\. ))"
)
_LIST_AFTER = r"(?=\s?[,/)]|\s(?:and|or)\s)"
# ...and never as Go-to-market, R&D, C-suite, Series C, "Go to".
_LIST_REJECT_AFTER = r"(?![-&']| to\b)"
_LIST_REJECT_BEFORE = r"(?<!Series )"


@dataclass(frozen=True)
class Skill:
    slug: str
    category: str
    regex: re.Pattern[str]
    suppress_for: frozenset[str]


@dataclass(frozen=True)
class SkillDictionary:
    version: int
    skills: tuple[Skill, ...]


def _literal(s: str) -> str:
    # Spaces in an alias also match a hyphen or nothing:
    # "spring boot" ~ "spring-boot" ~ "springboot".
    return r"[\s\-]?".join(re.escape(part) for part in s.split(" "))


def _compile(slug: str, spec: dict) -> re.Pattern[str]:
    alts: list[str] = []
    for a in spec.get("aliases", []):
        alts.append(f"(?i:{_LEFT}{_literal(str(a))}{_RIGHT})")
    for e in spec.get("exact", []):
        alts.append(f"{_LEFT}{_literal(str(e))}{_RIGHT}")
    for t in spec.get("listed", []):
        tok = f"{_LEFT}{_LIST_REJECT_BEFORE}{re.escape(str(t))}{_RIGHT}{_LIST_REJECT_AFTER}"
        alts.append(f"{_LIST_BEFORE}{tok}")
        alts.append(f"{tok}{_LIST_AFTER}")
    for p in spec.get("pattern", []):
        alts.append(f"(?:{p})")
    if not alts:
        raise ValueError(f"skill {slug!r} has no matchers")
    return re.compile("|".join(alts))


def load_dictionary(path: Path = DICTIONARY_PATH) -> SkillDictionary:
    raw = yaml.safe_load(path.read_text())
    skills = tuple(
        Skill(
            slug=slug,
            category=spec["cat"],
            regex=_compile(slug, spec),
            suppress_for=frozenset(c.lower() for c in spec.get("suppress_for", [])),
        )
        for slug, spec in raw["skills"].items()
    )
    return SkillDictionary(version=raw["version"], skills=skills)


@lru_cache(maxsize=1)
def _default_dictionary() -> SkillDictionary:
    return load_dictionary()


def posting_text(
    title: str | None,
    description: str | None,
    boilerplate: set[str] | None = None,
) -> str:
    # Title first, separated so a title's last word can't fuse with the body.
    return f"{clean_text(title)} . {strip_boilerplate(description, boilerplate or set())}"


def extract_skills(
    text: str,
    company: str | None = None,
    dictionary: SkillDictionary | None = None,
) -> set[str]:
    d = dictionary or _default_dictionary()
    employer = (company or "").strip().lower()
    return {
        s.slug
        for s in d.skills
        if employer not in s.suppress_for and s.regex.search(text)
    }


# --- DB replay --------------------------------------------------------------


def _fetch_postings(conn) -> list[tuple[str, str, str, str]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT posting_key, company, title_raw, description_raw FROM postings"
        )
        return cur.fetchall()


def _replace_posting_skills(conn, rows: list[tuple[str, str]]) -> None:
    with conn.transaction(), conn.cursor() as cur:
        cur.execute("DELETE FROM posting_skills")
        with cur.copy("COPY posting_skills (posting_key, skill) FROM STDIN") as copy:
            for row in rows:
                copy.write_row(row)


def _report(results: dict[str, set[str]], titles: dict[str, str], version: int) -> None:
    n = len(results)
    with_any = sum(1 for s in results.values() if s)
    total = sum(len(s) for s in results.values())
    print(f"=== skills (dictionary v{version}) ===")
    print(f"  postings={n}  with>=1 skill={with_any} ({with_any / max(n, 1):.0%})"
          f"  mean skills/posting={total / max(n, 1):.1f}")

    counts = Counter(s for skills in results.values() for s in skills)
    print("\n=== top 40 skills (postings containing) ===")
    for slug, c in counts.most_common(40):
        print(f"  {c:>5}  {c / n:>4.0%}  {slug}")

    empty = [titles[k] for k, s in results.items() if not s]
    print(f"\n=== sample titles with zero skills ({len(empty)}) ===")
    for t in sorted(empty)[:25]:
        print(f"  {t}")


def _boilerplate_for(postings) -> dict[str, set[str]]:
    return find_boilerplate((company, desc) for _k, company, _t, desc in postings)


def _print_boilerplate(postings, boilerplate: dict[str, set[str]]) -> None:
    sizes = Counter((company or "").lower() for _k, company, _t, _d in postings)
    for company, keys in sorted(boilerplate.items(), key=lambda kv: -len(kv[1])):
        print(f"=== {company} ({sizes[company]} postings, {len(keys)} boilerplate segments) ===")
        for k in sorted(keys):
            print(f"  {k[:150]}")


def _explain(postings, dictionary: SkillDictionary, slug: str, limit: int) -> None:
    skill = next((s for s in dictionary.skills if s.slug == slug), None)
    if skill is None:
        raise SystemExit(f"unknown skill {slug!r}")
    boilerplate = _boilerplate_for(postings)
    shown = 0
    for _key, company, title, desc in postings:
        employer = (company or "").lower()
        if employer in skill.suppress_for:
            continue
        text = posting_text(title, desc, boilerplate.get(employer))
        m = skill.regex.search(text)
        if not m:
            continue
        lo, hi = max(m.start() - 60, 0), min(m.end() + 60, len(text))
        print(f"[{company} | {title}]")
        print(f"  ...{text[lo:m.start()]}>>{m.group()}<<{text[m.end():hi]}...")
        shown += 1
        if shown >= limit:
            break


def replay(conn, dry_run: bool = False, postings=None, dictionary=None) -> None:
    """Recompute skills for every posting and replace posting_skills."""
    dictionary = dictionary or load_dictionary()
    postings = postings if postings is not None else _fetch_postings(conn)
    boilerplate = _boilerplate_for(postings)
    results = {
        key: extract_skills(
            posting_text(title, desc, boilerplate.get((company or "").lower())),
            company,
            dictionary,
        )
        for key, company, title, desc in postings
    }
    titles = {key: title for key, _c, title, _d in postings}
    _report(results, titles, dictionary.version)

    if dry_run:
        print("\n(dry run — posting_skills unchanged)")
        return
    rows = [(key, slug) for key, skills in results.items() for slug in sorted(skills)]
    _replace_posting_skills(conn, rows)
    print(f"\nwrote {len(rows)} rows to posting_skills")


def main() -> int:
    import psycopg
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="compute and report only")
    parser.add_argument("--explain", metavar="SKILL", help="print match contexts for a skill")
    parser.add_argument("--limit", type=int, default=20, help="contexts to show with --explain")
    parser.add_argument("--boilerplate", action="store_true",
                        help="print the segments treated as boilerplate per company")
    args = parser.parse_args()

    load_dotenv()
    dictionary = load_dictionary()
    with psycopg.connect(os.environ["SUPABASE_DB_URL"]) as conn:
        postings = _fetch_postings(conn)

        if args.explain:
            _explain(postings, dictionary, args.explain, args.limit)
            return 0

        if args.boilerplate:
            _print_boilerplate(postings, _boilerplate_for(postings))
            return 0
        replay(conn, dry_run=args.dry_run, postings=postings, dictionary=dictionary)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

# The Indian Tech Job Market Panel

A longitudinal dataset of Indian tech job postings, observed daily for 12 weeks, and the analysis program built on top of it.

**Status:** collection in progress — findings publish week 12.

## What this is

Most job-market analyses use a single-day snapshot: what's open right now. This project builds something structurally different — the same postings observed day after day — so we can ask questions a snapshot can't answer:

- How long does a SQL-heavy role stay open compared to a Python-heavy one?
- Has the skill mix shifted between August and November in ways distinguishable from noise?
- Which cities specialise in which kinds of work?
- Who discloses salary, and how does that bias headline pay estimates?

Nobody has this dataset for the Indian market. That's the whole point.

## Findings

_Will be published as a static report on completion. Link goes here._

## Data & methodology

Full details in [`DATA_DICTIONARY.md`](./DATA_DICTIONARY.md) (published week 2).

**Scope:** Full-time technical individual-contributor roles (software engineering, data, analytics, ML) posted by companies hiring in India.

**Sources:** Greenhouse, Lever, and Ashby applicant tracking system APIs. Public, documented, unauthenticated JSON. ~300 company tokens covering India-HQ tech companies (Razorpay, Zerodha, Postman, Swiggy, Freshworks, ...) and India-hiring globals (Stripe, Atlassian, GitLab, ...).

**Collection window:** 12 weeks starting September 2026.

**Cadence:** Daily at 08:00 IST via GitHub Actions.

### Collection discipline

- We respect `robots.txt` for any HTML sources.
- Rate limit: ~1 request per 2.5 seconds per host.
- User-Agent identifies the project and includes a contact email.
- Content-hashed dedup: postings whose content is unchanged are not re-fetched.
- Only publicly posted job information is stored. No personal data.

### The identity problem

Longitudinal analysis depends on recognising that today's posting is the same one we saw yesterday. We use a two-tier key: `source_id` where the ATS gives us a stable one; otherwise a SHA256 of `(normalised company, title, location, description[:500])`. Aggressive normalisation (lowercase, punctuation stripped, whitespace collapsed) prevents phantom postings from a single re-rendered space. The key's stability is validated by inspecting the distribution of observed lifetimes — a large share of one-day postings would mean the key is unstable and needs fixing.

### The closure problem

A posting disappearing from the feed can mean it was filled, withdrawn, expired, or hidden by a scraper failure. We can't distinguish these perfectly, but we do two things:

1. **Closure guard.** If a run fetches fewer rows than 40% of the trailing 7-day median for that source, we skip the closure step entirely and log an error. Otherwise a single network hiccup would mark the whole universe as closed.
2. **Closure timing check.** During analysis we check whether closures cluster at suspicious intervals (like exactly 30 days) that would indicate platform expiry rather than filling, and treat those separately.

## Architecture

Four moving parts, deliberately:

- **Collector** (`collectors/`) — fetches raw payloads, knows nothing about skills or salaries.
- **Extractor** (`extraction/`) — separately re-runnable over full history so improvements to parsing logic backfill week-1 data.
- **Store** — Postgres (Supabase free tier). Raw JSONB payloads are stored gzipped in `payload_raw`; parsed fields sit alongside.
- **Analysis** (`analysis/`) — Jupyter notebooks + written report.

The rule that governs everything: **every posting, every day, keeps its original payload**. In week 9 we'll want a field we didn't think to parse in week 1; if we kept the raw we can backfill the whole history in an afternoon.

## Running it

Requirements: Python 3.11+, a Postgres database (Supabase free tier is enough).

```bash
# Install
pip install -e .

# Configure
cp .env.example .env
# Fill in SUPABASE_DB_URL and COLLECTOR_CONTACT_EMAIL

# Apply schema
psql "$SUPABASE_DB_URL" -f db/schema.sql

# One-off test run
python run_collectors.py

# Data-quality checks
python -m quality.assertions
python -m quality.monitor

# Probe token list (find stale/empty tokens)
python -m tools.probe_tokens

# Skill extraction: replays extraction/skills.yaml over every posting
python -m extraction.skills --dry-run      # report only
python -m extraction.skills                # rewrite posting_skills
python -m extraction.skills --explain go   # audit match contexts for one skill

# Tests
pip install -e '.[dev]' && pytest
```

For production, `.github/workflows/collect.yml` runs the above daily at 08:00 IST. Repo secrets required: `SUPABASE_DB_URL`, `COLLECTOR_CONTACT_EMAIL`.

## Repository layout

```
collectors/     # ATS collectors (base + greenhouse/lever/ashby)
extraction/     # skills.yaml + parsers for titles, salary, geography
db/             # schema.sql + upsert loop
quality/        # assertions + volume/parse-rate monitor
tools/          # token probing, one-off utilities
analysis/       # Jupyter notebooks (A1 clustering, A2 survival, ...)
report/         # findings.md + figures
data/           # tokens.yaml (public) + tokens.private.yaml (gitignored)
```

## Data licence

_Full licence published with the dataset release in week 12._ The intent is CC-BY-4.0 for the aggregated dataset and dictionary; the raw scraped payloads are covered by the ATS providers' own terms and are not part of the release.

## Contact

Rohit Raj — rohitraj251204@gmail.com

If you're rate-limiting this scraper or have a data-provenance concern, please email; I'd rather hear about it than get blocked.

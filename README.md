# The Indian Tech Job Market Panel

A longitudinal dataset of India-located tech job postings, observed daily from employers' own job boards, and the analysis built on top of it.

**Status:** collection day 42 of 84. Findings below are a **draft** and will be re-run on the final panel (2026-11-29); the skill-drift analysis runs on 2026-12-01 as [pre-registered](report/A3_preregistration.md).

## Findings (draft)

1. **Who is hiring explains a posting's skill requirements about as well as its title does** (Cramér's V 0.46 for employer vs 0.49 for title family, across 213 technical postings).
2. **"Software engineer" hides an infrastructure role:** the most stable cross-employer skill cluster is split 22 / 20 between postings titled software engineer and SRE/infra.
3. **Postings stay open a median of 43 days** (95% CI 33–50) from publication, against 17 days by the naive average.
4. **Role family doesn't detectably predict time-to-close;** most apparent predictors (remote, NCR, principal) are employer composition.
5. **Pay transparency is near zero:** 3.8% of postings show an India pay range, all from one employer.

→ **[Read the report](report/findings.md)**

## What this is

Most job-market analyses use a single-day snapshot: what's open right now. This project builds something structurally different — the same postings observed day after day — so it can ask questions a snapshot can't: how long postings stay open and what predicts it, whether one job title hides several roles, whether the skill mix shifts, and who discloses pay.

## Data & methodology

Full details in [`DATA_DICTIONARY.md`](./DATA_DICTIONARY.md): every table and field, null semantics, vocabularies, known issues and coverage.

**Scope:** India-located postings; the analysis population is full-time technical individual-contributor roles (software, infrastructure, data, ML, QA, security).

**Sources:** Greenhouse and Ashby applicant tracking system APIs — public, documented, unauthenticated JSON. 13 employers have India postings in the panel, mostly global tech companies' India offices (Okta, MongoDB, Stripe, GitLab, Twilio, Elastic, Airbnb, OpenAI, ...). Lever is supported by the collector but no Lever board yielded India postings, and no working board was found for Indian-headquartered companies, so **this is a panel of global employers hiring in India, not of the Indian tech market as a whole** (see the report's limitations).

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

# Title normalisation: rules baseline vs Gemini, scored on hand labels
python -m extraction.titles --guide                 # labelling guide / taxonomy
python -m extraction.titles --label-llm             # fill Gemini cache (needs GEMINI_API_KEY)
python -m extraction.titles --evaluate              # score both vs data/title_labels.csv
python -m extraction.titles --method rules          # write title_family/seniority

# Geography: canonical city (NCR collapsed), remote/hybrid with signal source
python -m extraction.geography --dry-run

# All extractors in order (what the daily cron runs after collection)
python -m extraction

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
analysis/       # one script per analysis (a1_clustering.py, ...) + shared loader
report/         # findings.md + figures
data/           # tokens.yaml (public) + tokens.private.yaml (gitignored)
```

## Data licence

_Full licence published with the dataset release in week 12._ The intent is CC-BY-4.0 for the aggregated dataset and dictionary; the raw scraped payloads are covered by the ATS providers' own terms and are not part of the release.

## Contact

Rohit Raj — rohitraj251204@gmail.com

If you're rate-limiting this scraper or have a data-provenance concern, please email; I'd rather hear about it than get blocked.

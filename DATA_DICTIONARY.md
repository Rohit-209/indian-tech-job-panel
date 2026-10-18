# Data dictionary — Indian Tech Job Panel

A daily panel of job postings for roles based in India, collected from employers' own applicant
tracking systems (Greenhouse, Ashby) from 2026-09-07. This document defines every table and field,
how each derived field is produced, and what NULL means in each case. Any rule stated here as a
guarantee is enforced by `quality/assertions.py`, which runs after every daily collection.

**Dates are IST calendar days** (`Asia/Kolkata`) throughout, so runs on different machines agree on
what "today" is.

## Coverage

<!-- COVERAGE:START -->
*Generated 2026-10-18 by `python -m tools.coverage --write`. Do not edit by hand.*

| | |
|---|---|
| Panel window | 2026-09-07 → 2026-10-18 (42 collection days) |
| Postings ever seen | 586 (330 active, 256 closed) |
| Observations (posting × day) | 13530 |
| By source | greenhouse 561 (96%), ashby 25 (4%) |
| In scope (technical IC) | 270 (46%) |
| Employers | 13: Okta 190 (32%), Mongodb 119 (20%), Stripe 67 (11%), Gitlab 65 (11%), Twilio 37 (6%), Elastic 32 (5%), Airbnb 29 (5%), Openai 15 (3%), Datadog 14 (2%), Cursor 8 (1%), Postman 6 (1%), Cloudflare 2 (0%), Deepgram 2 (0%) |

| Field | Coverage / distribution |
|---|---|
| `title_family` | software_engineering 218 (37%), non_tech 194 (33%), solutions_support 59 (10%), sre_infra 41 (7%), product_design 24 (4%), data_analytics 20 (3%), ml_engineering 10 (2%), qa_testing 10 (2%), data_engineering 5 (1%), security_engineering 4 (1%), data_science 1 (0%) |
| `seniority` | senior 166 (28%), unspecified 142 (24%), staff 130 (22%), manager 59 (10%), junior 30 (5%), principal 20 (3%), director 20 (3%), mid 15 (3%), intern 4 (1%) |
| `city` | Bengaluru 394 (67%), NCR 114 (19%), NULL 63 (11%), Mumbai 14 (2%), Hyderabad 1 (0%); multi-city 23, multi-country 7 |
| workplace (`is_remote`/`is_hybrid`) | NULL 199 (34%), hybrid 191 (33%), remote 131 (22%), onsite 65 (11%) |
| `workplace_source` | NULL 199 (34%), ats_field 163 (28%), li_tag 116 (20%), location 50 (9%), company_policy 36 (6%), phrase 22 (4%) |
| skills | 503 postings with ≥ 1 skill (86%); 4650 posting–skill rows; 155 of 178 dictionary skills observed (dictionary v1) |
| salary | 24 postings (4.1%): INR 22 (4%), USD 2 (0%) |
<!-- COVERAGE:END -->

## How the data is made

```
ATS APIs ──► collectors/ ──► postings (raw) + observations ──► extraction/ ──► postings (derived) + posting_skills
              daily 08:00 IST          one row per posting per day      replayed in full every run
```

- **Collection** (`run_collectors.py`): fetches every board in `data/tokens.yaml`, keeps postings
  whose location passes the India filter (`collectors/base.py::is_india_location`), upserts them,
  and writes one observation per posting per day.
- **Extraction** (`python -m extraction`): recomputes every derived field for every posting from
  the stored raw data. Nothing derived is ever edited by hand; a parsing fix is applied to the whole
  history by re-running. The raw payload is stored so fields nobody thought to parse can still be
  recovered later.

## Tables

### `postings` — one row per distinct posting

**Identity and lifecycle (collector)**

| Column | Type | Meaning |
|---|---|---|
| `posting_key` | text, PK | `{source}:{source_id}` when the ATS gives a stable ID (every posting so far), else `{source}:hash:{sha256[:24]}` of normalised company, title, location and description[:500]. |
| `source` | text | `greenhouse` or `ashby`. Lever is supported but no Lever board has yielded India postings. |
| `source_id` | text | The ATS's own job ID. |
| `company` | text | Employer, derived from the board token (e.g. `Mongodb`, `Openai`). Case as stored; compare case-insensitively. |
| `first_seen` / `last_seen` | date | First and most recent IST day the posting appeared in a fetch. |
| `is_active` | bool | Present in the most recent fetch of its source. |
| `closed_on` | date, NULL while active | First day the posting was missing. **If a closed posting reappears it is reopened**: `is_active` → true and `closed_on` → NULL. Temporary disappearances are visible only in `observations`. Guaranteed: NULL ⇔ active; never before `last_seen`. |
| `content_hash` | text | Hash of the posting's content as last fetched; compare with `observations.content_hash` to detect edits. |

**Raw content (collector, never modified by extraction)**

| Column | Type | Meaning |
|---|---|---|
| `title_raw`, `location_raw`, `url` | text | As published. `location_raw` for Ashby falls back to secondary locations when the primary is empty. |
| `description_raw` | text | Description as fetched. **For Greenhouse this still contains HTML tags**: the API returns escaped HTML, and the collector strips tags before unescaping. Deliberately not fixed mid-panel, because changing it would change every `content_hash` and fake a wave of edits. Use `extraction.text.clean_text()`. |
| `payload_raw` | bytea | gzip-compressed JSON of the ATS's original record. Holds fields not parsed into columns, notably the **publish date** (Greenhouse `first_published`, Ashby `publishedAt`; see `analysis.common.load_lifecycle()`), Ashby `workplaceType`, and Greenhouse custom `metadata` fields. |

**Derived (extraction, recomputed for every posting on every run)**

| Column | Type | Produced by | Meaning, and what NULL means |
|---|---|---|---|
| `title_family` | text | `extraction/titles.py` | The role's **function**, one of the 11 families below. Never NULL after extraction. |
| `seniority` | text | `extraction/titles.py` | The role's **level**, one of the 9 values below; independent of family. `unspecified` (not NULL) when the title has no level marker. |
| `city` | text | `extraction/geography.py` | Canonical city of the **first** India location listed. Delhi, Gurugram, Noida, Ghaziabad and Faridabad are collapsed to `NCR`. **NULL = the posting names no Indian city** (e.g. "India", "Remote – India"); not an extraction failure. |
| `cities` | text[] | geography | Every canonical India city listed, in order, deduplicated. Empty array when `city` is NULL. Guaranteed: `city = cities[1]`. |
| `is_multi_country` | bool | geography | The location also lists non-India places ("Bangalore, India; Remote, Canada"): a global role India is eligible for. |
| `is_remote`, `is_hybrid` | bool | geography | Workplace model: remote = (true, false), hybrid = (false, true), onsite = (false, false). **NULL in both = unknown.** The absence of a remote signal is *not* recorded as onsite. Guaranteed: never both true; NULL together with `workplace_source`. |
| `workplace_source` | text | geography | Which signal decided the workplace model, in precedence order: `ats_field` (Ashby `workplaceType`; Greenhouse custom fields "Workplace Type", "Working Model Eligibility") › `li_tag` (`#LI-Hybrid` etc. in the description) › `location` ("Remote" in an India location segment) › `phrase` (an explicit statement such as "hybrid model of 3 days in office") › `company_policy` (GitLab only: all-remote, evidenced by every GitLab posting that states a model). NULL when unknown. |
| `salary_min`, `salary_max` | numeric | `extraction/salary.py` | Disclosed pay range, **annualised**. NULL = not disclosed. Placeholder values in ATS custom fields (Okta/MongoDB `0.0`, `USD 0.01`) are treated as not disclosed, never as zero. `salary_min` is NULL for "up to X". Guaranteed: min ≤ max, non-negative. |
| `salary_currency` | text | salary | `INR` or `USD`. A USD range on an India posting (GitLab) is a US range, not India pay. |
| `salary_period` | text | salary | Always `annual` when a salary is present (monthly figures are converted). |
| `salary_from_monthly` | bool | salary | The range was published monthly and multiplied by 12. |
| `extracted_at` | timestamptz | `python -m extraction` | When the derived fields were last recomputed. |
| `is_internship` | bool | — | **Not populated.** Internships were dropped from scope in Phase 0 (the ATS boards used don't list them reliably); use `seniority = 'intern'`. |

### `observations` — one row per posting per day it was seen

| Column | Type | Meaning |
|---|---|---|
| `posting_key` | text, FK | |
| `observed_on` | date | IST day. PK is (`posting_key`, `observed_on`). |
| `content_hash` | text | Content hash that day. |

This is the panel. A posting missing for some days and then present again has a gap here, while
its `postings` row shows it as continuously active.

### `posting_skills` — posting × skill

| Column | Type | Meaning |
|---|---|---|
| `posting_key` | text, FK | |
| `skill` | text | Canonical skill slug from `extraction/skills.yaml` (178 skills, 15 categories). PK is (`posting_key`, `skill`). |

Produced by `extraction/skills.py`: dictionary regexes over title + description, after removing
**company boilerplate**. A segment shared by ≥ 40% of one employer's postings (for employers with
≥ 4 postings) is dropped, so legal notices and company intros don't count as requirements. A
skill is also suppressed for the employer that makes it (no `mongodb` on MongoDB postings). The
table is replaced in full on every run, so it always reflects exactly one dictionary version.

### `title_llm_cache` — Gemini title classifications

| Column | Type | Meaning |
|---|---|---|
| `title_key` | text | `lower(company) \| lower(title)`, whitespace collapsed. |
| `model`, `prompt_version` | text, int | PK with `title_key`; a prompt or model change re-classifies without overwriting. |
| `title_family`, `seniority` | text | Gemini's answer, constrained to the vocabularies below. |
| `company`, `title_raw`, `created_at` | | |

Used only if `TITLE_METHOD=llm`. Until the hand-label evaluation (`python -m extraction.titles
--evaluate`) shows Gemini beats the rules, `postings` uses the rules classifier.

### `run_log` — one row per collector run per source

`started_at`, `finished_at`, `source`, `rows_fetched`, `rows_new`, `rows_updated`, `rows_closed`,
`errors`, `status` (`ok` | `skipped_closure` | `failed`), `notes`. `skipped_closure` means the
closure guard tripped: the fetch returned < 40% of the 7-day trailing median, so no postings were
closed that run.

## Vocabularies

**`title_family`**: the role's function.

| Value | Covers |
|---|---|
| `software_engineering` | Backend, frontend, full-stack, mobile, embedded, architects, business-systems developers |
| `sre_infra` | SRE, DevOps, platform, cloud, infrastructure, network, production/release engineering |
| `data_engineering` | Data pipelines and platforms, analytics engineering |
| `data_science` | Data, applied and research scientists; statisticians |
| `ml_engineering` | ML / AI engineers who ship models or LLM systems; ML platform; research engineers |
| `data_analytics` | Data, product, business, BI and people analysts; analytics leads |
| `qa_testing` | QA, test automation, SDET |
| `security_engineering` | Security engineers, AppSec, detection & response, security analysts |
| `solutions_support` | Solutions / sales / forward-deployed engineers and architects, technical support, TAMs |
| `product_design` | Product and program managers, designers, UX researchers, technical writers |
| `non_tech` | Sales, marketing, finance, operations, recruiting, HR, legal, admin |

**`seniority`**: the role's level. `intern`, `junior` (junior / associate / I / L1), `mid` (II /
L2 / "intermediate"), `senior` (senior / III / L3), `staff` (staff / lead / architect / IV / L4),
`principal` (principal / distinguished / V+ / L5+), `manager` (people managers only, so product,
program and account managers are ICs), `director` (director / head of / VP), `unspecified`.

**In scope.** The project's analysis population, `extraction.titles.in_scope()`, is
`title_family` ∈ {software_engineering, sre_infra, data_engineering, data_science,
ml_engineering, data_analytics, qa_testing, security_engineering} and `seniority` ∉ {manager,
director, intern}: full-time technical individual-contributor roles.

## Known issues and how analyses handle them

| Issue | Effect | Handling |
|---|---|---|
| **Employer concentration**: Okta is about a third of postings, half of in-scope | Raw prevalences describe Okta | Analyses report within-company or company-balanced estimates alongside raw ones (A1 company-capped stability, A2 company strata, A3 Mantel–Haenszel) |
| **Duplicates**: 13% of postings are near-copies (one JD at several levels, one role per city, reposts) | Rows aren't independent | A7 quantifies; A1 merges same-company identical skill sets; A2 censors reposts in sensitivity |
| **Reopened postings**: 47 postings vanished and returned under the same ID | `postings` shows them continuously active | Use `observations` for exact presence; A2 treats the spell as continuing (first gap = event in sensitivity) |
| **Board disappearance**: Postman left Greenhouse on 2026-09-17; its 6 postings vanished at once | Looks like 6 closures | A2 detects company-wide vanishing and censors it rather than counting it as closure |
| **Panel day 1 had two runs** | 5 postings have `closed_on` = `first_seen` = 2026-09-07 | A closure is taken as `max(closed_on, last_seen + 1)` |
| **Greenhouse HTML in `description_raw`** | Raw text contains tags | `extraction.text.clean_text()` |
| **Skill mention ≠ requirement** | "Partner with SRE teams" counts as `sre` | Documented limitation |
| **Title classification is provisional** | Families from rules, pending hand-label evaluation | Rules and Gemini agree on 98% of in-scope decisions |
| **Publish date can predate relisting** | A small share of postings appear long after their publish date | A2 `born` specification restricts to postings first seen within 1 day of publishing |
| **Salary is almost never disclosed** | 22 postings with India pay, all one employer | See `report/A4_disclosure.md` |

## Releasing the dataset

The derived fields, skills, observations and URLs are the dataset's own contribution. **The raw
descriptions and payloads are the employers' text.** They're public, but republishing them in bulk
is a different act from linking to them. The suggested release is the derived tables plus `url`,
with `description_raw` / `payload_raw` withheld (anyone can re-fetch live postings from the URLs).
This is a decision for the dataset owner, not a technical constraint.

## Regenerating

```bash
python -m extraction          # recompute every derived field
python -m quality.assertions  # check every guarantee above
python -m tools.coverage --write
```

-- Indian Tech Job Panel — schema
-- One-shot migration; safe to re-run (uses IF NOT EXISTS everywhere).
-- Apply with: psql "$SUPABASE_DB_URL" -f db/schema.sql

-- postings: one row per distinct posting (identity determined by posting_key).
-- payload_raw is gzipped JSON to fit within Supabase free tier storage.
-- The rule that governs the whole build: keep the raw so we can backfill
-- any field we didn't parse in week one.
CREATE TABLE IF NOT EXISTS postings (
    posting_key       TEXT PRIMARY KEY,
    source            TEXT NOT NULL,
    source_id         TEXT,
    company           TEXT,
    title_raw         TEXT,
    location_raw      TEXT,
    url               TEXT,
    description_raw   TEXT,
    payload_raw       BYTEA,           -- gzip(JSON of source's original record)
    content_hash      TEXT,
    first_seen        DATE NOT NULL,
    last_seen         DATE NOT NULL,
    is_active         BOOLEAN NOT NULL DEFAULT TRUE,
    closed_on         DATE,

    -- Extracted fields, filled by the extractor (nullable until it runs).
    title_family      TEXT,
    seniority         TEXT,
    city              TEXT,
    is_remote         BOOLEAN,
    is_hybrid         BOOLEAN,
    is_internship     BOOLEAN,
    salary_min        NUMERIC,
    salary_max        NUMERIC,
    salary_period     TEXT,             -- 'annual' | 'monthly' | 'hourly'
    salary_currency   TEXT,             -- ISO 4217; we normalise to 'INR'
    salary_from_monthly BOOLEAN,        -- flag: was value derived from monthly figure?
    extracted_at      TIMESTAMPTZ,

    -- Structural sanity — cheap DB-level guards.
    CONSTRAINT postings_dates_ordered CHECK (first_seen <= last_seen),
    CONSTRAINT postings_closed_only_if_inactive CHECK (
        (is_active = TRUE  AND closed_on IS NULL) OR
        (is_active = FALSE AND closed_on IS NOT NULL)
    ),
    CONSTRAINT postings_salary_nonneg CHECK (
        (salary_min IS NULL OR salary_min >= 0) AND
        (salary_max IS NULL OR salary_max >= 0)
    ),
    CONSTRAINT postings_salary_ordered CHECK (
        salary_min IS NULL OR salary_max IS NULL OR salary_min <= salary_max
    )
);

-- Added in Phase 2 (extraction/geography.py). ALTER keeps re-runs safe on
-- the live table.
ALTER TABLE postings ADD COLUMN IF NOT EXISTS cities TEXT[];
ALTER TABLE postings ADD COLUMN IF NOT EXISTS is_multi_country BOOLEAN;
ALTER TABLE postings ADD COLUMN IF NOT EXISTS workplace_source TEXT;  -- ats_field | li_tag | location | phrase | company_policy

CREATE INDEX IF NOT EXISTS postings_source_first_seen_idx
    ON postings (source, first_seen);
CREATE INDEX IF NOT EXISTS postings_is_active_idx
    ON postings (is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS postings_title_family_idx
    ON postings (title_family) WHERE title_family IS NOT NULL;
CREATE INDEX IF NOT EXISTS postings_city_idx
    ON postings (city) WHERE city IS NOT NULL;
CREATE INDEX IF NOT EXISTS postings_needs_extraction_idx
    ON postings (posting_key) WHERE extracted_at IS NULL;

-- posting_skills: many-to-many between postings and canonical skills.
-- Populated by the extractor from description_raw against extraction/skills.yaml.
CREATE TABLE IF NOT EXISTS posting_skills (
    posting_key TEXT NOT NULL REFERENCES postings(posting_key) ON DELETE CASCADE,
    skill       TEXT NOT NULL,
    PRIMARY KEY (posting_key, skill)
);

CREATE INDEX IF NOT EXISTS posting_skills_skill_idx ON posting_skills (skill);

-- title_llm_cache: Gemini's (title_family, seniority) per distinct
-- (company, title), filled by extraction/titles_llm.py. Keyed by model and
-- prompt_version so a prompt change re-classifies without losing old rows.
CREATE TABLE IF NOT EXISTS title_llm_cache (
    title_key      TEXT NOT NULL,        -- lower(company) || '|' || lower(title)
    model          TEXT NOT NULL,
    prompt_version INT NOT NULL,
    company        TEXT,
    title_raw      TEXT,
    title_family   TEXT NOT NULL,
    seniority      TEXT NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (title_key, model, prompt_version)
);

-- observations: one row per (posting, day-we-saw-it). This is the panel.
-- Also stores content_hash so we can detect edits by comparing across days.
CREATE TABLE IF NOT EXISTS observations (
    posting_key   TEXT NOT NULL REFERENCES postings(posting_key) ON DELETE CASCADE,
    observed_on   DATE NOT NULL,
    content_hash  TEXT,
    PRIMARY KEY (posting_key, observed_on)
);

CREATE INDEX IF NOT EXISTS observations_observed_on_idx ON observations (observed_on);

-- run_log: one row per collector invocation per source.
-- Non-optional: this is how we notice a source silently degrading.
CREATE TABLE IF NOT EXISTS run_log (
    run_id        BIGSERIAL PRIMARY KEY,
    started_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at   TIMESTAMPTZ,
    source        TEXT NOT NULL,
    rows_fetched  INT NOT NULL DEFAULT 0,
    rows_new      INT NOT NULL DEFAULT 0,
    rows_updated  INT NOT NULL DEFAULT 0,
    rows_closed   INT NOT NULL DEFAULT 0,
    errors        INT NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'ok', -- 'ok' | 'skipped_closure' | 'failed'
    notes         TEXT
);

CREATE INDEX IF NOT EXISTS run_log_source_started_idx
    ON run_log (source, started_at DESC);

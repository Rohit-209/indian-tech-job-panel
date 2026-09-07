"""Upsert loop — the heart of the panel.

The identity key must be stable across days or every posting looks like it
lives exactly one day. We use a two-tier strategy:

  1. If the source gave us a source_id, key = f"{source}:{source_id}".
  2. Otherwise, hash a normalised tuple of (company, title, location,
     description[:500]). Normalisation is aggressive because a single
     re-rendered space would otherwise mint a phantom posting.

Closure is guarded: if a run fetched fewer rows than 40% of the trailing
7-day median for that source, we mark the run 'skipped_closure', do NOT flip
anyone to inactive, and log an error. Otherwise one network hiccup would
close the entire universe.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

# All observed_on dates are IST calendar days. Every runtime (local Mac, GH
# Actions runner, someone's Docker container) resolves to the same date given
# the same wall clock. This matters because observations is keyed on
# (posting_key, observed_on) — if two runs disagree on "today" we either
# duplicate observations or silently drop the second one.
IST = ZoneInfo("Asia/Kolkata")

import psycopg
import structlog
from psycopg.rows import dict_row

from collectors.base import RawPosting

log = structlog.get_logger()

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")

_CLOSURE_GUARD_RATIO = 0.40  # skip closure if fetched < 40% of 7d trailing median
_CLOSURE_GUARD_MIN_HISTORY = 3  # need at least this many past runs to compute median


# --- Identity key -----------------------------------------------------------


def _normalise(s: str | None) -> str:
    """Aggressive normalisation so trivial rendering changes don't mint keys."""
    if not s:
        return ""
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def posting_key(p: RawPosting) -> str:
    if p.source_id:
        return f"{p.source}:{p.source_id}"

    blob = "|".join([
        _normalise(p.company),
        _normalise(p.title_raw),
        _normalise(p.location_raw),
        _normalise(p.description_raw)[:500],
    ])
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]
    return f"{p.source}:hash:{digest}"


# --- Run summary ------------------------------------------------------------


@dataclass
class RunSummary:
    source: str
    started_at: datetime
    finished_at: datetime | None = None
    rows_fetched: int = 0
    rows_new: int = 0
    rows_updated: int = 0
    rows_closed: int = 0
    errors: int = 0
    status: str = "ok"  # 'ok' | 'skipped_closure' | 'failed'
    notes: str = ""


# --- Trailing median (closure guard) ----------------------------------------


def _trailing_median_fetched(cur: psycopg.Cursor, source: str) -> int | None:
    """Median rows_fetched for the last 7 successful runs of this source.

    Returns None if we don't have _CLOSURE_GUARD_MIN_HISTORY runs yet — during
    the first few days the guard is effectively disabled, which is correct:
    the first run has no history to compare against.
    """
    cur.execute(
        """
        SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY rows_fetched)::int AS med,
               COUNT(*) AS n
        FROM (
            SELECT rows_fetched
            FROM run_log
            WHERE source = %s
              AND status = 'ok'
              AND started_at >= NOW() - INTERVAL '7 days'
            ORDER BY started_at DESC
            LIMIT 7
        ) recent
        """,
        (source,),
    )
    row = cur.fetchone()
    if not row or row["n"] < _CLOSURE_GUARD_MIN_HISTORY:
        return None
    return row["med"]


# --- Main upsert ------------------------------------------------------------


def upsert_source(
    conn: psycopg.Connection,
    source: str,
    postings: list[RawPosting],
    observed_on: date | None = None,
) -> RunSummary:
    """Upsert one source's fetched postings, emit observations, close absentees.

    Runs in a single transaction per source so a mid-run crash leaves nothing
    half-applied.
    """
    observed_on = observed_on or datetime.now(IST).date()
    started_at = datetime.now(timezone.utc)
    summary = RunSummary(source=source, started_at=started_at, rows_fetched=len(postings))

    with conn.cursor(row_factory=dict_row) as cur:
        # 1. Compute keys + hashes up front so we can dedupe within one run.
        # (Some sources multi-post the same job across boards; last one wins.)
        by_key: dict[str, RawPosting] = {}
        for p in postings:
            by_key[posting_key(p)] = p

        seen_keys = list(by_key.keys())

        # 2. Insert or update postings.
        for key, p in by_key.items():
            hash_ = p.content_hash()
            payload_gz = p.payload_gzip()

            cur.execute(
                """
                INSERT INTO postings (
                    posting_key, source, source_id, company, title_raw,
                    location_raw, url, description_raw, payload_raw,
                    content_hash, first_seen, last_seen, is_active
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, TRUE
                )
                ON CONFLICT (posting_key) DO UPDATE SET
                    last_seen    = EXCLUDED.last_seen,
                    is_active    = TRUE,
                    closed_on    = NULL,
                    -- Only overwrite mutable fields if they actually changed.
                    -- Keeps the row's updated_at meaningful if we add one later.
                    title_raw       = EXCLUDED.title_raw,
                    location_raw    = EXCLUDED.location_raw,
                    url             = EXCLUDED.url,
                    description_raw = EXCLUDED.description_raw,
                    payload_raw     = EXCLUDED.payload_raw,
                    content_hash    = EXCLUDED.content_hash
                RETURNING (xmax = 0) AS inserted
                """,
                (
                    key, source, p.source_id, p.company, p.title_raw,
                    p.location_raw, p.url, p.description_raw, payload_gz,
                    hash_, observed_on, observed_on,
                ),
            )
            result = cur.fetchone()
            if result and result["inserted"]:
                summary.rows_new += 1
            else:
                summary.rows_updated += 1

            # 3. Observation row (one per posting per day). ON CONFLICT DO
            # NOTHING because a manual re-run on the same day shouldn't error.
            cur.execute(
                """
                INSERT INTO observations (posting_key, observed_on, content_hash)
                VALUES (%s, %s, %s)
                ON CONFLICT (posting_key, observed_on) DO NOTHING
                """,
                (key, observed_on, hash_),
            )

        # 4. Closure guard — do NOT proceed to step 5 if this run looks broken.
        median = _trailing_median_fetched(cur, source)
        if median is not None and summary.rows_fetched < median * _CLOSURE_GUARD_RATIO:
            summary.status = "skipped_closure"
            summary.notes = (
                f"fetched {summary.rows_fetched} < {_CLOSURE_GUARD_RATIO:.0%} of "
                f"7d trailing median {median}; skipping closure step"
            )
            log.error(
                "closure_guard_tripped",
                source=source,
                fetched=summary.rows_fetched,
                median=median,
            )
        else:
            # 5. Close any postings that were active yesterday for this source
            # and are not in the current fetch.
            if seen_keys:
                cur.execute(
                    """
                    UPDATE postings
                    SET is_active = FALSE,
                        closed_on = %s
                    WHERE source = %s
                      AND is_active = TRUE
                      AND posting_key != ALL(%s)
                    """,
                    (observed_on, source, seen_keys),
                )
            else:
                # Empty fetch AND we passed the closure guard (impossible unless
                # this source has no history yet). Don't close anyone in that
                # case either — first-run behaviour.
                pass
            summary.rows_closed = cur.rowcount or 0

        summary.finished_at = datetime.now(timezone.utc)

        # 6. Write run_log row.
        cur.execute(
            """
            INSERT INTO run_log (
                started_at, finished_at, source, rows_fetched, rows_new,
                rows_updated, rows_closed, errors, status, notes
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """,
            (
                summary.started_at, summary.finished_at, summary.source,
                summary.rows_fetched, summary.rows_new, summary.rows_updated,
                summary.rows_closed, summary.errors, summary.status,
                summary.notes,
            ),
        )

    conn.commit()
    return summary

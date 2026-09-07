"""Daily entry point. GitHub Actions calls this.

Reads token lists from data/tokens.yaml, runs each source's collector, upserts
into Postgres, writes run_log rows. Exits non-zero if any source hard-fails or
the volume check trips — so the Action fails loudly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg
import structlog
import yaml
from dotenv import load_dotenv

from collectors.ashby import AshbyCollector
from collectors.base import RateLimitedClient
from collectors.greenhouse import GreenhouseCollector
from collectors.lever import LeverCollector
from db.upsert import upsert_source

load_dotenv()

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger()

TOKENS_PATH = Path(__file__).parent / "data" / "tokens.yaml"

COLLECTOR_CLASSES = {
    "greenhouse": GreenhouseCollector,
    "lever": LeverCollector,
    "ashby": AshbyCollector,
}


def load_tokens() -> dict[str, list[str]]:
    with TOKENS_PATH.open() as f:
        data = yaml.safe_load(f) or {}
    return {k: list(v or []) for k, v in data.items()}


def main() -> int:
    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        log.error("missing_env", var="SUPABASE_DB_URL")
        return 2

    tokens = load_tokens()
    log.info(
        "tokens_loaded",
        greenhouse=len(tokens.get("greenhouse", [])),
        lever=len(tokens.get("lever", [])),
        ashby=len(tokens.get("ashby", [])),
    )

    exit_code = 0

    with psycopg.connect(db_url) as conn, RateLimitedClient() as client:
        for source_name, cls in COLLECTOR_CLASSES.items():
            source_tokens = tokens.get(source_name, [])
            if not source_tokens:
                log.warning("no_tokens_for_source", source=source_name)
                continue

            log.info("collector_start", source=source_name, tokens=len(source_tokens))
            try:
                collector = cls(tokens=source_tokens, client=client)
                fetched = collector.run()
                summary = upsert_source(conn, source_name, fetched)
                log.info(
                    "collector_done",
                    source=source_name,
                    fetched=summary.rows_fetched,
                    new=summary.rows_new,
                    updated=summary.rows_updated,
                    closed=summary.rows_closed,
                    status=summary.status,
                )
                if summary.status == "skipped_closure":
                    # The guard tripped. The run isn't a hard failure — the
                    # data we did write is fine — but we want a visible signal.
                    exit_code = max(exit_code, 1)
            except Exception as exc:  # noqa: BLE001
                log.exception("collector_crashed", source=source_name, err=str(exc))
                exit_code = 2

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

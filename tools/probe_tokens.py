"""Probe every token in data/tokens.yaml against its ATS API.

For each token: does the endpoint resolve? How many total live postings does
it return? How many pass the India location filter? Prints a table so we can
prune tokens that are stale, empty, or have zero India presence.

Run: python -m tools.probe_tokens
"""

from __future__ import annotations

import sys
from pathlib import Path

import structlog
import yaml

from collectors.ashby import AshbyCollector
from collectors.base import RateLimitedClient, is_india_location
from collectors.greenhouse import GreenhouseCollector
from collectors.lever import LeverCollector

structlog.configure(processors=[structlog.dev.ConsoleRenderer()])
log = structlog.get_logger()

CLASSES = {
    "greenhouse": GreenhouseCollector,
    "lever": LeverCollector,
    "ashby": AshbyCollector,
}


def main() -> int:
    tokens_path = Path(__file__).parent.parent / "data" / "tokens.yaml"
    with tokens_path.open() as f:
        tokens = yaml.safe_load(f) or {}

    print(f"{'source':<12} {'token':<24} {'total':>6} {'india':>6}  status")
    print("-" * 68)

    with RateLimitedClient() as client:
        for source, cls in CLASSES.items():
            for token in tokens.get(source, []) or []:
                collector = cls(tokens=[token], client=client)
                try:
                    postings = list(collector.fetch_for_token(token))
                    total = len(postings)
                    india = sum(1 for p in postings if is_india_location(p.location_raw))
                    status = "ok" if total > 0 else "empty"
                    print(f"{source:<12} {token:<24} {total:>6} {india:>6}  {status}")
                except Exception as exc:  # noqa: BLE001
                    err = str(exc)[:40]
                    print(f"{source:<12} {token:<24} {'-':>6} {'-':>6}  ERR {err}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Gemini title classifier, cached in title_llm_cache.

Each distinct (company, title) is classified once per (model, prompt_version)
and cached in the DB, so replays are free and reproducible: the same cache
row is reused until the prompt or model changes. Company is part of the key
because level ladders are company-specific ("Engineer 3" at MongoDB vs
"L3" at Twilio).

The prompt is built from the same FAMILIES / SENIORITIES text used as the
human labelling guide, and the response is constrained to those enums by a
JSON schema — the model can't invent a family.

Run:
  python -m extraction.titles_llm --list-models    # which models this key can use
"""

from __future__ import annotations

import json
import os
import re
import time

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from extraction.titles import FAMILIES, SENIORITIES

log = structlog.get_logger()

# Bump on any prompt or taxonomy change: old cache rows stay for comparison
# but stop being read.
PROMPT_VERSION = 1
DEFAULT_MODEL = "gemini-3.8-flash"
BATCH_SIZE = 40
_MIN_SECONDS_BETWEEN_CALLS = 4.0  # free tier is ~15 requests/minute

_WS = re.compile(r"\s+")


def title_key(company: str | None, title: str | None) -> str:
    norm = lambda s: _WS.sub(" ", (s or "").strip().lower())  # noqa: E731
    return f"{norm(company)}|{norm(title)}"


def model_name() -> str:
    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)


def _guide(d: dict[str, str]) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in d.items())


SYSTEM_PROMPT = f"""You classify job titles from tech companies hiring in India.
For each title return a title_family (the role's function) and a seniority (its level).
Family and seniority are independent: an Engineering Manager is
software_engineering + manager.

title_family:
{_guide(FAMILIES)}

seniority:
{_guide(SENIORITIES)}

Judge from the title and company only. Use "unspecified" when the title has no
level marker; do not infer seniority from the company or the team name.
Return exactly one result per input id."""

RESPONSE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "title_family": {"type": "string", "enum": list(FAMILIES)},
            "seniority": {"type": "string", "enum": list(SENIORITIES)},
        },
        "required": ["id", "title_family", "seniority"],
    },
}


def _client():
    from google import genai

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY is not set (see .env.example)")
    return genai.Client(api_key=key)


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=4, max=60), reraise=True)
def _call(client, model: str, items: list[tuple[str, str]]) -> list[dict]:
    from google.genai import types

    payload = "\n".join(
        json.dumps({"id": i, "company": c, "title": t}, ensure_ascii=False)
        for i, (c, t) in enumerate(items)
    )
    resp = client.models.generate_content(
        model=model,
        contents=payload,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
            seed=0,
            response_mime_type="application/json",
            response_json_schema=RESPONSE_SCHEMA,
        ),
    )
    results = json.loads(resp.text)
    ids = sorted(r["id"] for r in results)
    if ids != list(range(len(items))):
        # Partial or duplicated answers would silently misalign labels.
        raise ValueError(f"expected ids 0..{len(items) - 1}, got {ids}")
    return sorted(results, key=lambda r: r["id"])


def classify(items: list[tuple[str, str]], model: str | None = None) -> list[tuple[str, str]]:
    """Classify (company, title) pairs, in order. No caching — see fill_cache."""
    model = model or model_name()
    client = _client()
    out: list[tuple[str, str]] = []
    last_call = 0.0
    for start in range(0, len(items), BATCH_SIZE):
        batch = items[start:start + BATCH_SIZE]
        wait = _MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - last_call)
        if wait > 0:
            time.sleep(wait)
        last_call = time.monotonic()
        results = _call(client, model, batch)
        out.extend((r["title_family"], r["seniority"]) for r in results)
        log.info("llm_batch", model=model, done=len(out), total=len(items))
    return out


# --- Cache ------------------------------------------------------------------


def load_cache(conn, model: str | None = None) -> dict[str, tuple[str, str]]:
    model = model or model_name()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT title_key, title_family, seniority FROM title_llm_cache"
            " WHERE model = %s AND prompt_version = %s",
            (model, PROMPT_VERSION),
        )
        return {k: (f, s) for k, f, s in cur.fetchall()}


def fill_cache(conn, pairs: list[tuple[str, str]], model: str | None = None) -> int:
    """Classify every (company, title) not yet cached. Returns how many were added."""
    model = model or model_name()
    cached = load_cache(conn, model)
    todo, seen = [], set()
    for company, title in pairs:
        k = title_key(company, title)
        if k not in cached and k not in seen:
            seen.add(k)
            todo.append((company, title))
    if not todo:
        return 0

    labels = classify(todo, model)
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO title_llm_cache
                (title_key, model, prompt_version, company, title_raw, title_family, seniority)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            [
                (title_key(c, t), model, PROMPT_VERSION, c, t, f, s)
                for (c, t), (f, s) in zip(todo, labels, strict=True)
            ],
        )
    return len(todo)


def main() -> int:
    import argparse

    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description="Gemini helpers")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    if args.list_models:
        client = _client()  # keep a reference: the SDK closes clients on GC
        for m in client.models.list():
            if "generateContent" in (m.supported_actions or []):
                print(m.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""A7 — How much of the market is duplicated?

Every other analysis treats rows as independent postings. This one tests
that, and measures how wrong it is.

Similarity: TF-IDF (1–2 grams) over descriptions with company boilerplate
removed (extraction/text.py), cosine, all pairs within a company. With ~600
postings exact all-pairs is instant, so MinHash/LSH would only add
approximation error. Cross-company pairs are checked once and reported (no
syndication exists here: everything comes straight from employer ATSs).

A pair with similarity >= THRESHOLD is a duplicate. The threshold sits at the
obvious mode in the nearest-neighbour similarity distribution (see the
figure); 0.90 is reported as sensitivity. Duplicates are typed by title
(same / different after dropping location suffixes) and timing:

  multi_location   same title, open at the same time (one role, several cities)
  multi_level      different title, same text, open together (one JD, several levels)
  repost           same title, the second appears <= REPOST_DAYS after the
                   first disappears (the role didn't close; its listing did)
  sequential_other everything else that is near-identical but not concurrent

Consequences are computed, not just described: the effective number of
distinct roles, and A2 refitted with repost closures censored, because a
closure followed by its own repost is not a filled role.

Run: python -m analysis.a7_duplication
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from analysis.common import FIGURES, REPO, _query, load_lifecycle, load_postings
from extraction.text import find_boilerplate, strip_boilerplate

THRESHOLD = 0.95
SENSITIVITY = 0.90
REPOST_DAYS = 14

_LOC_SUFFIX = re.compile(
    r"\s*[-–,(|]\s*(?:india|bengaluru|bangalore|gurugram|gurgaon|mumbai|delhi|new delhi|"
    r"hyderabad|pune|chennai|remote)\b.*$", re.I)


def norm_title(t: str) -> str:
    return re.sub(r"\s+", " ", _LOC_SUFFIX.sub("", t or "")).strip().lower()


def load_texts() -> pd.DataFrame:
    import os

    import psycopg
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    with psycopg.connect(os.environ["SUPABASE_DB_URL"]) as conn:
        d = _query(conn, "SELECT posting_key, description_raw FROM postings")
    postings, _ = load_postings()
    df = postings.merge(d, on="posting_key").reset_index(drop=True)
    bp = find_boilerplate(zip(df.company, df.description_raw, strict=True))
    df["text"] = [strip_boilerplate(desc, bp.get((co or "").lower(), set()))
                  for co, desc in zip(df.company, df.description_raw, strict=True)]
    df["title_norm"] = df.title_raw.map(norm_title)
    return df


def similarity(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2).fit_transform(df.text)
    S = cosine_similarity(X)
    np.fill_diagonal(S, 0)
    same = df.company.to_numpy()[:, None] == df.company.to_numpy()[None, :]
    return S, same


def classify_pairs(df: pd.DataFrame, S: np.ndarray, same: np.ndarray, t: float) -> pd.DataFrame:
    rows = []
    for i, j in np.argwhere(np.triu(np.where(same, S, 0)) >= t):
        a, b = df.iloc[i], df.iloc[j]
        if a.first_seen > b.first_seen:
            a, b, i, j = b, a, j, i
        overlap = b.first_seen <= a.last_seen
        same_title = a.title_norm == b.title_norm
        gap = None if overlap else (b.first_seen - a.last_seen).days
        if overlap:
            kind = "multi_location" if same_title else "multi_level"
        elif same_title and gap <= REPOST_DAYS:
            kind = "repost"
        else:
            kind = "sequential_other"
        rows.append({"i": i, "j": j, "sim": float(S[i, j]), "company": a.company,
                     "kind": kind, "gap_days": gap, "earlier": a.posting_key,
                     "later": b.posting_key, "title_a": a.title_raw, "title_b": b.title_raw})
    return pd.DataFrame(rows)


def clusters(n: int, pairs: pd.DataFrame) -> np.ndarray:
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(pairs.i, pairs.j, strict=True):
        parent[find(i)] = find(j)
    return np.array([find(x) for x in range(n)])


def a2_without_reposts(repost_keys: set[str]) -> dict:
    """Refit A2's main within-company model with repost closures censored."""
    from lifelines import KaplanMeierFitter
    from lifelines.utils import median_survival_times

    from analysis import a2_survival as a2

    postings, skills = load_postings()
    published, obs = load_lifecycle()
    spells, _ = a2.build_spells(postings, published, obs)
    base = spells.merge(a2.covariates(postings, skills), on="posting_key")
    out = {}
    for label, d in (("as_in_A2", base), ("reposts_censored", base.assign(
            event=np.where(base.posting_key.isin(repost_keys), 0, base.event)))):
        km = KaplanMeierFitter().fit(d.duration, d.event, entry=d.entry)
        lo, hi = median_survival_times(km.confidence_interval_).iloc[0]
        cph, _ = a2.fit_cox(a2.design(d), entry=True, strata=True)
        hr = a2.hr_table(cph)
        out[label] = {
            "events": int(d.event.sum()), "km_median": float(km.median_survival_time_),
            "km_ci": [float(lo), float(hi)],
            "hr": {k: hr.loc[k].to_dict() for k in ("seniority=entry_mid", "workplace=onsite")
                   if k in hr.index},
        }
    return out


def plot_nn(nn: np.ndarray, path) -> None:
    from analysis.a1_clustering import INK_2, MUTED, SERIES_1, _style

    plt = _style()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.hist(nn, bins=np.arange(0, 1.0001, 0.025), color=SERIES_1, edgecolor="#fcfcfb",
            linewidth=1.5)
    for t, ls in ((THRESHOLD, "-"), (SENSITIVITY, ":")):
        ax.axvline(t, color=MUTED, lw=1.2, ls=ls)
        ax.annotate(f"{t:.2f}", (t, ax.get_ylim()[1] * 0.92), xytext=(-28, 0),
                    textcoords="offset points", color=INK_2, fontsize=8)
    ax.set(xlabel="cosine similarity to the most similar other posting at the same company",
           ylabel="postings")
    ax.set_title("A7 — nearest-neighbour similarity (descriptions, boilerplate removed)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    df = load_texts()
    S, same = similarity(df)
    nn = np.where(same, S, 0).max(axis=1)
    cross_max = float(np.where(~same, S, 0).max())

    print("=== A7: duplication ===")
    print(f"  postings {len(df)}; cross-company max similarity {cross_max:.2f} (no syndication)")
    results = {"n": len(df), "cross_company_max_sim": cross_max, "threshold": THRESHOLD}
    for t in (THRESHOLD, SENSITIVITY):
        pairs = classify_pairs(df, S, same, t)
        cl = clusters(len(df), pairs)
        n_roles = len(np.unique(cl))
        ins = df.in_scope.to_numpy()
        n_roles_in = len(np.unique(cl[ins]))
        kinds = pairs.kind.value_counts().to_dict()
        in_dup = np.isin(np.arange(len(df)), np.concatenate([pairs.i, pairs.j]))
        print(f"\n  threshold {t:.2f}: {len(pairs)} duplicate pairs; {in_dup.sum()} postings"
              f" ({in_dup.mean():.0%}) have a near-identical sibling")
        print(f"    distinct roles: {n_roles} of {len(df)} postings"
              f" (duplication {1 - n_roles / len(df):.0%});"
              f" in-scope {n_roles_in} of {ins.sum()} ({1 - n_roles_in / ins.sum():.0%})")
        print(f"    pair types: {kinds}")
        results[f"t{t:.2f}"] = {"pairs": len(pairs), "postings_with_sibling": int(in_dup.sum()),
                                "distinct_roles": n_roles, "distinct_roles_in_scope": n_roles_in,
                                "in_scope": int(ins.sum()), "pair_types": kinds}
        if t == THRESHOLD:
            main_pairs = pairs
            by_co = (pd.DataFrame({"company": df.company, "cluster": cl})
                     .groupby("company").agg(postings=("cluster", "size"),
                                             roles=("cluster", "nunique")))
            by_co["duplication"] = 1 - by_co.roles / by_co.postings
            by_co = by_co.sort_values("postings", ascending=False)
            print("    by employer:")
            print("      " + by_co.round(2).to_string().replace("\n", "\n      "))
            results["by_employer"] = by_co.reset_index().round(3).to_dict(orient="records")

    print("\n  examples:")
    for kind, g in main_pairs.groupby("kind"):
        for r in g.head(3).itertuples():
            gap = f", gap {r.gap_days:.0f} d" if pd.notna(r.gap_days) else ""
            print(f"    [{kind}] {r.company}: {r.title_a[:45]} ↔ {r.title_b[:45]}"
                  f" (sim {r.sim:.2f}{gap})")

    # Same-key relisting: the posting vanished and came back under the same ID.
    _, obs = load_lifecycle()
    days = sorted(obs.observed_on.unique())
    nxt = {d: days[i + 1] for i, d in enumerate(days[:-1])}
    seen = obs.groupby("posting_key").observed_on.apply(lambda s: sorted(set(s)))
    relisted = sum(any(b != nxt[a] for a, b in zip(s, s[1:], strict=False)) for s in seen)
    print(f"\n  same-key relisting (vanished, returned under the same ID): {relisted} postings")
    results["same_key_relisted"] = int(relisted)

    repost_keys = set(main_pairs.loc[main_pairs.kind == "repost", "earlier"])
    a2r = a2_without_reposts(repost_keys)
    results["a2_sensitivity"] = a2r
    print(f"\n  A2 with repost closures censored ({len(repost_keys)} closures were reposts):")
    for label, r in a2r.items():
        hrs = ", ".join(f"{k} {v['hr']:.2f} ({v['lo']:.2f}–{v['hi']:.2f})"
                        for k, v in r["hr"].items())
        print(f"    {label:<17} events {r['events']}, KM median {r['km_median']:.0f} d"
              f" ({r['km_ci'][0]:.0f}–{r['km_ci'][1]:.0f}); within-company {hrs}")

    plot_nn(nn, FIGURES / "a7_nn_similarity.png")
    main_pairs.drop(columns=["i", "j"]).to_csv(REPO / "report" / "a7_duplicate_pairs.csv",
                                                index=False)
    out = REPO / "report" / "a7_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  wrote {out.relative_to(REPO)}, report/a7_duplicate_pairs.csv, figure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""A3 — Has the skill mix actually shifted?

Implements data/a3_preregistration.yaml exactly; every design choice
(periods, population, skill list, tests, multiplicity) is read from there.
The code was written and committed with the pre-registration on 2026-10-18,
before any period-2 posting existed.

It refuses to compare the real periods before the pre-registered run date.
The only thing it will do earlier is a calibration check: shuffle period
labels across the postings available so far and confirm the tests reject at
their nominal rates. That uses no information about real trends.

Run:
  python -m analysis.a3_drift --permutation-check   # any time
  python -m analysis.a3_drift                       # on/after 2026-12-01
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

from analysis.common import REPO, load_lifecycle, load_postings

PREREG = REPO / "data" / "a3_preregistration.yaml"
SEED = 20261018


def load_prereg() -> dict:
    return yaml.safe_load(PREREG.read_text())


# --- Statistics -------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def two_prop(k1: int, n1: int, k2: int, n2: int) -> dict:
    """p2 - p1 with Newcombe's hybrid-score CI and a pooled two-sided z-test."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p2 - p1
    lo = d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)
    hi = d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
    pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    z = d / se if se > 0 else 0.0
    return {"p1": p1, "p2": p2, "diff": d, "lo": lo, "hi": hi, "z": z,
            "p": float(2 * norm.sf(abs(z))), "n1": n1, "n2": n2}


def bh(pvals: list[float], q: float) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini–Hochberg: -> (adjusted p-values, reject mask)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    out = np.empty(m)
    out[order] = adj
    return out, out <= q


def mantel_haenszel(df: pd.DataFrame, outcome: str) -> dict:
    """Company-stratified OR for period 2 vs 1, over companies in both periods."""
    from statsmodels.stats.contingency_tables import StratifiedTable

    tables = []
    for _, g in df.groupby("company"):
        if g.period.nunique() < 2:
            continue
        t = np.array([[((g.period == p) & g[outcome]).sum(),
                       ((g.period == p) & ~g[outcome]).sum()] for p in ("p2", "p1")])
        tables.append(t + 0.5 if (t == 0).any() else t)  # Haldane for empty cells
    if not tables:
        return {"strata": 0}
    st = StratifiedTable(tables)
    lo, hi = st.oddsratio_pooled_confint()
    return {"strata": len(tables), "or": float(st.oddsratio_pooled),
            "lo": float(lo), "hi": float(hi), "p": float(st.test_null_odds().pvalue)}


# --- Data -------------------------------------------------------------------


def build_frame(prereg: dict) -> pd.DataFrame:
    postings, skills = load_postings()
    published, _ = load_lifecycle()
    df = postings[postings.in_scope].merge(published, on="posting_key")

    def period(d):
        for name, w in prereg["periods"].items():
            if w["start"] <= d <= w["end"]:
                return name
        return None

    df["period"] = df.published_on.map(period)
    has = skills.groupby("posting_key").skill.apply(set)
    df["skills"] = df.posting_key.map(has).apply(lambda s: s if isinstance(s, set) else set())
    genai = set(prereg["primary"]["genai_skills"])
    df["any_genai"] = df.skills.apply(lambda s: bool(s & genai))
    for sk in prereg["secondary"]["per_skill"]["skills"]:
        df[f"skill:{sk}"] = df.skills.apply(lambda s, sk=sk: sk in s)
    start = prereg["periods"]["p1"]["start"]
    df["weeks"] = df.published_on.map(lambda d: (d - start).days / 7)
    return df


# --- Analyses ---------------------------------------------------------------


def primary(df: pd.DataFrame) -> dict:
    a = df[df.period == "p1"]
    b = df[df.period == "p2"]
    res = two_prop(int(a.any_genai.sum()), len(a), int(b.any_genai.sum()), len(b))
    res["mantel_haenszel"] = mantel_haenszel(df[df.period.notna()], "any_genai")
    return res


def per_skill(df: pd.DataFrame, prereg: dict) -> pd.DataFrame:
    a = df[df.period == "p1"]
    b = df[df.period == "p2"]
    rows = []
    for sk in prereg["secondary"]["per_skill"]["skills"]:
        c = f"skill:{sk}"
        rows.append({"skill": sk, **two_prop(int(a[c].sum()), len(a), int(b[c].sum()), len(b))})
    t = pd.DataFrame(rows)
    q = float(prereg["secondary"]["per_skill"]["multiplicity"]["q"])
    t["p_bh"], t["bh_reject"] = bh(t.p.tolist(), q=q)
    return t.reindex(t["diff"].abs().sort_values(ascending=False).index)


def trend(df: pd.DataFrame, prereg: dict) -> dict:
    import statsmodels.formula.api as smf

    end = prereg["periods"]["p2"]["end"]
    d = df[(df.published_on >= prereg["periods"]["p1"]["start"]) & (df.published_on <= end)]
    d = d.assign(y=d.any_genai.astype(int))
    # Companies with few postings pooled so fixed effects are estimable.
    counts = d.company.value_counts()
    d = d.assign(co=d.company.where(d.company.map(counts) >= 5, "other"))
    m = smf.logit("y ~ weeks + C(co)", data=d).fit(disp=False, cov_type="HC1")
    b, se = m.params["weeks"], m.bse["weeks"]
    return {"n": len(d), "or_per_4_weeks": math.exp(4 * b),
            "lo": math.exp(4 * (b - 1.96 * se)), "hi": math.exp(4 * (b + 1.96 * se)),
            "p": float(m.pvalues["weeks"])}


def stable_panel(df: pd.DataFrame) -> dict:
    d = df[df.period.notna()]
    both = d.groupby("company").period.nunique()
    keep = both[both == 2].index
    s = d[d.company.isin(keep)]
    a, b = s[s.period == "p1"], s[s.period == "p2"]
    return {"companies": list(keep),
            **two_prop(int(a.any_genai.sum()), len(a), int(b.any_genai.sum()), len(b))}


# --- Calibration ------------------------------------------------------------


def permutation_check(df: pd.DataFrame, prereg: dict, reps: int = 1000) -> dict:
    """Shuffle postings into two pseudo-periods of the expected sizes and
    check the error rates. Real period labels are never used."""
    rng = np.random.default_rng(SEED)
    pool = df[df.published_on >= prereg["periods"]["p1"]["start"]].reset_index(drop=True)
    n1 = int((pool.period == "p1").sum()) or 42
    n2 = min(n1 + 3, len(pool) - n1)
    skills = prereg["secondary"]["per_skill"]["skills"]
    q = float(prereg["secondary"]["per_skill"]["multiplicity"]["q"])
    h1_rej, any_bh, false_disc = 0, 0, []
    for _ in range(reps):
        idx = rng.permutation(len(pool))
        a, b = pool.iloc[idx[:n1]], pool.iloc[idx[n1:n1 + n2]]
        r = two_prop(int(a.any_genai.sum()), n1, int(b.any_genai.sum()), n2)
        h1_rej += r["p"] < prereg["primary"]["alpha"]
        ps = [two_prop(int(a[f"skill:{s}"].sum()), n1, int(b[f"skill:{s}"].sum()), n2)["p"]
              for s in skills]
        _, rej = bh(ps, q=q)
        any_bh += rej.any()
        false_disc.append(int(rej.sum()))
    return {"reps": reps, "pool": len(pool), "n1": n1, "n2": n2,
            "h1_type1_rate": h1_rej / reps,
            "bh_any_rejection_rate": any_bh / reps,
            "bh_mean_false_discoveries": float(np.mean(false_disc))}


# --- Main -------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="A3 — pre-registered skill drift")
    parser.add_argument("--permutation-check", action="store_true",
                        help="calibrate tests on shuffled labels (allowed any time)")
    args = parser.parse_args()

    prereg = load_prereg()
    df = build_frame(prereg)

    if args.permutation_check:
        r = permutation_check(df, prereg)
        print("=== A3 calibration (shuffled period labels; no real comparison) ===")
        print(f"  pool {r['pool']} in-scope postings published since panel start;"
              f" pseudo-periods n={r['n1']} / {r['n2']}, {r['reps']} shuffles")
        print(f"  H1 type-I error: {r['h1_type1_rate']:.3f} (nominal 0.05)")
        print(f"  BH (q={prereg['secondary']['per_skill']['multiplicity']['q']}) across"
              f" {len(prereg['secondary']['per_skill']['skills'])} skills:"
              f" P(any false discovery) {r['bh_any_rejection_rate']:.3f},"
              f" mean false discoveries {r['bh_mean_false_discoveries']:.3f}")
        return 0

    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    earliest = prereg["earliest_run_date"]
    if isinstance(earliest, str):
        earliest = date.fromisoformat(earliest)
    if today < earliest:
        raise SystemExit(f"pre-registered run date is {earliest}; today is {today}."
                         " Only --permutation-check is allowed before then.")

    res = {"prereg": str(PREREG.relative_to(REPO)), "run_on": str(today),
           "primary": primary(df), "trend": trend(df, prereg),
           "stable_panel": stable_panel(df)}
    t = per_skill(df, prereg)
    res["per_skill"] = t.round(4).to_dict(orient="records")

    h = res["primary"]
    print("=== A3: pre-registered skill drift ===")
    print(f"  H1 any-GenAI: p1 {h['p1']:.1%} (n={h['n1']}) → p2 {h['p2']:.1%} (n={h['n2']});"
          f" diff {h['diff']*100:+.1f} pp (95% {h['lo']*100:+.1f} to {h['hi']*100:+.1f}),"
          f" p={h['p']:.3f}")
    print(f"  composition-adjusted (MH by company): {h['mantel_haenszel']}")
    print(f"  trend: {res['trend']}")
    print(f"  stable panel: {res['stable_panel']}")
    print(f"  per-skill (BH): {int(t.bh_reject.sum())} of {len(t)} survive")
    print("    " + t[["skill", "p1", "p2", "diff", "lo", "hi", "p", "p_bh", "bh_reject"]]
          .round(3).to_string(index=False).replace("\n", "\n    "))
    out = REPO / "report" / "a3_results.json"
    out.write_text(json.dumps(res, indent=2, default=str))
    print(f"\n  wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

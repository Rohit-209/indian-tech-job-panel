"""A6 (short form) — Is the market geographically specialised?

The plan asks whether Bengaluru, Hyderabad, Pune and NCR hire for different
things. This panel can only ask it of Bengaluru vs NCR (Hyderabad has one
posting, Pune none), and even there city is close to a proxy for employer:
most NCR postings are MongoDB, most Bengaluru postings are Okta. So:

  Concentration — Herfindahl index of employers per city (the plan's measure),
                  to show how much "city" is "employer".
  Specialisation — location quotients of family by city, and a chi-square of
                  city x family, pooled and then with a permutation test that
                  shuffles city *within employer*. Only employers that post in
                  both cities carry information in the second test.
  Remote share  — by family, on postings whose workplace is known, pooled and
                  within employer.

Run: python -m analysis.a6_geography
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from analysis.a2_survival import covariates
from analysis.a3_drift import wilson
from analysis.common import REPO, load_postings

CITIES = ["Bengaluru", "NCR"]
N_PERM = 10_000
SEED = 20261020


def hhi(companies: pd.Series) -> float:
    """Herfindahl index on 0–10,000: 10,000 = one employer."""
    s = companies.value_counts(normalize=True)
    return float((s ** 2).sum() * 10_000)


def location_quotients(df: pd.DataFrame) -> pd.DataFrame:
    """LQ = family's share in the city / family's share across both cities."""
    ct = pd.crosstab(df.family, df.city)
    city_share = ct / ct.sum()
    overall = ct.sum(axis=1) / ct.values.sum()
    return city_share.div(overall, axis=0).round(2)


def _chi2(city: np.ndarray, family: np.ndarray) -> float:
    ct = pd.crosstab(city, family).to_numpy()
    return float(chi2_contingency(ct, correction=False)[0])


def within_employer_test(df: pd.DataFrame, n: int = N_PERM, seed: int = SEED) -> dict:
    """Chi-square of city x family, null distribution from shuffling city
    within each employer. Employers in one city only are fixed under the
    shuffle, so they contribute nothing to the evidence."""
    rng = np.random.default_rng(seed)
    city, family = df.city.to_numpy(), df.family.to_numpy()
    groups = [np.flatnonzero(df.company.to_numpy() == c) for c in df.company.unique()]
    observed = _chi2(city, family)
    hits = 0
    for _ in range(n):
        perm = city.copy()
        for g in groups:
            perm[g] = rng.permutation(city[g])
        hits += _chi2(perm, family) >= observed
    both = df.groupby("company").city.nunique()
    return {"chi2": round(observed, 2), "p_perm": round((hits + 1) / (n + 1), 4),
            "employers_in_both_cities": sorted(both[both == 2].index),
            "n_informative": int(df.company.isin(both[both == 2].index).sum())}


def remote_share(df: pd.DataFrame) -> dict:
    known = df[df.workplace != "unknown"]
    out = {}
    for fam, g in known.groupby("family"):
        k, n = int((g.workplace == "remote").sum()), len(g)
        lo, hi = wilson(k, n)
        out[fam] = {"remote": k, "n": n, "share": round(k / n, 3),
                    "ci": [round(lo, 3), round(hi, 3)]}
    return out


def main() -> int:
    postings, skills = load_postings()
    df = covariates(postings, skills).merge(postings[["posting_key", "company"]],
                                            on="posting_key")
    raw_city = postings.set_index("posting_key").city
    print("=== A6: geography (short form) ===")
    print("  postings by canonical city: " + ", ".join(
        f"{k} {v}" for k, v in raw_city.fillna("none").value_counts().items()))

    results = {"n": len(df), "cities": raw_city.fillna("none").value_counts().to_dict(),
               "concentration": {}, "specialisation": {}, "remote": {}}
    for label, d in (("all", df), ("in_scope", df[df.in_scope])):
        two = d[d.city.isin(CITIES)]
        conc = {}
        for c in CITIES:
            g = two[two.city == c]
            top = g.company.value_counts()
            conc[c] = {"n": len(g), "hhi": round(hhi(g.company)), "top_employer": top.index[0],
                       "top_share": round(float(top.iloc[0] / len(g)), 3)}
        results["concentration"][label] = conc
        print(f"\n  [{label}] employer concentration (HHI 0–10,000):")
        for c, v in conc.items():
            print(f"    {c}: n={v['n']}, HHI {v['hhi']}, top {v['top_employer']}"
                  f" {v['top_share']:.0%}")

        lq = location_quotients(two)
        pooled = chi2_contingency(pd.crosstab(two.city, two.family), correction=False)
        within = within_employer_test(two)
        results["specialisation"][label] = {
            "location_quotients": lq.to_dict(),
            "pooled": {"chi2": round(float(pooled[0]), 2), "df": int(pooled[2]),
                       "p": round(float(pooled[1]), 4)},
            "within_employer": within,
        }
        print("    location quotients (family share in city / share in both):\n      "
              + lq.to_string().replace("\n", "\n      "))
        print(f"    city x family, pooled: chi2={pooled[0]:.1f} df={pooled[2]} p={pooled[1]:.4f}")
        print(f"    city x family, within employer (permutation): chi2={within['chi2']}"
              f" p={within['p_perm']}; informative employers"
              f" {within['employers_in_both_cities']} (n={within['n_informative']})")

    results["remote"]["pooled"] = rs = remote_share(df)
    print("\n  remote share by family (workplace known only):")
    for fam, v in sorted(rs.items(), key=lambda kv: -kv[1]["share"]):
        print(f"    {fam:<15} {v['remote']:>3}/{v['n']:<3} {v['share']:.0%}"
              f" ({v['ci'][0]:.0%}–{v['ci'][1]:.0%})")
    known = df[df.workplace != "unknown"]
    by_co = known.groupby("company").workplace.apply(lambda s: (s == "remote").mean())
    results["remote"]["by_employer"] = by_co.round(3).to_dict()
    print("  remote share by employer: " + ", ".join(
        f"{k} {v:.0%}" for k, v in by_co.sort_values(ascending=False).items()))

    out = REPO / "report" / "a6_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

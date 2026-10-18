"""A4 — Who discloses pay? (short form)

The plan's A4 has two parts: a log-salary regression on disclosed postings,
and a selection model for who discloses. Neither is estimable here, and this
script documents why rather than forcing them:

  - Disclosure is an employer policy. Airbnb shows an India pay range on
    most postings; no other employer shows one at all (GitLab's ranges are
    US-only; Okta's and MongoDB's ATS salary fields hold placeholder zeros).
  - A regression with company-clustered SEs needs more than one cluster.
  - A disclosure model would learn the company indicator and nothing else.

What *is* estimable: the disclosure rate by employer, whether disclosure
varies by role inside the one disclosing employer, and a descriptive look
at the ranges themselves.

Run: python -m analysis.a4_disclosure
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from analysis.a3_drift import wilson as wilson_ci
from analysis.common import REPO, _query, load_postings

ENGINEERING = {"software_engineering", "sre_infra", "data_engineering", "ml_engineering",
               "qa_testing", "security_engineering"}


def load() -> pd.DataFrame:
    import os

    import psycopg
    from dotenv import load_dotenv

    postings, _ = load_postings()
    load_dotenv(REPO / ".env")
    with psycopg.connect(os.environ["SUPABASE_DB_URL"]) as conn:
        sal = _query(conn, "SELECT posting_key, salary_min, salary_max, salary_currency"
                           " FROM postings")
    df = postings.merge(sal, on="posting_key")
    # India disclosure only: a US range on an India posting tells an Indian
    # candidate nothing about their own pay.
    df["discloses_india_pay"] = df.salary_currency.eq("INR")
    return df


def main() -> int:
    df = load()
    n, k = len(df), int(df.discloses_india_pay.sum())
    lo, hi = wilson_ci(k, n)
    print("=== A4: pay disclosure ===")
    print(f"  India pay range shown: {k}/{n} postings ({k / n:.1%}, 95% {lo:.1%}–{hi:.1%})")
    ins = df[df.in_scope]
    ki = int(ins.discloses_india_pay.sum())
    li, hi_ = wilson_ci(ki, len(ins))
    print(f"  in-scope technical IC: {ki}/{len(ins)} ({ki / len(ins):.1%}, 95% {li:.1%}–{hi_:.1%})")

    by = (df.groupby("company").discloses_india_pay.agg(["size", "sum"])
          .rename(columns={"size": "postings", "sum": "disclosed"})
          .sort_values("postings", ascending=False))
    by["rate"] = by.disclosed / by.postings
    by["ci"] = [f"{max(a, 0):.0%}–{b:.0%}" for a, b in (wilson_ci(int(d), int(p))
                                                for d, p in zip(by.disclosed, by.postings,
                                                                strict=True))]
    print("\n  by employer:")
    print("    " + by.to_string().replace("\n", "\n    "))

    air = df[df.company == "Airbnb"].copy()
    air["engineering"] = air.title_family.isin(ENGINEERING)
    t = pd.crosstab(air.engineering, air.discloses_india_pay)
    t = t.reindex(index=[True, False], columns=[True, False], fill_value=0)
    odds, p = fisher_exact(t.to_numpy())
    print("\n  within Airbnb — engineering vs other roles:")
    print("    " + t.rename(index={True: "engineering", False: "other"},
                            columns={True: "disclosed", False: "not"})
          .to_string().replace("\n", "\n    "))
    print(f"    Fisher exact p = {p:.4f}")

    d = air[air.discloses_india_pay].assign(
        mid_lakh=lambda x: (x.salary_min.astype(float) + x.salary_max.astype(float)) / 2 / 1e5,
        width_pct=lambda x: (x.salary_max.astype(float) / x.salary_min.astype(float) - 1) * 100,
    ).sort_values("mid_lakh")
    print("\n  Airbnb India ranges (lakh ₹/yr, midpoint) by family:")
    print("    " + d.groupby("title_family").mid_lakh.agg(["size", "median", "min", "max"])
          .round(1).to_string().replace("\n", "\n    "))
    print(f"    range width: median {d.width_pct.median():.0f}% above the minimum"
          f" (IQR {d.width_pct.quantile(.25):.0f}–{d.width_pct.quantile(.75):.0f}%)")

    results = {
        "n": n, "disclosed": k, "rate": k / n, "rate_ci": [lo, hi],
        "in_scope": {"n": len(ins), "disclosed": ki},
        "by_employer": by.drop(columns="ci").reset_index().to_dict(orient="records"),
        "airbnb_engineering_table": t.to_numpy().tolist(), "airbnb_fisher_p": p,
        "airbnb_ranges": d[["title_raw", "title_family", "seniority", "salary_min",
                            "salary_max"]].astype({"salary_min": float, "salary_max": float})
        .to_dict(orient="records"),
        "range_width_median_pct": float(np.median(d.width_pct)),
    }
    out = REPO / "report" / "a4_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""A2 — How long do postings stay open, and what predicts closure?

Survival analysis on the panel, with the three problems that make the naive
mean wrong handled explicitly:

  Right censoring  — postings still open on the analysis date have lasted
                     *at least* this long. Kaplan–Meier / Cox use them as such.
  Left truncation  — 332 postings were already live on panel day 1. Time is
                     measured from the employer's own ATS publish date, and
                     each posting enters the risk set at its age when we first
                     saw it (delayed entry). A posting first seen at age 53
                     says nothing about the hazard before day 53.
  Fake closures    — a posting can vanish because its company's whole job
                     board disappeared (Postman moved off Greenhouse on
                     2026-09-17). Those are censored at their last sighting,
                     not counted as filled. Detected from the data: a company
                     whose remaining postings all vanish on one day and never
                     come back.

Specifications (same covariates):
  main      origin = publish date, delayed entry, event = final disappearance
  born      only postings we saw appear within 1 day of publishing; origin =
            first sighting, no truncation assumption at all
  first_gap event = first disappearance, even if the posting later returned
  no_30d    closures at 28–32 days censored, in case the ~30-day bump is
            expiry rather than hiring

Company is the confound to beat (Okta is a third of postings), so the Cox
model is fitted with company-clustered standard errors, and again stratified
by company so families are compared *within* employers.

Run: python -m analysis.a2_survival
"""

from __future__ import annotations

import json
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import median_survival_times

from analysis.common import FIGURES, REPO, load_lifecycle, load_postings

BORN_MAX_LAG_DAYS = 1      # "born in window": first seen within this of publishing
SOURCE_DEATH_MIN = 3       # company-wide vanish of >= this many postings = board gone
SOURCE_DEATH_QUIET = 3     # ...and nothing from that company for >= this many days after

FAMILY_GROUP = {
    "software_engineering": "software_eng",
    "sre_infra": "infra",
    "data_engineering": "data_ml", "data_science": "data_ml",
    "data_analytics": "data_ml", "ml_engineering": "data_ml",
    "qa_testing": "other_tech", "security_engineering": "other_tech",
    "solutions_support": "solutions",
    "product_design": "product_design",
    "non_tech": "non_tech",
}
SENIORITY_GROUP = {
    "intern": "entry_mid", "junior": "entry_mid", "mid": "entry_mid",
    "senior": "senior", "staff": "staff", "principal": "principal",
    "manager": "manager_plus", "director": "manager_plus",
    "unspecified": "unspecified",
}
REFERENCE = {"family": "software_eng", "seniority": "senior", "city": "Bengaluru",
             "workplace": "hybrid"}


# --- Spells -----------------------------------------------------------------


def source_deaths(postings: pd.DataFrame, panel_end) -> dict[str, object]:
    """company -> last day its board was seen, for boards that disappeared."""
    out = {}
    for company, g in postings.groupby("company"):
        last = g.last_seen.max()
        if (panel_end - last).days < SOURCE_DEATH_QUIET:
            continue
        vanished = g[(g.last_seen == last)]
        if len(vanished) >= SOURCE_DEATH_MIN and not g.is_active.any():
            out[company] = last
    return out


def build_spells(postings, published, obs) -> tuple[pd.DataFrame, dict]:
    panel_start, panel_end = obs.observed_on.min(), obs.observed_on.max()
    df = postings.merge(published, on="posting_key")
    deaths = source_deaths(df, panel_end)

    seen = obs.groupby("posting_key").observed_on.apply(lambda s: sorted(set(s)))
    days = sorted(obs.observed_on.unique())
    nxt = {d: days[i + 1] for i, d in enumerate(days[:-1])}

    def first_gap(key):
        s = seen[key]
        for a, b in zip(s, s[1:], strict=False):
            if b != nxt[a]:
                return nxt[a]  # first day it was missing
        return None

    rows = []
    for r in df.itertuples():
        died = r.company in deaths
        event = (not r.is_active) and not died
        # A closure happens after the last sighting. (On panel day 1 two runs
        # happened; the second "closed" five postings the first had just seen,
        # leaving closed_on == first_seen.)
        end = (max(r.closed_on, r.last_seen + timedelta(days=1)) if event
               else r.last_seen + timedelta(days=1))
        # Spell length in days from publish; a censored posting is known to
        # be open through the end of its last sighting.
        rows.append({
            "posting_key": r.posting_key, "company": r.company,
            "published_on": r.published_on, "first_seen": r.first_seen,
            "entry": (r.first_seen - r.published_on).days,
            "duration": (end - r.published_on).days,
            "event": int(event),
            "censored_by_source_death": died and not r.is_active,
            "left_truncated": r.first_seen == panel_start,
            "first_gap_on": first_gap(r.posting_key),
        })
    spells = pd.DataFrame(rows)
    meta = {"panel_start": str(panel_start), "panel_end": str(panel_end),
            "source_deaths": {k: str(v) for k, v in deaths.items()}}
    return spells, meta


def covariates(postings: pd.DataFrame, skills: pd.DataFrame) -> pd.DataFrame:
    p = postings.copy()
    p["family"] = p.title_family.map(FAMILY_GROUP)
    p["seniority"] = p.seniority.map(SENIORITY_GROUP)
    p["city"] = p.city.where(p.city.isin(["Bengaluru", "NCR"]),
                             np.where(p.city.isna(), "no_city", "other_city"))
    # Unknown workplace is its own level: its missingness is employer-driven,
    # so dropping it would drop whole companies.
    p["workplace"] = np.select(
        [p.is_remote == True, p.is_hybrid == True,  # noqa: E712 — nullable booleans
         (p.is_remote == False) & (p.is_hybrid == False)],  # noqa: E712
        ["remote", "hybrid", "onsite"], default="unknown",
    )
    n_skills = skills.groupby("posting_key").size()
    p["skills_per_5"] = p.posting_key.map(n_skills).fillna(0) / 5
    return p[["posting_key", "family", "seniority", "city", "workplace", "skills_per_5",
              "in_scope"]]


def design(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot with explicit reference levels, so hazard ratios read as
    'vs software_eng / senior / Bengaluru / hybrid'."""
    out = df[["duration", "event", "entry", "company", "skills_per_5"]].copy()
    for col, ref in REFERENCE.items():
        levels = [v for v in sorted(df[col].unique()) if v != ref]
        for v in levels:
            out[f"{col}={v}"] = (df[col] == v).astype(float)
    # Drop levels too sparse to estimate (fewer than 5 events).
    for c in [c for c in out.columns if "=" in c]:
        if out.loc[out[c] == 1, "event"].sum() < 5:
            out = out.drop(columns=c)
    return out


# --- Models -----------------------------------------------------------------


def fit_cox(d: pd.DataFrame, entry: bool, strata: bool) -> tuple[CoxPHFitter, pd.DataFrame]:
    """Company enters either as strata (within-employer comparison) or as the
    cluster for robust SEs — never as a covariate. Returns the model and the
    exact frame it was fitted on (the PH test needs it)."""
    frame = d if entry else d.drop(columns=["entry"])
    kw = {"duration_col": "duration", "event_col": "event"}
    if entry:
        kw["entry_col"] = "entry"
    kw["strata" if strata else "cluster_col"] = ["company"] if strata else "company"
    cph = CoxPHFitter(penalizer=0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(frame, **kw)
    return cph, frame


def hr_table(cph: CoxPHFitter) -> pd.DataFrame:
    s = cph.summary
    return pd.DataFrame({
        "hr": s["exp(coef)"], "lo": s["exp(coef) lower 95%"],
        "hi": s["exp(coef) upper 95%"], "p": s["p"],
    }).round(3)


def km_by(df: pd.DataFrame, col: str, entry: bool = True) -> dict:
    out = {}
    for level, g in df.groupby(col):
        if g.event.sum() < 5:
            continue
        kmf = KaplanMeierFitter().fit(g.duration, g.event,
                                      entry=g.entry if entry else None, label=str(level))
        med = kmf.median_survival_time_
        out[str(level)] = {"n": len(g), "events": int(g.event.sum()),
                           "median_days": None if np.isinf(med) else float(med), "km": kmf}
    return out


def closure_age_spikes(spells: pd.DataFrame) -> pd.DataFrame:
    """Platform expiry shows up as closures piling on round ages (30/60/90 d)."""
    ages = spells.loc[spells.event == 1, "duration"]
    counts = ages.value_counts().sort_index()
    rows = []
    for a in (30, 45, 60, 90, 120, 180):
        window = counts.reindex(range(a - 2, a + 3), fill_value=0)
        neighbours = counts.reindex(list(range(a - 12, a - 2)) + list(range(a + 3, a + 13)),
                                    fill_value=0)
        rows.append({"age": a, "closures_at_age±2": int(window.sum()),
                     "expected_from_neighbours": round(float(neighbours.mean() * 5), 1)})
    return pd.DataFrame(rows)


def first_gap_spells(df: pd.DataFrame) -> pd.DataFrame:
    """Robustness: a posting's first disappearance is the event, even if it
    later came back (relisting is then a new posting we don't model)."""
    out = df.copy()
    gap = out.first_gap_on.notna()
    out.loc[gap, "event"] = 1
    out.loc[gap, "duration"] = [
        (g - p).days for g, p in zip(out.loc[gap, "first_gap_on"],
                                     out.loc[gap, "published_on"], strict=True)
    ]
    return out


def group_test(df: pd.DataFrame, col: str) -> dict:
    """Do the strata differ at all? Log-rank doesn't handle delayed entry, so
    use its Cox equivalent: likelihood-ratio test of a model with only this
    factor (delayed entry) against the empty model."""
    d = df[["duration", "event", "entry"]].copy()
    ref = REFERENCE.get(col)
    for v in sorted(df[col].unique()):
        if v != ref and df.loc[df[col] == v, "event"].sum() >= 5:
            d[f"{col}={v}"] = (df[col] == v).astype(float)
    cph = CoxPHFitter().fit(d, "duration", "event", entry_col="entry")
    t = cph.log_likelihood_ratio_test()
    return {"chi2": round(float(t.test_statistic), 2), "df": int(t.degrees_freedom),
            "p": round(float(t.p_value), 4)}


# --- Figures ----------------------------------------------------------------


def plot_km(groups: dict, path, title: str, order: list[str] | None = None) -> None:
    from analysis.a1_clustering import INK_2, _style

    plt = _style()
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    names = [n for n in (order or groups) if n in groups][:7]
    for color, name in zip(palette, names, strict=False):
        g = groups[name]
        kmf = g["km"]
        sf = kmf.survival_function_.iloc[:, 0]
        ci = kmf.confidence_interval_survival_function_
        ax.step(sf.index, sf.values, where="post", color=color, lw=2,
                label=f"{name} (n={g['n']}, events={g['events']})")
        if len(names) <= 3:  # more bands than that overlap into mush
            ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], step="post",
                            color=color, alpha=0.12, lw=0)
    ax.axhline(0.5, color="#c3c2b7", lw=1, ls=":")
    ax.set(xlim=(0, 240), ylim=(0, 1.02), xlabel="days since the employer published it",
           ylabel="share still open")
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK_2)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_hr(pooled: pd.DataFrame, within: pd.DataFrame, path, title: str) -> None:
    """Pooled (company-clustered SEs) vs within-company (company strata) hazard
    ratios. The gap between the two marks is the employer confound."""
    from matplotlib.ticker import FixedLocator, NullLocator

    from analysis.a1_clustering import INK_2, MUTED, SERIES_1, SURFACE, _style

    plt = _style()
    rows = [r for r in pooled.index if r in within.index][::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.8))
    y = np.arange(len(rows))
    ax.axvline(1, color=MUTED, lw=1)
    for t, off, color, name in ((pooled, 0.17, SERIES_1, "pooled, company-clustered SEs"),
                                (within, -0.17, "#eb6834", "within company (strata)")):
        t = t.loc[rows]
        ax.hlines(y + off, t.lo, t.hi, color=color, lw=2, alpha=0.35)
        ax.scatter(t.hr, y + off, s=45, color=color, edgecolor=SURFACE, linewidth=2,
                   zorder=3, label=name)
    ax.set_xscale("log")
    ticks = [0.25, 0.5, 1, 2, 4, 8]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_xlim(0.2, 10)
    ax.set_yticks(y, rows, color=INK_2, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("hazard ratio of closing, 95% CI (log scale; > 1 = closes faster)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --- Main -------------------------------------------------------------------


def main() -> int:
    postings, skills = load_postings()
    published, obs = load_lifecycle()
    spells, meta = build_spells(postings, published, obs)
    df = spells.merge(covariates(postings, skills), on="posting_key")

    print("=== A2: posting survival ===")
    print(f"  panel {meta['panel_start']} → {meta['panel_end']}; postings {len(df)},"
          f" closures {df.event.sum()}, censored {(df.event == 0).sum()}")
    print(f"  source deaths (censored, not events): {meta['source_deaths'] or 'none'}")
    print(f"  left-truncated (live on day 1): {df.left_truncated.sum()};"
          f" median age at entry {df.loc[df.left_truncated, 'entry'].median():.0f} d")

    naive_closed = df.loc[df.event == 1, "duration"].mean()
    naive_window = (df.loc[df.event == 1, "duration"] - df.loc[df.event == 1, "entry"]).mean()
    overall = KaplanMeierFitter().fit(df.duration, df.event, entry=df.entry)
    med = overall.median_survival_time_
    lo, hi = median_survival_times(overall.confidence_interval_).iloc[0].to_numpy()
    print("\n  median time open")
    print(f"    naive mean of closed postings (age at close): {naive_closed:.0f} d")
    print(f"    naive mean of closed postings (days we watched): {naive_window:.0f} d")
    print(f"    Kaplan–Meier with delayed entry: {med:.0f} d (95% CI {lo:.0f}–{hi:.0f})")

    spikes = closure_age_spikes(df)
    print("\n  closure-age spikes (platform expiry check):")
    print("    " + spikes.to_string(index=False).replace("\n", "\n    "))

    results = {"meta": meta, "n": len(df), "events": int(df.event.sum()),
               "km_median_days": float(med), "naive_mean_closed_age": float(naive_closed),
               "spikes": spikes.to_dict(orient="records"), "km": {}, "cox": {}}

    for col in ("family", "seniority"):
        groups = km_by(df, col)
        results["km"][col] = {k: {kk: vv for kk, vv in v.items() if kk != "km"}
                              for k, v in groups.items()}
        results["km"][f"{col}_test"] = test = group_test(df, col)
        print(f"\n  KM median by {col}: " + ", ".join(
            f"{k} {v['median_days'] or '>window':}"
            + (" d" if v["median_days"] else "") + f" (n={v['n']})"
            for k, v in sorted(groups.items(), key=lambda kv: kv[1]["median_days"] or 1e9)))
        print(f"    any difference (Cox LR test, delayed entry): chi2={test['chi2']}"
              f" df={test['df']} p={test['p']}")
        order = sorted(groups, key=lambda k: -groups[k]["n"])
        plot_km(groups, FIGURES / f"a2_km_by_{col}.png",
                f"A2 — time open by {col.replace('_', ' ')} (Kaplan–Meier, delayed entry)",
                order)

    specs = {
        "main": (df, True),
        "born": (df[(df.entry <= BORN_MAX_LAG_DAYS) & ~df.left_truncated]
                 .assign(duration=lambda x: x.duration - x.entry, entry=0), False),
        "first_gap": (first_gap_spells(df), True),
        # Closures bunch up at ~30 days (spike check above); if that's expiry
        # rather than filling, those aren't events.
        "no_30d": (df.assign(event=lambda x: np.where(
            x.duration.between(28, 32) & (x.event == 1), 0, x.event)), True),
    }
    tables = {}
    for name, (d, entry) in specs.items():
        X = design(d)
        for strata in (False, True):
            label = f"{name}{'_company_strata' if strata else ''}"
            cph, frame = fit_cox(X, entry=entry, strata=strata)
            tables[label] = hr_table(cph)
            results["cox"][label] = {
                "n": len(X), "events": int(X.event.sum()),
                "concordance": round(float(cph.concordance_index_), 3),
                "hr": tables[label].to_dict(orient="index"),
            }
            # lifelines can't compute Schoenfeld residuals with delayed entry,
            # so the PH check runs on the born cohort (no truncation).
            if name == "born":
                ph = proportional_hazard_test(cph, frame, time_transform="rank")
                viol = ph.summary[ph.summary.p < 0.05]
                results["cox"][label]["ph_violations"] = viol.round(4).to_dict(orient="index")
        print(f"\n  Cox [{name}] n={len(X)} events={int(X.event.sum())}")

    show = pd.concat({k: tables[k].hr for k in tables}, axis=1)
    print("\n  hazard ratios across specifications (>1 = closes faster):")
    print("    " + show.to_string().replace("\n", "\n    "))
    main_t = tables["main"]
    print("\n  main model with 95% CI (company-clustered SEs):")
    print("    " + main_t.to_string().replace("\n", "\n    "))
    for label in ("born", "born_company_strata"):
        v = results["cox"][label].get("ph_violations", {})
        print(f"  PH test [{label}] (scaled Schoenfeld, rank time, p<0.05):"
              f" {v or 'none'}")

    plot_hr(main_t, tables["main_company_strata"], FIGURES / "a2_cox_hazard_ratios.png",
            "A2 — what predicts closure (Cox with delayed entry)\n"
            "vs software_eng / senior / Bengaluru / hybrid")
    out = REPO / "report" / "a2_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  wrote {out.relative_to(REPO)} and figures in report/figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

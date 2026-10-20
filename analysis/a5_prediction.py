"""A5 — Can you predict, on day one, which postings close within 14 days?

The point of A5 is the evaluation, not the model:

  Temporal split   — train on postings first seen in panel weeks 1–8, test on
                     the ones first seen afterwards. A random split would let
                     the model learn from postings that appeared after the ones
                     it is scored on.
  Baseline ladder  — majority class → single best feature → logistic
                     regression → gradient boosting, with the lift of each
                     rung over the one below and a paired bootstrap CI on it.
  Metrics          — ~17% of postings close within 14 days, so PR AUC is
                     reported beside ROC AUC, plus the Brier score and a
                     calibration curve.
  Importance       — permutation importance on the test set, read as
                     association, not cause.
  Rolling origin   — secondary: refit at each week's end on everything before
                     it, predict the next week, pool the predictions. The
                     final test set has only ~10 positives; pooling the weekly
                     folds roughly triples that, at the cost of mixing models.

Sample: postings we saw being *born* (first seen after panel day 1, within
BORN_MAX_LAG_DAYS of the employer's publish date), so "day one" really is the
posting's first day. Label: closed within HORIZON_DAYS of first sighting.
A closure that A7 identifies as a repost is not a close (the role carried on
under a new ID), and a posting whose board disappeared inside the horizon has
no label and is dropped. A posting is labelled only once the panel has
watched it for the full horizon.

Spec frozen 2026-10-20, before any final-test posting existed (see
report/A5_prediction.md). Nothing below is tuned: with ~60 positives in the
final panel there is no data to tune on without fooling ourselves.

Run: python -m analysis.a5_prediction            # final split (needs panel end)
     python -m analysis.a5_prediction --interim  # pipeline check on current data
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis.common import FIGURES, REPO, load_lifecycle, load_postings

HORIZON_DAYS = 14
BORN_MAX_LAG_DAYS = 1
MIN_COMPANY_ROWS = 5       # companies with fewer training postings pool into "other"
N_BOOT = 2000
SEED = 20261020

# Splits are fixed dates, not fractions, so the final test set is defined
# before it exists. "final" is the plan's weeks 1–8 vs 9–12; postings from the
# last two weeks can't be labelled (no 14 days of follow-up), so in practice
# the test set is weeks 9–10.
SPLITS = {
    "final": {"train_end": date(2026, 11, 1), "needs_panel_end": date(2026, 11, 30)},
    # Weeks 1–3 vs whatever is labellable after: exercises the pipeline only.
    "interim": {"train_end": date(2026, 9, 27), "needs_panel_end": None},
}

CATEGORICAL = ["family", "seniority", "city", "workplace", "company"]
NUMERIC = ["skills_per_5", "log_desc_words", "log_company_open"]
BINARY = ["relisted", "is_multi_country"]
FEATURES = CATEGORICAL + NUMERIC + BINARY


# --- Sample and label -------------------------------------------------------


def label_born(spells: pd.DataFrame, panel_start, panel_end,
               repost_keys: set[str]) -> pd.DataFrame:
    """Born postings with a known 14-day outcome, and y = closed within it."""
    d = spells[(spells.first_seen > panel_start)
               & (spells.entry <= BORN_MAX_LAG_DAYS)].copy()
    d = d[d.first_seen + timedelta(days=HORIZON_DAYS) <= panel_end]
    days_open = d.duration - d.entry  # closure (or last sighting + 1) minus first sighting
    within = days_open <= HORIZON_DAYS
    # Board vanished inside the horizon: we can't know whether it would have closed.
    d = d[~(d.censored_by_source_death & within)]
    within = within.loc[d.index]
    real_close = (d.event == 1) & ~d.posting_key.isin(repost_keys)
    d["y"] = (real_close & within).astype(int)
    return d


def day_one_features(sample: pd.DataFrame, postings: pd.DataFrame, skills: pd.DataFrame,
                     obs: pd.DataFrame, texts: pd.DataFrame) -> pd.DataFrame:
    """Everything here is knowable on the posting's first day."""
    from analysis import a2_survival as a2

    cov = a2.covariates(postings, skills)
    out = sample[["posting_key", "company", "first_seen", "y"]].merge(cov, on="posting_key")
    out = out.merge(postings[["posting_key", "is_multi_country"]], on="posting_key")
    out["is_multi_country"] = out.is_multi_country.fillna(False).astype(int)

    # Same employer, same normalised title, seen on an earlier day.
    t = texts.set_index("posting_key")
    first = postings.set_index("posting_key").first_seen
    seen_before = {}
    for key in out.posting_key:
        co, title, fs = t.at[key, "company"], t.at[key, "title_norm"], first[key]
        prior = t[(t.company == co) & (t.title_norm == title)].index
        seen_before[key] = int(any(first[k] < fs for k in prior if k != key))
    out["relisted"] = out.posting_key.map(seen_before)

    # How many postings the employer had live that day: a hiring-volume proxy.
    o = obs.merge(postings[["posting_key", "company"]], on="posting_key")
    live = o.groupby(["company", "observed_on"]).size()
    out["log_company_open"] = np.log1p([live.get((c, d), 0) for c, d in
                                        zip(out.company, out.first_seen, strict=True)])
    out["log_desc_words"] = np.log1p(out.posting_key.map(t.n_words))
    return out


def split(df: pd.DataFrame, train_end) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = df[df.first_seen <= train_end].copy(), df[df.first_seen > train_end].copy()
    # Company levels come from training only; an employer new in the test
    # period is "other", as it would be on the day.
    keep = train.company.value_counts()
    keep = set(keep[keep >= MIN_COMPANY_ROWS].index)
    for d in (train, test):
        d["company"] = d.company.where(d.company.isin(keep), "other")
    return train, test


# --- Models -----------------------------------------------------------------


def _encoder(cats: list[str], nums: list[str]) -> ColumnTransformer:
    parts = []
    if cats:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), cats))
    if nums:
        parts.append(("num", StandardScaler(), nums))
    return ColumnTransformer(parts)


def logistic(features: list[str]) -> Pipeline:
    cats = [f for f in features if f in CATEGORICAL]
    nums = [f for f in features if f not in CATEGORICAL]
    return Pipeline([("enc", _encoder(cats, nums)),
                     ("clf", LogisticRegression(C=1.0, max_iter=2000))])


def boosting() -> Pipeline:
    enc = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("rest", "passthrough", NUMERIC + BINARY)])
    # Shallow and slow: on a few hundred rows anything deeper memorises.
    clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                         min_samples_leaf=10, random_state=SEED)
    return Pipeline([("enc", enc), ("clf", clf)])


def best_single_feature(train: pd.DataFrame) -> tuple[str, dict]:
    """Chosen by 5-fold CV ROC AUC on the training set only."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = {}
    for f in FEATURES:
        p = cross_val_predict(logistic([f]), train[[f]], train.y, cv=cv,
                              method="predict_proba")[:, 1]
        scores[f] = round(float(roc_auc_score(train.y, p)), 3)
    return max(scores, key=scores.get), scores


def fit_ladder(train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, dict, str, dict]:
    """-> (test probabilities per rung, fitted models, chosen feature, its CV scores)."""
    feature, cv_scores = best_single_feature(train)
    rungs = {
        "logistic": logistic(FEATURES),
        "boosting": boosting(),
        # Diagnostic, not a rung: how much of the signal is just the employer?
        "logistic_no_company": logistic([f for f in FEATURES if f != "company"]),
    }
    probs = {"majority": np.full(len(test), train.y.mean())}
    single = logistic([feature]).fit(train[[feature]], train.y)
    probs["single_feature"] = single.predict_proba(test[[feature]])[:, 1]
    models = {"single_feature": single}
    for name, model in rungs.items():
        cols = FEATURES if name != "logistic_no_company" else [f for f in FEATURES
                                                                if f != "company"]
        models[name] = clone(model).fit(train[cols], train.y)
        probs[name] = models[name].predict_proba(test[cols])[:, 1]
    return probs, models, feature, cv_scores


# --- Evaluation -------------------------------------------------------------


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    if len(np.unique(p)) == 1:  # constant score: AUC is 0.5, AP is the prevalence
        roc, ap = 0.5, float(y.mean())
    else:
        roc, ap = float(roc_auc_score(y, p)), float(average_precision_score(y, p))
    return {"roc_auc": roc, "pr_auc": ap, "brier": float(brier_score_loss(y, p))}


def bootstrap(y: np.ndarray, probs: dict, pairs: list[tuple[str, str]],
              n: int = N_BOOT, seed: int = SEED) -> dict:
    """Percentile CIs per model, and paired CIs on each rung's lift.

    Resamples are stratified by outcome so each keeps the test set's positives;
    with ~10 positives an unstratified resample can contain one or none.
    """
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    draws = {k: [] for k in probs}
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for k, p in probs.items():
            draws[k].append(metrics(y[idx], p[idx]))
    out = {"models": {}, "lift": {}}
    for k, ds in draws.items():
        frame = pd.DataFrame(ds)
        out["models"][k] = {m: [round(float(frame[m].quantile(q)), 3) for q in (0.025, 0.975)]
                            for m in frame}
    for hi, lo in pairs:
        d = pd.DataFrame(draws[hi]) - pd.DataFrame(draws[lo])
        out["lift"][f"{hi} - {lo}"] = {
            m: [round(float(d[m].quantile(q)), 3) for q in (0.025, 0.975)] for m in d}
    return out


def calibration(y: np.ndarray, p: np.ndarray, bins: int) -> dict:
    """Quantile-binned reliability, plus calibration-in-the-large and slope."""
    order = np.argsort(p)
    groups = np.array_split(order, bins)
    rel = [{"mean_pred": round(float(p[g].mean()), 3), "observed": round(float(y[g].mean()), 3),
            "n": len(g)} for g in groups if len(g)]
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    slope = LogisticRegression(C=1e6).fit(logit[:, None], y).coef_[0, 0]
    return {"bins": rel, "observed_rate": round(float(y.mean()), 3),
            "mean_predicted": round(float(p.mean()), 3), "slope": round(float(slope), 2)}


def importance(model, test: pd.DataFrame, cols: list[str]) -> dict:
    r = permutation_importance(model, test[cols], test.y, scoring="roc_auc",
                               n_repeats=50, random_state=SEED)
    return {c: {"mean": round(float(m), 3), "sd": round(float(s), 3)}
            for c, m, s in sorted(zip(cols, r.importances_mean, r.importances_std,
                                      strict=True), key=lambda x: -x[1])}


def plot_calibration(cal: dict, path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="perfectly calibrated")
    for name, c in cal.items():
        b = c["bins"]
        ax.plot([x["mean_pred"] for x in b], [x["observed"] for x in b], marker="o",
                label=f"{name} (slope {c['slope']})")
    top = max(0.6, max(x["observed"] for c in cal.values() for x in c["bins"]) + 0.05)
    ax.set(xlim=(0, top), ylim=(0, top), xlabel="mean predicted probability (bin)",
           ylabel="observed share closed within 14 d", title=title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def rolling_origin(df: pd.DataFrame, first_cutoff) -> tuple[pd.DataFrame, dict]:
    """Weekly expanding-window folds -> pooled out-of-time predictions."""
    preds, folds = [], []
    cutoff = first_cutoff
    while (df.first_seen > cutoff).any():
        nxt = cutoff + timedelta(days=7)
        train, test = split(df[df.first_seen <= nxt], cutoff)
        if len(test) and train.y.nunique() == 2:
            probs, *_ = fit_ladder(train, test)
            preds.append(test[["posting_key", "y"]].assign(**probs, fold=str(nxt)))
            folds.append({"train_end": str(cutoff), "test_end": str(nxt), "n": len(test),
                          "pos": int(test.y.sum())})
        cutoff = nxt
    return pd.concat(preds, ignore_index=True), folds


# --- Main -------------------------------------------------------------------


def load_sample() -> tuple[pd.DataFrame, dict]:
    from analysis import a2_survival as a2
    from analysis import a7_duplication as a7
    from extraction.text import clean_text

    postings, skills = load_postings()
    published, obs = load_lifecycle()
    spells, meta = a2.build_spells(postings, published, obs)

    texts = a7.load_texts()
    S, same = a7.similarity(texts)
    pairs = a7.classify_pairs(texts, S, same, a7.THRESHOLD)
    repost_keys = set(pairs.loc[pairs.kind == "repost", "earlier"])
    # Raw length, not boilerplate-stripped: the boilerplate is learned from the
    # whole corpus, future postings included.
    texts["n_words"] = [len(clean_text(d).split()) for d in texts.description_raw]

    start, end = (pd.Timestamp(meta[k]).date() for k in ("panel_start", "panel_end"))
    sample = label_born(spells, start, end, repost_keys)
    meta.update({"born_labelled": len(sample), "repost_closures": len(repost_keys)})
    return day_one_features(sample, postings, skills, obs, texts), meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interim", action="store_true", help="pipeline check on current data")
    args = ap.parse_args(argv)
    name = "interim" if args.interim else "final"
    cfg = SPLITS[name]

    df, meta = load_sample()
    panel_end = pd.Timestamp(meta["panel_end"]).date()
    if cfg["needs_panel_end"] and panel_end < cfg["needs_panel_end"]:
        print(f"final split needs the panel through {cfg['needs_panel_end']}"
              f" (have {panel_end}); use --interim for a pipeline check")
        return 1
    train, test = split(df, cfg["train_end"])
    y = test.y.to_numpy()

    print(f"=== A5: 14-day closure prediction [{name} split] ===")
    print(f"  panel {meta['panel_start']} → {panel_end}; born + labelled {len(df)}"
          f" ({df.y.sum()} closed within {HORIZON_DAYS} d, {df.y.mean():.0%})")
    print(f"  train ≤ {cfg['train_end']}: n={len(train)} pos={train.y.sum()};"
          f" test > {cfg['train_end']}: n={len(test)} pos={int(y.sum())}")
    if y.sum() < 2 or (y == 0).sum() < 2:
        print("  test set too small to score")
        return 1

    probs, models, feature, cv_scores = fit_ladder(train, test)
    ladder = ["majority", "single_feature", "logistic", "boosting"]
    pairs = list(zip(ladder[1:], ladder[:-1], strict=True)) + [
        ("logistic", "logistic_no_company")]
    point = {k: {m: round(v, 3) for m, v in metrics(y, p).items()} for k, p in probs.items()}
    boot = bootstrap(y, probs, pairs)

    print(f"\n  single best feature (train CV ROC AUC): {feature}"
          f" — all: {dict(sorted(cv_scores.items(), key=lambda kv: -kv[1]))}")
    print("\n  model           ROC AUC (95% CI)      PR AUC (95% CI)       Brier")
    for k in [*ladder, "logistic_no_company"]:
        pt, ci = point[k], boot["models"][k]
        print(f"  {k:<20} {pt['roc_auc']:.3f} ({ci['roc_auc'][0]:.2f}–{ci['roc_auc'][1]:.2f})"
              f"   {pt['pr_auc']:.3f} ({ci['pr_auc'][0]:.2f}–{ci['pr_auc'][1]:.2f})"
              f"   {pt['brier']:.3f}")
    print("\n  lift (point, paired bootstrap 95% CI):")
    for (hi, lo) in pairs:
        ci = boot["lift"][f"{hi} - {lo}"]
        print(f"    {hi} over {lo}: ROC AUC {point[hi]['roc_auc'] - point[lo]['roc_auc']:+.3f}"
              f" ({ci['roc_auc'][0]:+.2f} to {ci['roc_auc'][1]:+.2f}),"
              f" PR AUC {point[hi]['pr_auc'] - point[lo]['pr_auc']:+.3f}"
              f" ({ci['pr_auc'][0]:+.2f} to {ci['pr_auc'][1]:+.2f})")

    bins = 5 if name == "final" else 3
    cal = {k: calibration(y, probs[k], bins) for k in ("logistic", "boosting")}
    for k, c in cal.items():
        print(f"\n  calibration [{k}]: observed {c['observed_rate']}, mean predicted"
              f" {c['mean_predicted']}, slope {c['slope']} (1 = ideal)")
    imp = {"logistic": importance(models["logistic"], test, FEATURES),
           "boosting": importance(models["boosting"], test, FEATURES)}
    print("\n  permutation importance on test (drop in ROC AUC, mean ± sd over 50):")
    for k, v in imp.items():
        print(f"    {k}: " + ", ".join(f"{f} {x['mean']:+.3f}±{x['sd']:.3f}"
                                       for f, x in list(v.items())[:5]))

    ro, folds = rolling_origin(df, SPLITS["interim"]["train_end"])
    ry = ro.y.to_numpy()
    rprobs = {k: ro[k].to_numpy() for k in probs}
    # Each fold's majority rung is a different constant, so pooled it would
    # rank postings by fold. One constant keeps it the no-information baseline.
    rprobs["majority"] = np.full(len(ro), rprobs["majority"].mean())
    rpoint = {k: {m: round(v, 3) for m, v in metrics(ry, p).items()} for k, p in rprobs.items()}
    rboot = bootstrap(ry, rprobs, pairs)
    print(f"\n  rolling origin: {len(folds)} weekly folds, n={len(ro)} pos={int(ry.sum())}")
    for k in [*ladder, "logistic_no_company"]:
        pt, ci = rpoint[k], rboot["models"][k]
        print(f"    {k:<20} ROC AUC {pt['roc_auc']:.3f}"
              f" ({ci['roc_auc'][0]:.2f}–{ci['roc_auc'][1]:.2f})"
              f"   PR AUC {pt['pr_auc']:.3f} ({ci['pr_auc'][0]:.2f}–{ci['pr_auc'][1]:.2f})")

    suffix = "" if name == "final" else "_interim"
    plot_calibration(cal, FIGURES / f"a5_calibration{suffix}.png",
                     f"A5 — calibration on the test set ({name} split)")
    results = {
        "meta": {**meta, "split": name, "train_end": str(cfg["train_end"]),
                 "horizon_days": HORIZON_DAYS},
        "n": {"train": len(train), "train_pos": int(train.y.sum()),
              "test": len(test), "test_pos": int(y.sum())},
        "single_feature": {"chosen": feature, "train_cv_roc_auc": cv_scores},
        "test": point, "bootstrap": boot, "calibration": cal, "importance": imp,
        "rolling_origin": {"folds": folds, "n": len(ro), "pos": int(ry.sum()),
                           "metrics": rpoint, "bootstrap": rboot},
    }
    out = REPO / "report" / f"a5_results{suffix}.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  wrote {out.relative_to(REPO)} and report/figures/a5_calibration{suffix}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())

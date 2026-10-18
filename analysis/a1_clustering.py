"""A1 — Does one job title hide several distinct roles?

If postings' skill requirements cluster into distinguishable groups that cut
across title families, the title label is hiding structure. k-means always
returns clusters, even on noise, so the work here is showing whether any
structure is real:

  1. Choose k honestly — silhouette and gap curves over k, each against a
     null distribution, reported as curves rather than one picked number.
  2. Test stability — bootstrap the postings, re-cluster, measure agreement
     (ARI overall, Jaccard per cluster so we know *which* clusters hold).
  3. Test against a null — "curveball" randomisation of the posting x skill
     matrix, which preserves both how common each skill is and how many
     skills each posting lists, while destroying which skills go together.
     Shuffling columns alone would keep only the first, and skills-per-
     posting drives cosine similarity, so it would flatter the real data.
     The gap statistic uses the same null as its reference distribution
     (a uniform reference is meaningless for binary data).

Then the failure mode the plan names: clusters that track something other
than role. Here the dominant risk is *employer* (Okta is half the in-scope
sample), so clusters are cross-tabulated against company as well as
seniority and title family, and stability is re-tested with a per-company
cap so no employer's volume can carry a cluster.

Run: python -m analysis.a1_clustering [--family software_engineering]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize

from analysis.common import FIGURES, REPO, load_postings

SEED = 20261018
K_RANGE = range(2, 11)
N_NULL = 50
N_BOOT = 100
MIN_DF = 5          # a skill must appear in >= this many postings to be a feature
COMPANY_CAP = 15    # per-company cap for the employer-robustness resampling
NPMI_MIN_CO = 5     # pairs must co-occur in >= this many postings


# --- Matrix -----------------------------------------------------------------


@dataclass
class Matrix:
    X: np.ndarray            # bool, postings x skills
    rows: pd.DataFrame       # posting metadata aligned with X
    skills: list[str]
    n_before_dedupe: int


def build_matrix(postings: pd.DataFrame, skills: pd.DataFrame, family: str | None) -> Matrix:
    p = postings[postings.in_scope]
    if family:
        p = p[p.title_family == family]
    sk = skills[skills.posting_key.isin(p.posting_key)]
    sets = sk.groupby("posting_key").skill.apply(lambda s: tuple(sorted(s)))
    p = p.assign(skillset=p.posting_key.map(sets)).dropna(subset=["skillset"])
    n_before = len(p)
    # The same role reposted per city has an identical (company, skill set);
    # left in, those copies form artificially tight clusters.
    p = p.drop_duplicates(["company", "skillset"]).reset_index(drop=True)

    df = pd.Series([s for ss in p.skillset for s in ss]).value_counts()
    vocab = sorted(df[df >= MIN_DF].index)
    col = {s: j for j, s in enumerate(vocab)}
    X = np.zeros((len(p), len(vocab)), dtype=bool)
    for i, ss in enumerate(p.skillset):
        for s in ss:
            if s in col:
                X[i, col[s]] = True
    keep = X.any(axis=1)
    return Matrix(X[keep], p[keep].reset_index(drop=True), vocab, n_before)


def embed(X: np.ndarray) -> np.ndarray:
    """TF-IDF on the binary matrix, L2-normalised: Euclidean k-means on these
    rows is spherical (cosine) k-means."""
    Z = TfidfTransformer(smooth_idf=True).fit_transform(X.astype(float)).toarray()
    return normalize(Z)


def cluster(Z: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, float]:
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
    return km.labels_, km.inertia_


# --- Null model -------------------------------------------------------------


def curveball(X: np.ndarray, rng: np.random.Generator, trades: int | None = None) -> np.ndarray:
    """Random binary matrix with the same row and column sums (Strona et al. 2014)."""
    rows = [set(np.flatnonzero(r)) for r in X]
    n = len(rows)
    for _ in range(trades or 5 * n):
        a, b = rng.choice(n, size=2, replace=False)
        only_a, only_b = rows[a] - rows[b], rows[b] - rows[a]
        if not only_a or not only_b:
            continue
        pool = list(only_a | only_b)
        rng.shuffle(pool)
        shared = rows[a] & rows[b]
        rows[a] = shared | set(pool[:len(only_a)])
        rows[b] = shared | set(pool[len(only_a):])
    out = np.zeros_like(X)
    for i, r in enumerate(rows):
        out[i, list(r)] = True
    return out


def k_curves(X: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    Z = embed(X)
    nulls = [embed(curveball(X, rng)) for _ in range(N_NULL)]
    out = []
    for k in K_RANGE:
        labels, inertia = cluster(Z, k, SEED)
        sil = silhouette_score(Z, labels, metric="cosine")
        null_sil, null_logw = [], []
        for j, Zn in enumerate(nulls):
            ln, wn = cluster(Zn, k, SEED + j)
            null_sil.append(silhouette_score(Zn, ln, metric="cosine"))
            null_logw.append(math.log(wn))
        null_logw = np.array(null_logw)
        out.append({
            "k": k,
            "silhouette": sil,
            "null_sil_mean": float(np.mean(null_sil)),
            "null_sil_sd": float(np.std(null_sil, ddof=1)),
            "null_sil_p": float((np.sum(np.array(null_sil) >= sil) + 1) / (N_NULL + 1)),
            "gap": float(null_logw.mean() - math.log(inertia)),
            # Tibshirani's s_k: sd of the reference log-dispersion, inflated
            # for simulation error.
            "gap_s": float(null_logw.std(ddof=1) * math.sqrt(1 + 1 / N_NULL)),
        })
    curves = pd.DataFrame(out)
    curves["sil_excess_z"] = (curves.silhouette - curves.null_sil_mean) / curves.null_sil_sd
    return curves


def gap_rule_k(curves: pd.DataFrame) -> int | None:
    """Tibshirani's rule: smallest k with gap(k) >= gap(k+1) - s(k+1).
    None if the gap is still rising at the end of the range — i.e. the data
    has structure at every k but no natural number of clusters."""
    c = curves.set_index("k")
    for k in c.index[:-1]:
        if c.gap[k] >= c.gap[k + 1] - c.gap_s[k + 1]:
            return int(k)
    return None


def stability_curve(Z: np.ndarray, rows: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Bootstrap ARI per k. Used to choose k when the gap has no stopping point:
    the k whose partition survives resampling is the one worth naming."""
    out = []
    for k in K_RANGE:
        ref, _ = cluster(Z, k, SEED)
        s = stability(Z, rows, k, ref, rng, "bootstrap", n_boot=N_BOOT // 2)
        out.append({"k": k, "ari_mean": s["ari_mean"], "ari_lo": s["ari_ci"][0],
                    "ari_hi": s["ari_ci"][1],
                    "n_stable_clusters": int(sum(j >= 0.75 for j in s["jaccard_by_cluster"]))})
    return pd.DataFrame(out)


# --- Stability --------------------------------------------------------------


def _jaccard_per_cluster(ref: np.ndarray, new: np.ndarray, k: int) -> np.ndarray:
    """Hennig's clusterwise stability: best Jaccard match of each reference cluster."""
    out = np.full(k, np.nan)
    for c in range(k):
        a = ref == c
        if not a.any():
            continue
        out[c] = max(
            (a & (new == d)).sum() / (a | (new == d)).sum() for d in np.unique(new)
        )
    return out


def stability(Z: np.ndarray, rows: pd.DataFrame, k: int, ref: np.ndarray,
              rng: np.random.Generator, scheme: str, n_boot: int = N_BOOT) -> dict:
    aris, jacs = [], []
    by_company = rows.groupby("company").indices
    for b in range(n_boot):
        if scheme == "bootstrap":
            idx = rng.choice(len(Z), size=len(Z), replace=True)
        else:  # company_cap: at most COMPANY_CAP postings per employer
            idx = np.concatenate([
                rng.choice(ix, size=min(len(ix), COMPANY_CAP), replace=False)
                for ix in by_company.values()
            ])
        if len(np.unique(idx)) <= k:
            continue
        labels, _ = cluster(Z[idx], k, SEED + b)
        # Score on distinct sampled postings (bootstrap repeats carry no info).
        uniq, first = np.unique(idx, return_index=True)
        aris.append(adjusted_rand_score(ref[uniq], labels[first]))
        jacs.append(_jaccard_per_cluster(ref[uniq], labels[first], k))
    jacs = np.array(jacs)
    return {
        "scheme": scheme,
        "n": len(aris),
        "ari_mean": float(np.mean(aris)),
        "ari_ci": [float(np.percentile(aris, 2.5)), float(np.percentile(aris, 97.5))],
        "jaccard_by_cluster": [float(x) for x in np.nanmean(jacs, axis=0)],
    }


# --- Description & confounds ------------------------------------------------


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    t = pd.crosstab(a, b).to_numpy().astype(float)
    n = t.sum()
    expected = t.sum(1, keepdims=True) @ t.sum(0, keepdims=True) / n
    chi2 = ((t - expected) ** 2 / expected).sum()
    r, c = t.shape
    return float(math.sqrt(chi2 / (n * (min(r, c) - 1)))) if min(r, c) > 1 else 0.0


def describe(m: Matrix, labels: np.ndarray, top: int = 8) -> list[dict]:
    overall = m.X.mean(axis=0)
    out = []
    for c in range(labels.max() + 1):
        mask = labels == c
        prev = m.X[mask].mean(axis=0)
        lift = np.divide(prev, overall, out=np.zeros_like(prev), where=overall > 0)
        # Signature: skills common in the cluster AND over-represented in it.
        order = np.argsort(-(lift * (prev >= 0.3)))
        sub = m.rows[mask]
        out.append({
            "cluster": c,
            "n": int(mask.sum()),
            "signature": [
                {"skill": m.skills[j], "prevalence": round(float(prev[j]), 2),
                 "lift": round(float(lift[j]), 2)}
                for j in order[:top] if prev[j] >= 0.3
            ],
            "top_company_share":
                sub.company.value_counts(normalize=True).round(2).head(3).to_dict(),
            "title_family": sub.title_family.value_counts().to_dict(),
            "seniority": sub.seniority.value_counts().to_dict(),
            "example_titles": sub.title_raw.drop_duplicates().head(6).tolist(),
        })
    return out


def npmi_pairs(m: Matrix) -> pd.DataFrame:
    """Normalised PMI for skill pairs: association beyond what each skill's
    own frequency predicts. Raw co-occurrence would just rediscover that
    python and aws are common."""
    X = m.X.astype(float)
    n = len(X)
    p = X.mean(axis=0)
    co = X.T @ X
    rows = []
    for i, j in itertools.combinations(range(len(m.skills)), 2):
        if co[i, j] < NPMI_MIN_CO:
            continue
        pij = co[i, j] / n
        pmi = math.log(pij / (p[i] * p[j]))
        rows.append({"a": m.skills[i], "b": m.skills[j], "co": int(co[i, j]),
                     "npmi": pmi / -math.log(pij)})
    return pd.DataFrame(rows).sort_values("npmi", ascending=False).reset_index(drop=True)


# --- Figures ----------------------------------------------------------------

INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SERIES_1 = "#2a78d6"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
    })
    return plt


def plot_curves(curves: pd.DataFrame, k: int, path, title: str) -> None:
    plt = _style()
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 3.8))
    ks = curves.k
    lo = curves.null_sil_mean - 2 * curves.null_sil_sd
    hi = curves.null_sil_mean + 2 * curves.null_sil_sd
    a1.fill_between(ks, lo, hi, color=GRID, label="null ±2 sd (curveball)")
    a1.plot(ks, curves.null_sil_mean, color=MUTED, lw=1.5, ls="--", label="null mean")
    a1.plot(ks, curves.silhouette, color=SERIES_1, lw=2, marker="o", ms=6, label="observed")
    a1.set(title="Silhouette (cosine) vs null", xlabel="k", ylabel="silhouette")
    a1.legend(frameon=False, fontsize=8)

    a2.errorbar(ks, curves.gap, yerr=curves.gap_s, color=SERIES_1, lw=2, marker="o", ms=6,
                capsize=3, label="gap ± s_k")
    a2.set(title="Gap statistic (curveball reference)", xlabel="k", ylabel="gap")
    a2.legend(frameon=False, fontsize=8)

    a3.fill_between(ks, curves.ari_lo, curves.ari_hi, color=GRID, label="bootstrap 95%")
    a3.plot(ks, curves.ari_mean, color=SERIES_1, lw=2, marker="o", ms=6, label="mean ARI")
    a3.axvline(k, color=MUTED, lw=1, ls=":")
    a3.annotate(f"chosen k = {k}", (k, curves.ari_hi.max()), xytext=(6, -10),
                textcoords="offset points", color=INK_2, fontsize=8)
    a3.set(title="Stability across bootstraps", xlabel="k", ylabel="adjusted Rand index",
           ylim=(0, 1))
    a3.legend(frameon=False, fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_composition(m: Matrix, labels: np.ndarray, path, title: str) -> None:
    """Cluster x company share heatmap — the employer-confound check, visually."""
    from matplotlib.colors import LinearSegmentedColormap

    plt = _style()
    rows = m.rows.assign(cluster=labels)
    top = rows.company.value_counts().index[:6]
    rows["employer"] = rows.company.where(rows.company.isin(top), "Other")
    t = pd.crosstab(rows.cluster, rows.employer, normalize="index")
    t = t[[c for c in list(top) + ["Other"] if c in t.columns]]
    fig, ax = plt.subplots(figsize=(1.1 * t.shape[1] + 2, 0.55 * t.shape[0] + 1.4))
    cmap = LinearSegmentedColormap.from_list("seq", ["#f0efec"] + SEQ)
    ax.imshow(t.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(t.shape[1]), t.columns, rotation=30, ha="right", color=INK_2)
    sizes = rows.cluster.value_counts().sort_index()
    ax.set_yticks(range(t.shape[0]), [f"cluster {c} (n={sizes[c]})" for c in t.index],
                  color=INK_2)
    for i, j in itertools.product(range(t.shape[0]), range(t.shape[1])):
        v = t.iat[i, j]
        if v >= 0.05:
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                    color="#fcfcfb" if v > 0.5 else INK)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pairs(pairs: pd.DataFrame, path, title: str, top: int = 25) -> None:
    """Top skill pairs by NPMI as a ranked dot chart. The associations are
    mostly isolated pairs plus one data-stack clique, so a network layout
    either sprawls or stacks labels; a ranked list reads cleanly."""
    plt = _style()
    e = pairs.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(e) + 1.4))
    y = np.arange(len(e))
    ax.hlines(y, 0, e.npmi, color=GRID, lw=2)
    ax.scatter(e.npmi, y, s=60, color=SERIES_1, edgecolor=SURFACE, linewidth=2, zorder=3)
    for yi, r in zip(y, e.itertuples(), strict=True):
        ax.annotate(f"{r.co}", (r.npmi, yi), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8, color=MUTED)
    ax.set_yticks(y, [f"{r.a} + {r.b}" for r in e.itertuples()], color=INK_2, fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("normalised PMI (0 = independent, 1 = always together)")
    ax.set_title(title + f"\ntop {len(e)} pairs co-occurring in ≥ {NPMI_MIN_CO} postings;"
                 " grey number = postings with both", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --- Main -------------------------------------------------------------------


def run(family: str | None) -> dict:
    rng = np.random.default_rng(SEED)
    postings, skills = load_postings()
    m = build_matrix(postings, skills, family)
    tag = family or "all_in_scope"
    print(f"=== A1: {tag} ===")
    print(f"  postings {m.n_before_dedupe} -> {len(m.rows)} after merging same-company"
          f" duplicate skill sets; {len(m.skills)} skill features (df >= {MIN_DF})")

    Z = embed(m.X)
    curves = k_curves(m.X, rng).merge(stability_curve(Z, m.rows, rng), on="k")
    gap_k = gap_rule_k(curves)
    k = int(curves.loc[curves.ari_mean.idxmax(), "k"])
    print("\n  k  silhouette  null mean±sd      z     p   gap ± s          ARI (95%)       stable")
    for r in curves.itertuples():
        print(f"  {r.k:<2} {r.silhouette:9.3f}  {r.null_sil_mean:.3f}±{r.null_sil_sd:.3f}"
              f"  {r.sil_excess_z:5.1f}  {r.null_sil_p:.2f}  {r.gap:.3f} ± {r.gap_s:.3f}"
              f"   {r.ari_mean:.2f} ({r.ari_lo:.2f}–{r.ari_hi:.2f})  {r.n_stable_clusters}/{r.k}")
    print(f"  gap rule: {gap_k or 'no stopping point in range (gap still rising)'};"
          f" most stable k = {k}")

    labels, _ = cluster(Z, k, SEED)
    stab = [stability(Z, m.rows, k, labels, rng, s) for s in ("bootstrap", "company_cap")]
    for s in stab:
        jac = ", ".join(f"{x:.2f}" for x in s["jaccard_by_cluster"])
        print(f"  stability [{s['scheme']}] ARI {s['ari_mean']:.2f}"
              f" (95% {s['ari_ci'][0]:.2f}–{s['ari_ci'][1]:.2f}); Jaccard by cluster: {jac}")

    lab = pd.Series(labels, name="cluster")
    confounds = {c: cramers_v(lab, m.rows[c]) for c in ("company", "seniority", "title_family")}
    print("  Cramér's V vs cluster: " + ", ".join(f"{c} {v:.2f}" for c, v in confounds.items()))

    clusters = describe(m, labels)
    for c in clusters:
        sig = ", ".join(f"{s['skill']} ({s['prevalence']:.0%}, ×{s['lift']})"
                        for s in c["signature"][:6])
        print(f"\n  cluster {c['cluster']} n={c['n']}  companies {c['top_company_share']}")
        print(f"    signature: {sig}")
        print(f"    families {c['title_family']}  seniority {c['seniority']}")
        print(f"    e.g. {c['example_titles'][:4]}")

    pairs = npmi_pairs(m)
    print("\n  top NPMI pairs: " + ", ".join(
        f"{r.a}+{r.b} {r.npmi:.2f}" for r in pairs.head(12).itertuples()))

    FIGURES.mkdir(parents=True, exist_ok=True)
    label = family.replace("_", " ") if family else "technical IC postings"
    plot_curves(curves, k, FIGURES / f"a1_{tag}_k_curves.png",
                f"A1 — choosing k ({label}, n={len(m.rows)})")
    plot_composition(m, labels, FIGURES / f"a1_{tag}_cluster_by_company.png",
                     f"Cluster membership by employer (row share), k={k}")
    plot_pairs(pairs, FIGURES / f"a1_{tag}_skill_pairs.png",
               f"Skill associations — {label}")

    results = {
        "tag": tag, "seed": SEED, "n_before_dedupe": m.n_before_dedupe,
        "n": len(m.rows), "n_features": len(m.skills), "k": k, "gap_rule_k": gap_k,
        "curves": curves.round(4).to_dict(orient="records"),
        "stability": stab, "cramers_v": confounds, "clusters": clusters,
        "top_npmi": pairs.head(30).round(3).to_dict(orient="records"),
    }
    out = REPO / "report" / f"a1_{tag}_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  wrote {out.relative_to(REPO)} and figures in report/figures/")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="A1 — skill clustering")
    parser.add_argument("--family", help="restrict to one title_family")
    args = parser.parse_args()
    run(args.family)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# A3 — Pre-registration: has the skill mix shifted?

**Registered 2026-10-18 (panel day 42), before any period-2 posting exists.** The machine-readable
version, `data/a3_preregistration.yaml`, is what `analysis/a3_drift.py` executes. Both files, and the
code, were committed together. The code refuses to compare the real periods before 2026-12-01.

## Why pre-register

The plan names the failure mode: plot forty skills, notice three rising, and tell a story about
them. With 27 tests at the 5% level you expect about one false positive before looking at anything.
Fixing the periods, the population, the skill list and the tests before the data exists removes the
freedom to choose them after seeing which choice gives a result.

## Design

| | Pre-registered choice | Why |
|---|---|---|
| Periods | **P1** 2026-09-07 → 09-27 (weeks 1–3); **P2** 2026-11-09 → 11-29 (weeks 10–12) | The plan's first-three vs last-three weeks |
| Unit | **In-scope postings whose ATS publish date falls in the period** | Each posting belongs to at most one period, so the samples are independent. Comparing *open* postings would put long-lived postings in both periods; dropping that overlap would select on time-to-close, which A2 shows is related to skill count. |
| Primary hypothesis **H1** | Share of new tech postings mentioning **any GenAI skill** (LLM, GenAI, AI agents, RAG, prompt engineering, LangChain, fine-tuning, embeddings, evals, OpenAI API, Hugging Face) differs between P1 and P2 | One test, no multiplicity penalty; the drift question people actually ask in 2026 |
| H1 test | Two-proportion z-test, two-sided, α = 0.05; effect as pp difference with Newcombe 95% CI | |
| H1 composition control | Mantel–Haenszel OR stratified by company | Employer mix dominated A1 and A2; a shift in *who* posts must not read as a shift in *what* they ask for |
| Secondary: per skill | 27 skills present in ≥ 15% of in-scope postings open during weeks 1–3, **chosen from period-1 data only**. Two-proportion tests, **Benjamini–Hochberg q = 0.05** across all 27, raw and adjusted p reported, ranked by \|difference\| | The plan's table |
| Secondary: trend | Logistic regression of any-GenAI on weeks since 2026-09-07 + company fixed effects, all 12 weeks; OR per 4 weeks | The plan's "stronger version" |
| Secondary: stable panel | H1 on companies with postings in both periods | The plan's composition guard |
| Dictionary | `skills.yaml` v1; any later version is applied to both periods by replay | Extraction is replayable, so both periods are always parsed identically |

## Power, stated in advance

P1 has **42** in-scope postings by publish date, and P2 is expected to have about the same. At
n ≈ 42 vs 45 a two-proportion test has 80% power only for a shift of about **29 percentage points**
(baseline 30%). The open-stock GenAI share in weeks 1–3 is 31%, so GenAI mentions would need to
reach roughly 60% of new tech postings to be reliably detected. The 12-week trend model doesn't buy
much: for a linear trend, the middle weeks add less information than the extremes.

**So the likely outcome is a null, and it is pre-registered as interpretable:** a non-significant H1
will be reported as *"no shift larger than about 29 pp detected"*, not as "no drift". The plan
already says twelve weeks is short for labour-market drift; this quantifies how short.

## Calibration (run 2026-10-18, shuffled labels only)

`python -m analysis.a3_drift --permutation-check` puts the 101 in-scope postings published so far
into random pseudo-periods of 42 and 45, 1,000 times:

- H1 type-I error **0.059** (nominal 0.05; the z-test runs slightly liberal at this n, within about
  1 simulation SE)
- BH across 27 skills: P(any false discovery) **0.047**; mean false discoveries **0.066**

The tests behave as intended under no true difference. No real-period comparison was run.

## Interpretation rules (fixed now)

1. A non-significant H1 is reported as a bounded null with its CI, never as "no change".
2. Per-skill results are exploratory; only BH survivors are described as shifted.
3. If H1's raw and company-adjusted estimates disagree in sign, the adjusted one is the finding.
4. Anything not listed here is labelled exploratory in the report.

## Deviations

*None yet. Any change after 2026-10-18 is logged here with its date and reason.*

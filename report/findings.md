# Who's hiring, not what for: six weeks of India tech job postings, observed daily

*Rohit Raj · **Draft, 2026-10-18** (panel day 42 of 84). Numbers will be refreshed on the final
panel; A3 runs on 2026-12-01 exactly as [pre-registered](A3_preregistration.md). Every figure here
is produced by code in [`analysis/`](../analysis) from the dataset described in the
[data dictionary](../DATA_DICTIONARY.md).*

## Findings

1. **Who is hiring explains a posting's skill requirements about as well as its job title does.**
   Clustering 213 technical postings by required skills, cluster membership is as strongly
   associated with the employer (Cramér's V 0.46) as with the title family (0.49), and twice as
   strongly as with seniority (0.24). Two of the five clusters are essentially one company's
   template.
2. **"Software engineer" hides an infrastructure role.** The most stable cross-employer cluster
   (Go, Kubernetes, Terraform, multi-cloud) is split 22 / 20 between postings titled
   *software engineer* and postings titled *SRE / infrastructure*. It survives resampling (Jaccard
   0.67) and survives capping every employer at 15 postings (0.59).
3. **Postings stay open about six weeks, and the naive estimate is off by 2.5×.** The median posting
   closes **43 days** after the employer publishes it (95% CI 33–50; 47 days once reposts are
   excluded). Averaging how long closed postings were observed gives 17 days, because it ignores
   both still-open postings and how old postings already were when observation began.
4. **Role family doesn't detectably predict how long a posting stays open; most apparent
   predictors are employer mix.** No family differs from software engineering (joint test
   p = 0.60; infrastructure HR 0.84, 0.46–1.52). Remote work looks like it speeds closure (HR 1.82)
   until postings are compared within the same employer (0.91, 0.39–2.08). Of the effects tested,
   only onsite-vs-hybrid survives that comparison (HR 1.98, 1.15–3.41).
5. **Pay transparency is close to zero, and it's a company policy, not a market norm.** 3.8% of
   postings show an India pay range (95% CI 2.5–5.6%), all from one employer. Even there,
   engineering roles withhold pay far more often than other roles (3 of 9 disclose, against 19 of
   20; Fisher p = 0.001).
6. **Cities look specialised only because each city is mostly one employer.** NCR over-indexes on
   non-technical roles (p = 0.0003 pooled), but 80% of its postings are MongoDB's. Comparing within
   employers leaves no city difference (p = 0.19), and remote work is likewise set per employer
   (100% at GitLab and Twilio, 1% at Okta).

The thread running through all six: in this market sample, **the employer is the strongest
signal in the data**, and any analysis that doesn't condition on it reports employer composition
as if it were a fact about roles.

---

## The data

**What it is.** Every India-located posting on the public job boards of 13 tech employers, fetched
daily at 08:00 IST from their applicant tracking systems (Greenhouse, Ashby) since 2026-09-07.
After 42 days: **586 postings, 13,530 posting-days, 256 closures**. The same posting observed day
after day is what makes this a *panel*: it records when each posting appeared, when it disappeared,
and whether it changed, which a one-off scrape can't.

**Why ATS boards.** They're the employer's own source of truth, with stable job IDs and no
aggregator reposting. That makes daily identity reliable: every posting so far has an ATS ID, and no
two employers' postings are near-copies of each other (maximum cross-company text similarity 0.26).

**What's derived.** From the raw stored text and JSON, extraction produces:
- a **role family** (11 functions) and **seniority** (9 levels), as independent fields;
- **skills** (178-skill dictionary, after removing each employer's boilerplate text);
- **canonical city**, with Delhi, Gurugram and Noida collapsed to NCR;
- **workplace model** (remote / hybrid / onsite), recording which signal decided it, and NULL when
  unknown rather than assumed onsite;
- **disclosed pay**.

Extraction is re-run over the full history every day, so a parsing fix applies to week 1 as well
as today.

**The analysis population** is technical individual-contributor roles (software, infrastructure,
data, ML, QA, security; not managers or interns): **270 postings, 46%**. Analyses that contrast role
types use all 586.

**Choices that matter for the numbers, made before the results were seen:**

- **Time is measured from the employer's own publish date**, stored in the raw ATS record, not from
  first sighting. For postings first seen after day 1, 81% were seen within a day of publishing.
- **Postings already live on day 1 (332, median age 53 days) enter survival analysis at their
  age** (delayed entry), not at zero, and aren't discarded.
- **A whole board vanishing isn't a closure.** One employer (Postman) left Greenhouse mid-panel;
  its postings are censored, not counted as filled.
- **Duplicates are measured, not assumed away:** 13% of postings, 19% of technical ones, are
  near-copies of another (one description posted at several levels, one role per city, reposts).
  See A7.

Full field definitions, null semantics and known issues: [data dictionary](../DATA_DICTIONARY.md).

---

## A1 — Does one job title hide several distinct roles?

**Question.** If skill requirements form groups that cut across job titles, the title is
concealing structure.

**Approach.**
- Posting × skill matrix (213 technical postings after merging same-company identical skill sets;
  108 skills), TF-IDF weighted, spherical k-means.
- Because k-means always finds clusters, each k is tested against a **null** that keeps every
  skill's frequency and every posting's skill count but scrambles which skills go together
  (curveball randomisation).
- Stability is tested by bootstrap, and by **capping each employer at 15 postings** so no single
  company's volume can carry a cluster.

**Result.**
- **The structure is real but graded, not discrete.** Observed silhouette beats the null at every k
  (z = 26–40). But absolute silhouettes are weak (≤ 0.15), the gap statistic never stops rising,
  and bootstrap agreement is moderate at every k (ARI ≈ 0.4–0.5). The most stable partition is
  k = 5:

| Cluster | n | Signature | Reading | Stability (bootstrap / employer-capped) |
|---|---|---|---|---|
| Data & analytics | 30 | Tableau, dbt, statistics, Snowflake, SQL | **Cross-employer archetype**; includes 8 postings titled *software engineer* | 0.69 / 0.68 |
| Cloud infrastructure | 47 | Go, Kubernetes, Terraform, AWS/GCP/Azure | **Cross-employer archetype**; 22 *software engineer*, 20 *SRE/infra* | 0.67 / 0.59 |
| Okta product engineering | 67 | Java, React, identity | 78% one employer; **dissolves when capped** | 0.51 / 0.34 |
| GitLab Rails | 26 | Ruby, Rails | 81% one employer: a company's stack, not a role | 0.75 / 0.73 |
| Residual | 43 | GenAI + business systems | Not stable | 0.48 / 0.45 |

- **Within software-engineering titles alone,** membership tracks the employer (V 0.56) far more
  than seniority (0.22).

![A1 cluster membership by employer](figures/a1_all_in_scope_cluster_by_company.png)

**What would change my mind.** A cluster structure that held up once employer is controlled for
(for example, the Okta cluster surviving the cap), or a natural number of clusters showing up in
the gap curve on the full 12-week panel. Both are checked again at the end of collection.

[Full A1 write-up →](A1_clustering.md)

---

## A2 — How long do postings stay open, and what predicts closure?

**Question.** Which roles are hard to fill? A posting open for eight weeks says something about
supply and demand that a snapshot can't.

**Approach.**
- Kaplan–Meier and Cox models with time from the publish date.
- Open postings are right-censored; postings already live on day 1 enter at their age (left
  truncation, handled by delayed entry).
- Cox is fitted twice: pooled with company-clustered standard errors, and **stratified by
  company**, so each employer has its own baseline and comparisons are within employers.
- Robustness:
  - only postings seen at birth;
  - the first disappearance counted as closure;
  - closures at 28–32 days censored, in case a monthly cycle is expiry rather than hiring;
  - reposts censored (from A7).

**Result.**
- **The median posting is open 43 days** (95% CI 33–50).
- **Role family: no detectable effect** (Cox likelihood-ratio test across families, p = 0.60).
- **Effects that look real in the pooled model shrink once compared within employers:**

| | Pooled HR (95% CI) | Within employer | |
|---|---|---|---|
| Remote vs hybrid | 1.82 (1.07–3.12) | 0.91 (0.39–2.08) | employer mix |
| NCR vs Bengaluru | 0.69 (0.51–0.92) | 0.78 (0.41–1.50) | employer mix |
| Principal vs senior | 0.50 (0.27–0.92) | 0.59 (0.27–1.28) | employer mix |
| **Onsite vs hybrid** | 2.00 (0.90–4.43) | **1.98 (1.15–3.41)** | survives every spec |
| Entry/mid vs senior | 1.32 (0.85–2.06) | 1.79 (1.02–3.13) | borderline once reposts are censored (0.99–3.19) |
| Infrastructure vs SWE | 0.76 (0.51–1.14) | 0.84 (0.46–1.52) | no detectable difference |

![A2 hazard ratios, pooled vs within employer](figures/a2_cox_hazard_ratios.png)

**Caveats on the surviving effects.** The onsite effect rests on three employers that actually vary
their workplace model. The entry/mid group is mostly non-technical associate roles; the panel has
no junior *technical* postings at all.

**What would change my mind.** If the onsite effect disappeared when the three employers are
dropped one at a time, it's one company's practice, not a pattern. If closures turned out to be
mostly withdrawals rather than fills, none of this measures difficulty to hire.

[Full A2 write-up →](A2_survival.md)

---

## A3 — Has the skill mix shifted? *(pre-registered; runs 2026-12-01)*

**Question.** Did requirements change between the first and last three weeks of the panel, in a way
distinguishable from noise?

**Approach, fixed on 2026-10-18 before any data from the second period existed**
([pre-registration](A3_preregistration.md), committed and pushed):
- One primary test: the share of new technical postings mentioning any GenAI skill, weeks 1–3 vs
  weeks 10–12, with a company-adjusted (Mantel–Haenszel) version.
- 27 secondary skills, chosen mechanically from weeks 1–3 only, with Benjamini–Hochberg correction.
- Calibrated in advance on shuffled period labels (false-positive rate 0.059 vs nominal 0.05).

**What the design can and can't detect.** With about 42 postings per period, only shifts of
roughly **29 percentage points** are detectable with 80% power. A null result is likely, and it's
pre-declared as meaning *no shift larger than ~29 pp*, not *no drift*.

*Result: pending.*

---

## A5 — Can you predict, on day one, which postings close within 14 days? *(spec frozen; runs 2026-11-30)*

**Question.** From what a posting shows on its first day, can a model tell which ones will be gone
within two weeks?

**Approach, fixed on 2026-10-20 before any test posting existed** ([design](A5_prediction.md)):
- Postings seen from their first day; label = closed within 14 days, reposts not counted as closes.
- Temporal split: train on weeks 1–8, test on postings first seen afterwards. Also a weekly
  rolling-origin version.
- The plan's ladder: majority → single best feature → logistic regression → gradient boosting,
  with paired bootstrap CIs on each rung's lift. PR AUC, Brier and a calibration curve. No tuning.

**What the design can and can't detect.** The test set will have about 10 positives, so only lifts
of roughly 0.25 AUC are resolvable; the rolling pool (~35 positives) narrows that to about ±0.08.
An interim pipeline check already shows the trap to watch: most of its lift came from one
employer's remote postings, picked up through location rather than company.

*Result: pending.*

---

## A4 — Who discloses pay?

**Question.** The plan was to model pay among disclosers, and the selection into disclosure.

**Result: neither model is estimable, and that's the finding.**
- Only **22 of 586 postings (3.8%) show an India pay range, all from Airbnb.**
- Other employers' ATS records have salary fields filled with placeholders (0.0, USD 0.01). One
  employer tracks an INR budget per requisition but leaves it unpublished.
- With one disclosing employer, a pay regression has a single cluster, and a disclosure model would
  learn only the company name.
- **Within Airbnb, engineering withholds:** 3 of 9 engineering postings disclose, against 19 of 20
  other roles (Fisher p = 0.001).

**What would change my mind.** A second employer disclosing would make a real between-company
comparison possible. Nothing in this panel says anything about pay *levels* outside one company's
bands.

[Full A4 write-up →](A4_disclosure.md)

---

## A6 — Is the market geographically specialised? *(short form)*

**Question.** Do cities hire for different things, or are they scaled copies of each other?

**Approach.** Only Bengaluru (394) and NCR (114) have enough postings. Herfindahl concentration of
employers per city; location quotients and a city × family χ², pooled and with city shuffled
within employer ([details](A6_geography.md)).

**Result. City is a stand-in for employer.** Pooled, NCR over-indexes on non-technical roles
(LQ 1.56) and under-indexes on infrastructure (0.46), and the difference is significant
(p = 0.0003). But 80% of NCR postings are MongoDB's (HHI 6,542), and comparing within employers
leaves nothing (p = 0.19; technical roles p = 0.78). Remote work is an employer policy too: 100%
of GitLab's and Twilio's India postings, 1% of Okta's.

**What would change my mind.** A panel sampled to cover cities, with more than the two employers
that hire technical roles in both. Here that is a design point for a next panel, not a fix.

---

## A7 — How much of the panel is duplicated?

**Question.** Every other analysis treats postings as independent. Are they?

**Approach.**
- TF-IDF similarity of descriptions with company boilerplate removed.
- Exact all-pairs comparison within each employer; at this size, approximate methods like MinHash
  would only add error.
- Threshold 0.95, at the clear spike in nearest-neighbour similarity; 0.90 as sensitivity.

**Result.**
- **586 postings are 512 distinct roles (13% duplication); among technical roles, 219 of 270
  (19%).**
- The commonest form is **one description posted at several levels** (44 pairs), ahead of one role
  per city (25) and reposts (13).
- **This corrected A2:** 13 closures were reposts, the same role relisted under a new ID within two
  weeks. Censoring them moves the median from 43 to 47 days and turns the entry/mid finding from
  significant into borderline.

[Full A7 write-up →](A7_duplication.md)

---

## Limitations

These are the reasons not to over-read this report, in rough order of importance.

1. **This is a sample of global tech employers' India offices, not the Indian tech market.**
   Collecting from ATS job boards means covering companies that use Greenhouse or Ashby: funded,
   mostly US-headquartered firms. Indian-headquartered companies (Razorpay, Swiggy, CRED and
   others) aren't in the panel because no working Greenhouse, Ashby or Lever board was found for
   them. Large IT-services firms, which employ a large share of Indian engineers, aren't either.
2. **One employer is a third of the data**, and three are 63% of postings. Every analysis conditions
   on employer, but with 13 employers the within-employer comparisons rest on few companies.
3. **A closed posting is not a hire.** Closures mix filled, withdrawn, paused and reposted roles.
   A2 bounds reposting and expiry; it can't separate fills from withdrawals.
4. **Twelve weeks is short for labour-market drift**, and A3's design can only detect large shifts.
   This was stated before the data existed, so a null can't be explained away afterwards.
5. **Skills are dictionary matches, not requirements.** "Work with the SRE team" counts as SRE. The
   dictionary is versioned and every run reprocesses all history, but mention isn't demand.
6. **Role classification is rules-based and checked once, in-sample.** On 150 labelled titles the
   rules get family right 95% of the time and both fields 90%; Gemini gets 93% and 92%, no
   significant difference (exact McNemar p = 0.63). The rules are more precise on the scope filter
   (98% vs 93%), so the analyses keep them. Two caveats cut in opposite directions: the rules were
   tuned on these same titles (favours rules), and the labels were drafted by an LLM (Claude), then
   reviewed by me — every judgment call plus a random 20 of the rest, with full agreement (favours
   Gemini). A holdout of titles first seen after the rules were frozen is the fair test.
7. **No junior technical roles.** The panel contains none, so it says nothing about entry-level
   tech hiring, arguably the segment people most want to understand.

## Reproduce

```bash
pip install -e '.[analysis]'
python -m extraction                      # derived fields from raw data
python -m analysis.a1_clustering
python -m analysis.a2_survival
python -m analysis.a4_disclosure
python -m analysis.a7_duplication
python -m analysis.a3_drift               # on or after 2026-12-01
```

*Draft status: findings 1–5 are from day 42 of 84 and will be re-run on the final panel. Pending:
A3 and A5 results; title-classifier holdout check; final coverage refresh; dataset release decision.*

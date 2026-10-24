# Manas Twin - AI/ML Model and Framework Details

## 1. Prediction task
For every person and every day *t* (once at least 60 days of history exist and the person is **not** already in an episode):

> **y(t) = 1 if a depressive episode starts on any day in (t, t+7], else 0.**

Days inside an episode are excluded from training. The twin's job is early warning, not detecting an episode that has already started.

## 2. Inputs
**Daily wearable / phone signals (8):** sleep duration, sleep-onset time, sleep efficiency, resting heart rate, heart-rate variability (RMSSD), steps, late-night screen time (23:00–05:00), social contacts (distinct people called or messaged).

**EHR context:** age, sex, student/professional, prior depressive episodes, family history, hypothyroidism, anaemia, vitamin D deficiency, PHQ-9 screening results, diagnoses (ICD-10 F32), antidepressant start (ATC N06AB10), therapy referral.

**Context:** a known exam / work-crunch calendar (academic and corporate calendars are known in advance).

## 3. The personal baseline (the core of the twin)
For each signal *s* and day *t*:

- **Personal normal:** mean and SD of *s* over days *t-58 … t-14* (45-day window, ending 14 days ago so a slow build-up does not contaminate "normal").
- **Deviation:** `z7 = (7-day mean - personal mean) / personal SD`, plus a 3-day version (`z3`) and the week-on-week change (`dz7`).
- **Twin Drift Index:** the average of signed deviations, each aligned so that "toward risk" is positive (e.g. lower HRV and later bedtime count positively; sleep counts in either direction because depression can bring both insomnia and hypersomnia).

## 4. Feature sets
| Set | Count | Contents |
|---|---|---|
| **Twin** | 50 | per signal: 7-day mean, z7, z3, dz7 (32) · Twin Drift Index · bedtime irregularity · non-wear days · 15 EHR / context features |
| **Population** (comparison) | 33 | per signal: 7-day and 3-day means (16) · bedtime irregularity · non-wear days · same 15 EHR / context features |

Missing days (watch not worn) are left as missing. XGBoost learns how to route them, and *non-wear itself* is a feature, because people often stop wearing devices when low.

## 5. Algorithm
**XGBoost** gradient-boosted decision trees: `n_estimators=350, max_depth=4, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=2`. No class re-weighting, so outputs stay usable as probabilities (Brier score 0.0095 at 0.96% prevalence).

**Why gradient boosting:** strong on tabular data, handles missing values natively, fast enough to re-score a person daily on a phone or edge server, and compatible with exact SHAP explanations. A sequence model (LSTM/Transformer) is on the roadmap once real longitudinal data is available. With a few hundred episodes, tree ensembles on engineered baseline features are the more reliable choice.

## 6. Validation protocol
- **Split by person:** 70% of people for training (420), 30% held out (180). No person appears in both.
- **Alert rule in the app (hysteresis):** the model's daily estimate is continuous, but the alert label is not. To avoid single-day flips, **Early warning** requires the estimate to be above the line *and* either the previous day already at Watch level or the estimate ≥ 1.5× the line; otherwise a day that just crossed the line is shown as **Watch · rising**. The hero text also says when the estimate is driven mainly by background risk (prior episode, PHQ-9 swing, family history) rather than this week's rhythm. Evaluation metrics below use the raw daily threshold.
- **Alert threshold:** chosen on a validation subset of the *training* people at 95% day-level specificity, then frozen (risk ≥ 3.0%, about 3× the base rate). "Watch" = half the threshold.
- **Episode-level metrics:** an episode counts as *detected* if risk crossed the threshold on any of the 7 days before onset. Lead time = days between first alert and onset. False alerts are measured on "quiet" days with no onset in the next 14 days.

## 7. Results (unseen synthetic people)
| Metric | Twin | Population |
|---|---|---|
| AUROC | **0.846** | 0.807 |
| AUPRC (prevalence 0.96%) | **0.048** (5× chance) | 0.035 |
| Day-level sensitivity / specificity at threshold | 0.25 / 0.96 | 0.26 / 0.95 |
| Episodes flagged in advance | **19 / 40 (47.5%)** | 16 / 40 (40%) |
| Median lead time | 6 days | 6.5 days |
| False-alert days per 30 quiet days | **1.17** | 1.55 |

Source: `reports/metrics.json`, reproducible with `python -m src.train`.

**Interpretation.** Personalisation gives a clear gain in discrimination and catches more episodes with fewer false alerts. Absolute precision is low because episode onsets are rare (under 1% of days). That is why the twin is designed as a **prompt for a human check-in**, not an autonomous diagnosis.

## 8. Explainability
SHAP TreeExplainer gives, for every prediction, how much each feature pushed the risk up or down. The dashboard translates the top drivers into plain language using the personal baseline, e.g. *"Daily steps 18% below usual"*, *"Sleep 54 min less than usual"*.

Most influential features overall (`reports/shap_global_importance.csv`): latest PHQ-9, late-night screen time, change in PHQ-9 since the previous screening, prior episodes, sleep efficiency, Twin Drift Index, resting-HR deviation.

Note on *change in PHQ-9*: large recent swings (e.g. a big drop after recovering from an episode) mark people with recent episode history, who carry higher recurrence risk. The model learns this, which is why a *fall* in PHQ-9 can still push risk up.

## 9. What-if simulation
`DigitalTwin.what_if()` edits the last 7 days of the person's own data (e.g. +1 h sleep, bedtime 45 min earlier, +3,000 steps, -45 min late-night screen, +5 contacts), rebuilds the twin features and re-scores. This is a **model-based sensitivity analysis**. It is associative, not a causal estimate, and is presented that way in the UI.

## 10. Real-data check (DEPRESJON)
Rhythm features computed from real wrist-actigraphy (55 people) separate depressed people from controls with **AUROC 0.82 (95% CI 0.71–0.92)** under leave-one-person-out cross-validation. Depressed participants showed lower mean activity, more rest and less day-to-day variability. The dataset cannot test the within-person early-warning step (no episode onsets, ~13 days per person). Full results: `reports/depresjon_validation.json`, script: `src/validate_depresjon.py`.

## 11. Known limitations
- Trained and evaluated on synthetic data (see `DATA.md`). Effect sizes are assumptions informed by literature, not measurements.
- A single global model with personalised features. True per-person model adaptation is future work.
- No fairness audit on real demographic data yet. Device-ownership bias is a real deployment risk.
- Out of scope: suicide or self-harm risk, bipolar disorder, diagnosis.

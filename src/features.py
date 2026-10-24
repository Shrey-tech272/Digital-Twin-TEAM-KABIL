"""
Feature engineering: turns raw daily signals + EHR into twin-state features.

The key idea of the twin is the PERSONAL BASELINE: for every signal we learn what
"normal" looks like for *this* person (rolling 45-day window that ends 14 days ago)
and express the last few days as deviations (z-scores) from that normal.

Two feature sets are produced:
  * TWIN features       -> personal-baseline deviations + raw levels + EHR context
  * POPULATION features -> raw levels + EHR context only (what a one-size-fits-all
                           model would see). Used as the comparison baseline.
"""
import numpy as np
import pandas as pd

from src import config as C

EHR_EVENT_TYPES = {"phq9_screening", "diagnosis", "medication_start", "therapy_referral"}


def _ehr_timeline(enc_p, n_days):
    """Daily EHR context known up to (and including) each day."""
    out = pd.DataFrame(index=range(n_days))
    e = enc_p[enc_p["type"].isin(EHR_EVENT_TYPES)].sort_values("day")

    phq = e.dropna(subset=["phq9"]).groupby("day")["phq9"].last()
    s = pd.Series(np.nan, index=out.index)
    s.loc[phq.index] = phq.values
    out["phq9_last"] = s.ffill()
    last_day = pd.Series(np.where(s.notna(), out.index, np.nan), index=out.index).ffill()
    out["days_since_phq9"] = out.index - last_day
    prev = s.dropna()
    delta = prev.diff()
    d = pd.Series(np.nan, index=out.index)
    d.loc[delta.index] = delta.values
    out["phq9_change"] = d.ffill()

    med = e[e["type"] == "medication_start"]["day"]
    out["on_antidepressant"] = (out.index >= med.min()).astype(int) if len(med) else 0
    dx = e[e["type"] == "diagnosis"]["day"]
    out["diagnosed_episodes_so_far"] = [int((dx <= t).sum()) for t in out.index]
    last_dx = pd.Series(np.nan, index=out.index)
    last_dx.loc[dx.values] = dx.values
    last_dx = last_dx.ffill()
    out["days_since_last_diagnosis"] = out.index - last_dx
    return out


def build_patient_features(daily_p, enc_p, patient):
    """Features for one patient (daily_p sorted by day)."""
    df = daily_p.sort_values("day").reset_index(drop=True).copy()
    n = len(df)
    feats = pd.DataFrame({"patient_id": df["patient_id"], "day": df["day"]})

    dev_terms = []
    for s in C.SIGNALS:
        x = df[s]
        m7 = x.rolling(7, min_periods=4).mean()
        m3 = x.rolling(3, min_periods=2).mean()
        hist = x.shift(C.BASELINE_GAP)
        bm = hist.rolling(C.BASELINE_WINDOW, min_periods=20).mean()
        bs = hist.rolling(C.BASELINE_WINDOW, min_periods=20).std()
        bs = np.maximum(bs, 0.02 * bm.abs() + 1e-3)   # guard against tiny std
        z7 = (m7 - bm) / bs
        z3 = (m3 - bm) / bs
        feats[f"raw7_{s}"] = m7
        feats[f"raw3_{s}"] = m3
        feats[f"base_{s}"] = bm
        feats[f"z7_{s}"] = z7
        feats[f"z3_{s}"] = z3
        feats[f"dz7_{s}"] = (m7 - m7.shift(7)) / bs        # change vs previous week
        direction = C.RISK_DIRECTION[s]
        dev_terms.append(z7.abs() if direction == 0 else direction * z7)

    feats["twin_deviation_index"] = pd.concat(dev_terms, axis=1).clip(-4, 4).mean(axis=1)
    feats["sleep_onset_irregularity7"] = df["sleep_onset"].rolling(7, min_periods=4).std()
    feats["missing_days7"] = df["sleep_hours"].isna().rolling(7, min_periods=1).sum()
    feats["known_stressor_period"] = df["known_stressor_period"]

    ehr = _ehr_timeline(enc_p, n)
    feats = pd.concat([feats, ehr.reset_index(drop=True)], axis=1)

    feats["age"] = patient["age"]
    feats["is_female"] = int(patient["sex"] == "F")
    feats["is_student"] = int(patient["cohort"] == "student")
    feats["prior_episodes"] = patient["prior_episodes"]
    feats["family_history_depression"] = int(bool(patient["family_history_depression"]))
    feats["hypothyroidism"] = int(bool(patient["hypothyroidism"]))
    feats["anaemia"] = int(bool(patient["anaemia"]))
    feats["vitamin_d_deficiency"] = int(bool(patient["vitamin_d_deficiency"]))
    return feats


def add_labels(feats, daily_p, horizon=C.HORIZON_DAYS):
    """Label = an episode starts within the next `horizon` days (only for days not already in an episode)."""
    df = daily_p.sort_values("day").reset_index(drop=True)
    onset = df["gt_onset"].values
    n = len(df)
    y = np.full(n, np.nan)
    for t in range(n - horizon):
        y[t] = onset[t + 1:t + 1 + horizon].max()
    feats = feats.copy()
    feats["label"] = y
    feats["in_episode"] = df["gt_in_episode"].values
    return feats


STATIC_EHR = ["phq9_last", "days_since_phq9", "phq9_change", "on_antidepressant",
              "diagnosed_episodes_so_far", "days_since_last_diagnosis", "age", "is_female",
              "is_student", "prior_episodes", "family_history_depression", "hypothyroidism",
              "anaemia", "vitamin_d_deficiency", "known_stressor_period"]
COMMON = ["sleep_onset_irregularity7", "missing_days7"]


def twin_feature_names():
    f = []
    for s in C.SIGNALS:
        f += [f"raw7_{s}", f"z7_{s}", f"z3_{s}", f"dz7_{s}"]
    return f + ["twin_deviation_index"] + COMMON + STATIC_EHR


def population_feature_names():
    f = []
    for s in C.SIGNALS:
        f += [f"raw7_{s}", f"raw3_{s}"]
    return f + COMMON + STATIC_EHR


def build_all(patients, daily, enc):
    out = []
    enc_g = dict(tuple(enc.groupby("patient_id")))
    daily_g = dict(tuple(daily.groupby("patient_id")))
    for _, p in patients.iterrows():
        pid = p["patient_id"]
        d = daily_g[pid]
        e = enc_g.get(pid, enc.iloc[0:0])
        f = build_patient_features(d, e, p)
        out.append(add_labels(f, d))
    return pd.concat(out, ignore_index=True)


def training_rows(feats):
    """Rows usable for training/evaluation."""
    m = (feats["day"] >= C.MIN_HISTORY_DAYS) & (feats["in_episode"] == 0) & feats["label"].notna()
    return feats[m].copy()

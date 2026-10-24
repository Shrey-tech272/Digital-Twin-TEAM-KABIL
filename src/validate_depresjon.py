"""
Real-data check of the twin's ideas on the public DEPRESJON actigraphy dataset.

DEPRESJON (Garcia-Ceja et al., MMSys 2018; Simula Research Laboratory) has minute-level
wrist-actigraphy for 23 people with depression (unipolar or bipolar, MADRS-scored) and
32 non-depressed controls, ~13 days each. It has no EHR and no labelled episode
onsets, so it cannot train the full 7-day early-warning model. It CAN answer three
questions on real humans:

  Q1. Do twin-style rhythm features (activity level, night activity, circadian
      regularity, day-to-day variability) separate depressed people from controls?
      -> leave-one-person-out logistic regression, AUROC with bootstrap CI.
  Q2. Does a *personal-baseline* view add information over raw levels?
      -> compare day-level classification using each person's own-baseline z-scores
         vs. raw daily values (grouped CV by person).
  Q3. Within the depressed group, do the features track symptom severity (MADRS)?
      -> Spearman correlations.

Run:  python -m src.validate_depresjon
Outputs: reports/depresjon_validation.json, reports/figures/depresjon_*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config as C

ROOT = C.DATA_DIR / "depresjon"
FEATS = ["mean_activity", "active_minutes", "night_share", "rel_amplitude", "intradaily_var", "rest_minutes"]
LABELS = {"mean_activity": "Mean activity", "active_minutes": "Active minutes / day", "night_share": "Night-time activity share",
          "rel_amplitude": "Day-night contrast (RA)", "intradaily_var": "Fragmentation of rhythm (IV)",
          "rest_minutes": "Minutes at rest / day"}


def daily_features(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    rows = []
    for d, g in df.groupby("date"):
        if len(g) < 1440 * 0.95:          # keep complete days only
            continue
        a = g["activity"].values.astype(float)
        hourly = g.groupby("hour")["activity"].mean().reindex(range(24)).fillna(0).values
        # relative amplitude: most-active 10 h vs least-active 5 h (circular)
        h2 = np.r_[hourly, hourly]
        m10 = max(h2[i:i + 10].mean() for i in range(24))
        l5 = min(h2[i:i + 5].mean() for i in range(24))
        ra = (m10 - l5) / (m10 + l5) if (m10 + l5) > 0 else 0
        # intradaily variability (hourly): mean squared successive difference / variance
        iv = (np.sum(np.diff(hourly) ** 2) / 23) / (np.var(hourly) + 1e-9)
        rows.append(dict(date=str(d), mean_activity=a.mean(), active_minutes=int((a > 100).sum()),
                         night_share=g[g.hour < 6]["activity"].sum() / max(a.sum(), 1),
                         rel_amplitude=ra, intradaily_var=iv, rest_minutes=int((a == 0).sum())))
    return pd.DataFrame(rows)


def load():
    if not ROOT.exists():
        raise FileNotFoundError(f"DEPRESJON not found at {ROOT}. See src/depresjon_loader.py docstring.")
    frames = []
    for grp, lab in [("condition", 1), ("control", 0)]:
        for f in sorted((ROOT / grp).glob("*.csv"), key=lambda p: int(p.stem.split("_")[1])):
            d = daily_features(f)
            d["person"], d["depressed"] = f.stem, lab
            frames.append(d)
    days = pd.concat(frames, ignore_index=True)
    scores = pd.read_csv(ROOT / "scores.csv").rename(columns={"number": "person"})
    return days, scores


def personal_z(days):
    """Each day expressed against the person's OWN baseline (leave-that-day-out mean/sd)."""
    out = days.copy()
    for f in FEATS:
        g = out.groupby("person")[f]
        n = g.transform("count")
        mu = (g.transform("sum") - out[f]) / (n - 1)
        sd = g.transform("std").replace(0, np.nan)
        out[f"z_{f}"] = ((out[f] - mu) / sd).fillna(0)
    return out


def boot_auc(y, p, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    vals = []
    for _ in range(n):
        pick = rng.choice(ids, len(ids), replace=True)
        idx = np.concatenate([np.where(groups == i)[0] for i in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    C.REPORT_DIR.mkdir(exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    days, scores = load()
    res = {"dataset": "DEPRESJON (Garcia-Ceja et al., 2018)", "people": int(days.person.nunique()),
           "depressed": int(days.groupby("person").depressed.first().sum()),
           "complete_days": int(len(days)), "days_per_person_median": float(days.groupby("person").size().median())}

    # ---- Q1: person-level separation --------------------------------------------------
    per = days.groupby("person").agg({**{f: "mean" for f in FEATS}, "depressed": "first"})
    per["day_to_day_var"] = days.groupby("person")["mean_activity"].std() / days.groupby("person")["mean_activity"].mean()
    X = per[FEATS + ["day_to_day_var"]].values
    y = per["depressed"].values
    groups = np.arange(len(per))
    p = np.zeros(len(per))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000)).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y, p)
    lo, hi = boot_auc(y, p, groups)
    res["q1_person_level"] = {"auroc_lopo": round(auc, 3), "ci95": [round(lo, 3), round(hi, 3)],
                              "sensitivity_at_spec80": None}
    neg = np.sort(p[y == 0])
    thr = neg[int(np.ceil(0.8 * len(neg))) - 1]
    res["q1_person_level"]["sensitivity_at_spec80"] = round(float((p[y == 1] >= thr).mean()), 3)
    # univariate group differences
    uni = {}
    for f in FEATS + ["day_to_day_var"]:
        a, b = per.loc[per.depressed == 1, f], per.loc[per.depressed == 0, f]
        u = mannwhitneyu(a, b, alternative="two-sided").pvalue
        uni[f] = {"depressed_median": round(float(a.median()), 3), "control_median": round(float(b.median()), 3),
                  "p_value": round(float(u), 4), "auroc_alone": round(float(roc_auc_score(y, per[f])), 3)}
    res["q1_univariate"] = uni

    # ---- Q2: day-level, raw vs personal-baseline ---------------------------------------
    dz = personal_z(days)
    gk = GroupKFold(n_splits=5)
    out = {}
    for name, cols in [("raw", FEATS), ("personal_z", [f"z_{f}" for f in FEATS]), ("raw+personal_z", FEATS + [f"z_{f}" for f in FEATS])]:
        Xd, yd, gd = dz[cols].values, dz["depressed"].values, dz["person"].values
        pd_ = np.zeros(len(dz))
        for tr, te in gk.split(Xd, yd, gd):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000)).fit(Xd[tr], yd[tr])
            pd_[te] = m.predict_proba(Xd[te])[:, 1]
        out[name] = {"auroc_day_level": round(roc_auc_score(yd, pd_), 3), "ci95": [round(v, 3) for v in boot_auc(yd, pd_, gd, 500)]}
    res["q2_day_level"] = out
    res["q2_note"] = ("Personal z-scores alone cannot separate groups here (every person is compared with themselves, "
                      "and there are no within-person episode onsets in this dataset), so the honest reading is: on real "
                      "data the raw rhythm features carry the between-person signal, and the personal baseline is the "
                      "layer that would detect within-person change over time - which needs longitudinal data to test.")

    # ---- Q3: severity within the depressed group ---------------------------------------
    cond = per[per.depressed == 1].merge(scores[["person", "madrs1", "madrs2"]], left_index=True, right_on="person")
    cond["madrs"] = cond[["madrs1", "madrs2"]].mean(axis=1)
    q3 = {}
    for f in FEATS + ["day_to_day_var"]:
        r, pv = spearmanr(cond[f], cond["madrs"])
        q3[f] = {"spearman_rho": round(float(r), 3), "p_value": round(float(pv), 3)}
    res["q3_madrs_correlation"] = q3

    # ---- examples for the app ------------------------------------------------------------
    ex = {}
    for person in ["condition_2", "control_3"]:
        d = days[days.person == person].head(14)
        ex[person] = {"dates": d.date.tolist(), "mean_activity": [round(v, 1) for v in d.mean_activity],
                      "night_share": [round(v, 3) for v in d.night_share], "depressed": int(d.depressed.iloc[0])}
    res["examples"] = ex
    (C.REPORT_DIR / "depresjon_validation.json").write_text(json.dumps(res, indent=2))

    # ---- figure -----------------------------------------------------------------------------
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, f in zip(ax, ["mean_activity", "night_share", "intradaily_var"]):
        data = [per.loc[per.depressed == 0, f], per.loc[per.depressed == 1, f]]
        a.boxplot(data, tick_labels=["Control (n=32)", "Depressed (n=23)"], widths=0.5,
                  patch_artist=True, boxprops=dict(facecolor="#CFE5E1"), medianprops=dict(color="#0F6E6E", lw=2))
        a.set_title(LABELS[f] + f"\n(p = {uni[f]['p_value']:.3f})", fontsize=11)
    fig.suptitle("Real wrist-actigraphy (DEPRESJON): twin rhythm features by group", fontsize=12)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "depresjon_group_features.png", dpi=160)
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=2))


if __name__ == "__main__":
    main()

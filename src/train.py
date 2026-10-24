"""
Train and evaluate the Manas Twin risk engine.

* Patient-level split (70% train / 30% test) - no person appears in both, so the
  test score reflects performance on NEW people, not memorised ones.
* Two models with identical algorithm and hyper-parameters:
    - TWIN model        : uses personal-baseline deviation features
    - POPULATION model  : uses raw values only (one-size-fits-all)
  The gap between them is the value added by personalisation.
* Alert threshold chosen on a validation slice of the TRAIN patients
  (target specificity 0.95), then frozen and applied to the test set.

Run:  python -m src.train
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

from src import config as C
from src import features as F

PARAMS = dict(n_estimators=350, max_depth=4, learning_rate=0.04, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
              eval_metric="logloss", random_state=C.SEED, n_jobs=4)


def load():
    p = pd.read_csv(C.DATA_DIR / "patients.csv")
    d = pd.read_csv(C.DATA_DIR / "daily_wearables.csv.gz")
    e = pd.read_csv(C.DATA_DIR / "encounters.csv")
    return p, d, e


def split_patients(ids, test_size, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    a, b = next(gss.split(ids, groups=ids))
    return set(np.asarray(ids)[a]), set(np.asarray(ids)[b])


def threshold_for_specificity(y, p, spec=0.95):
    neg = np.sort(p[y == 0])
    return float(neg[int(np.ceil(spec * len(neg))) - 1])


def episode_metrics(feats_test, risk_col, thr, horizon=C.HORIZON_DAYS):
    """Was each episode flagged in the `horizon` days before onset? With how much lead time?"""
    detected, leads, n_eps = 0, [], 0
    false_alert_days, quiet_days = 0, 0
    for pid, g in feats_test.groupby("patient_id"):
        g = g.set_index("day")
        onsets = g.index[(g["in_episode"] == 1) & (g["in_episode"].shift(1, fill_value=0) == 0)]
        for o in onsets:
            if o - horizon < C.MIN_HISTORY_DAYS:
                continue
            n_eps += 1
            w = g.loc[o - horizon:o - 1]
            w = w[w["in_episode"] == 0]
            hits = w.index[w[risk_col] >= thr]
            if len(hits):
                detected += 1
                leads.append(int(o - hits.min()))
        # false alerts: alerts on days with no onset in the next 14 days, outside episodes
        valid = g[(g.index >= C.MIN_HISTORY_DAYS) & (g["in_episode"] == 0)]
        for t, row in valid.iterrows():
            nxt = g.loc[t + 1:t + 14, "in_episode"]
            if nxt.sum() == 0 and len(nxt) == 14:
                quiet_days += 1
                false_alert_days += int(row[risk_col] >= thr)
    return {
        "episodes_evaluated": n_eps,
        "episodes_detected": detected,
        "episode_detection_rate": round(detected / max(n_eps, 1), 3),
        "median_lead_time_days": float(np.median(leads)) if leads else None,
        "false_alert_days_per_30_quiet_days": round(30 * false_alert_days / max(quiet_days, 1), 2),
    }


def fit(X, y):
    m = XGBClassifier(**PARAMS)
    m.fit(X, y)
    return m


def main():
    C.MODEL_DIR.mkdir(exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    patients, daily, enc = load()
    print("Building features ...")
    feats = F.build_all(patients, daily, enc)
    rows = F.training_rows(feats)

    ids = patients["patient_id"].values
    train_ids, test_ids = split_patients(ids, 0.30, C.SEED)
    fit_ids, val_ids = split_patients(np.array(sorted(train_ids)), 0.25, C.SEED + 1)

    sets = {"twin": F.twin_feature_names(), "population": F.population_feature_names()}
    results, models, thresholds = {}, {}, {}
    tr = rows[rows.patient_id.isin(train_ids)]
    te = rows[rows.patient_id.isin(test_ids)]

    for name, cols in sets.items():
        # threshold on validation patients
        a = rows[rows.patient_id.isin(fit_ids)]
        b = rows[rows.patient_id.isin(val_ids)]
        m_val = fit(a[cols], a["label"])
        thr = threshold_for_specificity(b["label"].values, m_val.predict_proba(b[cols])[:, 1], 0.95)
        # final model on all train patients
        m = fit(tr[cols], tr["label"])
        p = m.predict_proba(te[cols])[:, 1]
        y = te["label"].values
        pred = p >= thr
        res = {
            "auroc": round(roc_auc_score(y, p), 3),
            "auprc": round(average_precision_score(y, p), 3),
            "prevalence": round(float(y.mean()), 4),
            "brier": round(brier_score_loss(y, p), 4),
            "threshold": round(thr, 4),
            "sensitivity_day_level": round(float(pred[y == 1].mean()), 3),
            "specificity_day_level": round(float(1 - pred[y == 0].mean()), 3),
        }
        # episode-level evaluation needs every test day (incl. episode days)
        ft = feats[feats.patient_id.isin(test_ids)].copy()
        ft["risk"] = m.predict_proba(ft[cols])[:, 1]
        ft.loc[ft["day"] < C.MIN_HISTORY_DAYS, "risk"] = np.nan
        res.update(episode_metrics(ft, "risk", thr))
        results[name] = res
        models[name] = m
        thresholds[name] = thr
        roc = roc_curve(y, p)
        prc = precision_recall_curve(y, p)
        res["_roc"] = (roc[0], roc[1])
        res["_pr"] = (prc[1], prc[0])
        print(name, {k: v for k, v in res.items() if not k.startswith("_")})

    # ---- save models + metadata ----
    models["twin"].save_model(C.MODEL_DIR / "twin_model.json")
    models["population"].save_model(C.MODEL_DIR / "population_model.json")
    meta = {
        "twin_features": sets["twin"], "population_features": sets["population"],
        "threshold": thresholds["twin"], "population_threshold": thresholds["population"],
        "horizon_days": C.HORIZON_DAYS, "train_patients": sorted(train_ids),
        "test_patients": sorted(test_ids), "base_rate": float(tr["label"].mean()),
        "xgb_params": PARAMS,
    }
    (C.MODEL_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))

    # ---- risk scores for every patient-day (used by the dashboard) ----
    feats["risk"] = models["twin"].predict_proba(feats[sets["twin"]])[:, 1]
    feats.loc[feats["day"] < C.MIN_HISTORY_DAYS, "risk"] = np.nan
    keep = ["patient_id", "day", "risk", "twin_deviation_index"]
    feats[keep].round(5).to_csv(C.DATA_DIR / "risk_scores.csv.gz", index=False, compression="gzip")

    # ---- global explanation (SHAP) on test set ----
    cols = sets["twin"]
    sample = te[cols].sample(min(4000, len(te)), random_state=C.SEED)
    sv = shap.TreeExplainer(models["twin"]).shap_values(sample)
    imp = pd.Series(np.abs(sv).mean(0), index=cols).sort_values(ascending=False)
    imp.round(5).to_csv(C.REPORT_DIR / "shap_global_importance.csv", header=["mean_abs_shap"])

    # ---- metrics file ----
    clean = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items()}
    clean["data"] = json.loads((C.DATA_DIR / "cohort_summary.json").read_text())
    clean["split"] = {"train_patients": len(train_ids), "test_patients": len(test_ids),
                      "train_rows": int(len(tr)), "test_rows": int(len(te)),
                      "test_positive_rows": int(te["label"].sum())}
    clean["top_features"] = imp.head(12).round(4).to_dict()
    (C.REPORT_DIR / "metrics.json").write_text(json.dumps(clean, indent=2))

    # ---- figures ----
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    col = {"twin": "#0F6E6E", "population": "#B9B9B9"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for n in ["population", "twin"]:
        fpr, tpr = results[n]["_roc"]
        ax[0].plot(fpr, tpr, color=col[n], lw=2.4, label=f"{n.title()} model (AUROC {results[n]['auroc']})")
        r, pr = results[n]["_pr"]
        ax[1].plot(r, pr, color=col[n], lw=2.4, label=f"{n.title()} model (AUPRC {results[n]['auprc']})")
    ax[0].plot([0, 1], [0, 1], ls="--", c="#999", lw=1)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC - unseen test patients")
    ax[1].axhline(results["twin"]["prevalence"], ls="--", c="#999", lw=1)
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall - unseen test patients")
    for a in ax:
        a.legend(frameon=False, loc="lower right" if a is ax[0] else "upper right")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "roc_pr_twin_vs_population.png", dpi=160)
    plt.close(fig)

    top = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([readable(f) for f in top.index], top.values, color="#0F6E6E")
    ax.set(xlabel="Mean |SHAP value|", title="What drives the twin's risk estimate")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "shap_top_features.png", dpi=160)
    plt.close(fig)
    print("Saved models, metrics and figures.")


def readable(f):
    """Human-readable feature name."""
    from src.config import SIGNAL_LABELS as L
    for pre, txt in [("z7_", "{} vs personal normal (7d)"), ("z3_", "{} vs personal normal (3d)"),
                     ("dz7_", "{} week-on-week change"), ("raw7_", "{} (7-day avg)"),
                     ("raw3_", "{} (3-day avg)"), ("base_", "{} personal baseline")]:
        if f.startswith(pre):
            return txt.format(L[f[len(pre):]])
    return {
        "twin_deviation_index": "Overall drift from personal normal",
        "sleep_onset_irregularity7": "Irregular bedtime (7d)",
        "missing_days7": "Days without wearable data (7d)",
        "phq9_last": "Latest PHQ-9 score", "days_since_phq9": "Days since last PHQ-9",
        "phq9_change": "Change in PHQ-9", "on_antidepressant": "On antidepressant",
        "diagnosed_episodes_so_far": "Diagnosed episodes (in follow-up)",
        "days_since_last_diagnosis": "Days since last diagnosis",
        "prior_episodes": "Prior depressive episodes", "family_history_depression": "Family history",
        "known_stressor_period": "Exam / work-crunch period", "is_student": "Student",
        "is_female": "Female", "age": "Age", "hypothyroidism": "Hypothyroidism",
        "anaemia": "Anaemia", "vitamin_d_deficiency": "Vitamin D deficiency",
    }.get(f, f)


if __name__ == "__main__":
    main()

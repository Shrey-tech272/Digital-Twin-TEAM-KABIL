"""
The Digital Twin of one person.

A DigitalTwin object bundles, for a single patient:
  * their EHR (demographics, history, PHQ-9 screenings, diagnoses, medication)
  * their daily wearable / phone stream
  * their learned PERSONAL BASELINE ("what normal looks like for you")
  * the risk engine (XGBoost) + explainer (SHAP)
and exposes the four twin capabilities:
  1. state(day)        - mirror: current physiology vs personal normal
  2. risk(day)         - predict: probability of a depressive episode starting in the next 7 days
  3. explain(day)      - explain: which changes are driving the risk
  4. what_if(day, ...) - simulate: how risk changes if behaviour changes
"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from src import config as C
from src import features as F
from src.train import readable


@lru_cache(maxsize=1)
def load_assets():
    patients = pd.read_csv(C.DATA_DIR / "patients.csv")
    daily = pd.read_csv(C.DATA_DIR / "daily_wearables.csv.gz")
    enc = pd.read_csv(C.DATA_DIR / "encounters.csv")
    risk = pd.read_csv(C.DATA_DIR / "risk_scores.csv.gz")
    meta = json.loads((C.MODEL_DIR / "model_meta.json").read_text())
    model = XGBClassifier()
    model.load_model(C.MODEL_DIR / "twin_model.json")
    explainer = shap.TreeExplainer(model)
    return dict(patients=patients, daily=daily, enc=enc, risk=risk, meta=meta,
                model=model, explainer=explainer)


def fmt_clock(h):
    if h is None or pd.isna(h):
        return "-"
    h = h % 24
    return f"{int(h):02d}:{int(round((h % 1) * 60)) % 60:02d}"


def change_text(signal, now, base):
    """Plain-language description of how a signal differs from personal normal."""
    if pd.isna(now) or pd.isna(base):
        return "not enough data"
    if signal == "sleep_onset":
        mins = (now - base) * 60
        return f"{abs(mins):.0f} min {'later' if mins > 0 else 'earlier'} than usual"
    if signal == "sleep_hours":
        mins = (now - base) * 60
        return f"{abs(mins):.0f} min {'more' if mins > 0 else 'less'} sleep than usual"
    pct = 100 * (now - base) / base if base else 0
    return f"{abs(pct):.0f}% {'above' if pct > 0 else 'below'} usual"


class DigitalTwin:
    def __init__(self, patient_id, assets=None, frames=None):
        """frames: optional dict(patient=Series, daily=DataFrame, enc=DataFrame) for a person
        who is not in the stored cohort (e.g. a client enrolled live in the app)."""
        a = assets or load_assets()
        self.a = a
        self.pid = patient_id
        if frames is not None:
            self.patient = frames["patient"]
            self.daily = frames["daily"].sort_values("day").reset_index(drop=True)
            self.enc = frames["enc"].sort_values("day") if len(frames["enc"]) else a["enc"].iloc[0:0]
        else:
            self.patient = a["patients"].set_index("patient_id").loc[patient_id]
            self.daily = a["daily"][a["daily"].patient_id == patient_id].sort_values("day").reset_index(drop=True)
            self.enc = a["enc"][a["enc"].patient_id == patient_id].sort_values("day")
        self.cols = a["meta"]["twin_features"]
        self.threshold = a["meta"]["threshold"]
        self.base_rate = a["meta"]["base_rate"]
        self.feats = F.build_patient_features(self.daily, self.enc, dict(self.patient) | {"patient_id": patient_id})
        p = a["model"].predict_proba(self.feats[self.cols])[:, 1]
        p[self.feats["day"].values < C.MIN_HISTORY_DAYS] = np.nan
        self.feats["risk"] = p

    # ------------------------------------------------------------------ helpers
    @property
    def n_days(self):
        return len(self.daily)

    def band(self, r, prev=None):
        """Alert band with hysteresis: a single-day jump from a quiet baseline is 'Watch',
        not 'Early warning', unless the estimate is well above the line (1.5x)."""
        if r is None or pd.isna(r):
            return "Learning baseline"
        t = self.threshold
        if r >= t:
            if prev is None or pd.isna(prev) or prev >= t / 2 or r >= 1.5 * t:
                return "Early warning"
            return "Watch"
        if r >= t / 2:
            return "Watch"
        return "Stable"

    def prev_risk(self, day):
        if day <= 0:
            return np.nan
        return float(self.feats.loc[self.feats.day == day - 1, "risk"].iloc[0])

    def ehr_until(self, day):
        """EHR events visible to the system up to `day` (ground-truth rows hidden)."""
        e = self.enc[(self.enc.day <= day) & (self.enc.type.isin(F.EHR_EVENT_TYPES))]
        return e[["date", "type", "code", "detail", "phq9"]]

    # ------------------------------------------------------------------ 1. mirror
    def state(self, day):
        row = self.feats.loc[self.feats.day == day].iloc[0]
        sig = []
        for s in C.SIGNALS:
            sig.append(dict(signal=s, label=C.SIGNAL_LABELS[s], baseline=row[f"base_{s}"],
                            current_7d=row[f"raw7_{s}"], z=row[f"z7_{s}"],
                            change=change_text(s, row[f"raw7_{s}"], row[f"base_{s}"])))
        prev = self.prev_risk(day)
        r = row["risk"]
        band = self.band(r, prev)
        margin = (r / self.threshold - 1) if pd.notna(r) else np.nan
        return dict(day=day, date=self.daily.loc[day, "date"], risk=r, band=band, prev_risk=prev,
                    rising=bool(pd.notna(r) and r >= self.threshold and band == "Watch"),
                    borderline=bool(pd.notna(margin) and abs(margin) <= 0.2),
                    relative_risk=(row["risk"] / self.base_rate) if pd.notna(row["risk"]) else np.nan,
                    deviation_index=row["twin_deviation_index"], signals=pd.DataFrame(sig),
                    stressor=bool(row["known_stressor_period"]))

    # ------------------------------------------------------------------ 2. predict
    def risk_series(self):
        return self.feats[["day", "risk", "twin_deviation_index"]].assign(
            date=self.daily["date"].values, in_episode=self.daily["gt_in_episode"].values)

    # ------------------------------------------------------------------ 3. explain
    def explain(self, day, k=6, feats=None):
        f = self.feats if feats is None else feats
        x = f.loc[f.day == day, self.cols]
        sv = self.a["explainer"].shap_values(x)[0]
        df = pd.DataFrame({"feature": self.cols, "value": x.values[0], "shap": sv})
        df["driver"] = df["feature"].map(readable)
        row = f.loc[f.day == day].iloc[0]

        def detail(feat):
            for pre in ("z7_", "z3_", "dz7_", "raw7_"):
                if feat.startswith(pre):
                    s = feat[len(pre):]
                    return change_text(s, row[f"raw7_{s}"], row[f"base_{s}"])
            v = row[feat]
            if pd.isna(v):
                return "no record"
            if feat in ("known_stressor_period", "on_antidepressant", "family_history_depression",
                        "hypothyroidism", "anaemia", "vitamin_d_deficiency", "is_student", "is_female"):
                return "yes" if v else "no"
            if feat == "phq9_change":
                return f"{v:+.0f} points at last screening"
            return f"{v:.2f}".rstrip("0").rstrip(".")
        df["what_changed"] = df["feature"].map(detail)
        return df.sort_values("shap", ascending=False).head(k).reset_index(drop=True)

    # ------------------------------------------------------------------ 4. simulate
    def what_if(self, day, sleep_hours=0.0, sleep_onset_shift_h=0.0, steps=0.0,
                night_screen_min=0.0, social_contacts=0.0, window=7):
        """Apply behaviour changes to the last `window` days and re-run the twin.

        NOTE: this is a model-based sensitivity analysis on the twin, not a causal
        guarantee. It shows how the risk estimate responds if these signals had
        looked different.
        """
        d = self.daily.copy()
        m = (d.day > day - window) & (d.day <= day)
        d.loc[m, "sleep_hours"] = (d.loc[m, "sleep_hours"] + sleep_hours).clip(3, 12.5)
        d.loc[m, "sleep_onset"] = d.loc[m, "sleep_onset"] + sleep_onset_shift_h
        d.loc[m, "steps"] = (d.loc[m, "steps"] + steps).clip(lower=0)
        d.loc[m, "night_screen_min"] = (d.loc[m, "night_screen_min"] + night_screen_min).clip(lower=0)
        d.loc[m, "social_contacts"] = (d.loc[m, "social_contacts"] + social_contacts).clip(lower=0)
        f = F.build_patient_features(d, self.enc, dict(self.patient) | {"patient_id": self.pid})
        r_new = float(self.a["model"].predict_proba(f.loc[f.day == day, self.cols])[:, 1][0])
        r_old = float(self.feats.loc[self.feats.day == day, "risk"].iloc[0])
        prev = self.prev_risk(day)
        return dict(before=r_old, after=r_new, band_before=self.band(r_old, prev), band_after=self.band(r_new, prev),
                    drivers_after=self.explain(day, feats=f), feats_after=f)

"""
OPTIONAL / EXPERIMENTAL - real-data validation hook.

DEPRESJON (Garcia-Ceja et al., ACM MMSys 2018, Simula) contains minute-level
wrist actigraphy from 23 people with unipolar/bipolar depression and 32 controls,
plus MADRS depression scores. It has no EHR and no labelled episode onsets, so it
cannot train the full twin - but it can check that the twin's *personal-baseline
activity features* separate depressed from non-depressed people on real data.

How to use
  1. Download DEPRESJON from https://datasets.simula.no/depresjon/ (check its
     licence terms) and unzip it to  data/depresjon/  so you have:
        data/depresjon/condition/condition_1.csv ...
        data/depresjon/control/control_1.csv ...
        data/depresjon/scores.csv
  2. Run:  python -m src.depresjon_loader

The dataset is NOT redistributed in this repository.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C

ROOT = C.DATA_DIR / "depresjon"


def daily_features(minute_df):
    """Aggregate minute-level activity counts into daily twin-style signals."""
    df = minute_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    g = df.groupby("date")
    out = pd.DataFrame({
        "total_activity": g["activity"].sum(),
        "active_minutes": g["activity"].apply(lambda a: int((a > 100).sum())),
        "night_activity": df[df.hour < 6].groupby("date")["activity"].sum(),
        "rest_minutes": g["activity"].apply(lambda a: int((a == 0).sum())),
    }).fillna(0)
    # keep only complete days (1440 minutes)
    out = out[g.size() >= 1440 * 0.95]
    return out.reset_index()


def load(root=ROOT):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Put the DEPRESJON files in {root} (see module docstring).")
    rows = []
    for group in ["condition", "control"]:
        for f in sorted((root / group).glob("*.csv")):
            d = daily_features(pd.read_csv(f))
            d["person"] = f.stem
            d["depressed"] = int(group == "condition")
            rows.append(d)
    data = pd.concat(rows, ignore_index=True)
    scores_path = root / "scores.csv"
    scores = pd.read_csv(scores_path) if scores_path.exists() else None
    return data, scores


def personal_baseline_check(data):
    """Within-person variability and level of activity: depressed vs control."""
    s = data.groupby(["person", "depressed"]).agg(
        mean_activity=("total_activity", "mean"),
        cv_activity=("total_activity", lambda x: x.std() / max(x.mean(), 1)),
        night_share=("night_activity", "sum"),
        total=("total_activity", "sum")).reset_index()
    s["night_share"] = s["night_share"] / s["total"].clip(lower=1)
    return s.groupby("depressed")[["mean_activity", "cv_activity", "night_share"]].agg(["mean", "std"])


if __name__ == "__main__":
    data, scores = load()
    print(f"{data.person.nunique()} people, {len(data)} complete days")
    print(personal_baseline_check(data))

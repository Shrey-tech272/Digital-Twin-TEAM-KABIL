"""
Synthetic cohort generator for Manas Twin.

WHY SYNTHETIC?
No public dataset pairs daily wearable data with longitudinal health records for
Indian students and young professionals. To build and evaluate the twin end-to-end
we generate a *synthetic* cohort whose behaviour follows patterns reported in the
digital-phenotyping literature (sleep disruption, delayed sleep onset, lower HRV,
higher resting HR, reduced activity and social contact, more late-night phone use
before and during depressive episodes).

Everything produced here is SYNTHETIC. No real person is represented.
A loader for the real DEPRESJON actigraphy dataset is provided in
`src/depresjon_loader.py` so the pipeline can be validated on real data.

MODEL OF THE WORLD (ground truth, hidden from the twin)
  * Each person has a latent "depressive load" L(t) that follows a mean-reverting
    process driven by personal vulnerability, prior episodes, calendar stressors
    (exams for students, quarter-end crunch for professionals) and random life events.
  * A depressive episode starts when L crosses a threshold and lasts >= 14 days
    (mirrors the 2-week duration criterion for a major depressive episode).
  * Wearable signals drift away from the person's own baseline in proportion to L,
    so the build-up (prodrome) is visible in the data days before the episode starts.
  * EHR events: periodic PHQ-9 screenings, delayed diagnosis, antidepressant start,
    therapy referral. Treatment slowly lowers L.

Run:  python -m src.simulator
"""
import json

import numpy as np
import pandas as pd

from src import config as C


# ---------------------------------------------------------------------------
# Calendar stressors (known in advance - academic and corporate calendars)
# ---------------------------------------------------------------------------
def stressor_calendar(cohort, n_days, rng):
    """Return a 0/1 array marking predictable high-stress periods."""
    flag = np.zeros(n_days)
    if cohort == "student":
        # mid-semester exams (~mid Feb) and end-semester exams (~late Apr - mid May)
        windows = [(40, 52), (110, 132), (205, 215)]
    else:
        # quarter-end crunch (late Mar, late Jun) and appraisal cycle (Apr)
        windows = [(75, 90), (95, 105), (165, 181)]
    for a, b in windows:
        shift = rng.integers(-4, 5)
        flag[max(0, a + shift):min(n_days, b + shift)] = 1
    return flag


# ---------------------------------------------------------------------------
# One person
# ---------------------------------------------------------------------------
def simulate_person(pid, rng, n_days=C.N_DAYS, overrides=None):
    o = overrides or {}
    cohort = o.get("cohort", rng.choice(["student", "professional"]))
    student = cohort == "student"
    sex = o.get("sex", rng.choice(["F", "M"], p=[0.48, 0.52]))
    age = int(o.get("age", rng.integers(18, 25) if student else rng.integers(23, 36)))
    city = o.get("city", rng.choice(C.CITIES))

    vuln = float(o.get("vulnerability", rng.beta(2, 5)))    # 0..1 personal vulnerability
    family_history = bool(o.get("family_history_depression", rng.random() < 0.10 + 0.35 * vuln))
    prior_episodes = int(o.get("prior_episodes", rng.poisson(0.2 + 2.0 * vuln)))
    hypothyroid = bool(rng.random() < (0.06 if sex == "F" else 0.02))
    anaemia = bool(rng.random() < (0.25 if sex == "F" else 0.06))
    vit_d_def = bool(rng.random() < 0.35)
    vuln_eff = min(1.0, vuln + 0.05 * hypothyroid + 0.04 * anaemia)
    sleep_type = "hypersomnia" if rng.random() < 0.25 else "insomnia"
    reactivity = float(rng.lognormal(0, 0.25))              # how strongly the body shows it

    base = {
        "sleep_hours": rng.normal(6.9 if student else 6.6, 0.55),
        "sleep_onset": rng.normal(24.8 if student else 23.9, 0.7),
        "sleep_efficiency": rng.normal(88, 3),
        "resting_hr": rng.normal(66 if sex == "F" else 63, 6),
        "hrv_rmssd": max(18.0, rng.normal(52 - 0.6 * (age - 18), 12)),
        "steps": max(2000.0, rng.normal(7000 if student else 5800, 2000)),
        "night_screen_min": max(10.0, rng.normal(95 if student else 70, 30)),
        "social_contacts": max(4.0, rng.normal(28 if student else 22, 8)),
    }

    stress_flag = stressor_calendar(cohort, n_days, rng)
    dates = pd.date_range(C.START_DATE, periods=n_days, freq="D")
    weekend = (dates.dayofweek >= 5).astype(float)

    # Mean depressive load for this person
    mu = 0.14 + 0.57 * vuln_eff + 0.06 * min(prior_episodes, 3) + 0.08 * family_history

    L = mu + rng.normal(0, 0.05)
    event_load = 0.0
    in_ep, ep_len, ep_id, n_eps = False, 0, 0, 0
    diagnosed, on_med, med_day = False, False, None
    rows, encounters = [], []
    next_screen = int(rng.integers(5, 30))

    for t in range(n_days):
        # random life events (breakup, family illness, job loss, bereavement ...)
        if rng.random() < 1 / 140:
            event_load += rng.uniform(0.35, 0.9) * (0.6 + vuln_eff)
        event_load *= 0.93

        target = mu + 0.32 * stress_flag[t] * (0.5 + vuln_eff) + event_load
        if in_ep:
            target += 0.30                                   # episodes are self-sustaining
        if on_med and med_day is not None and t - med_day >= 14:
            target -= 0.45                                   # treatment effect after ~2 weeks
        L = 0.92 * L + 0.08 * target + rng.normal(0, 0.055)

        # --- episode state machine ---
        onset_today = False
        if not in_ep and L > 1.0:
            in_ep, ep_len, onset_today = True, 0, True
            n_eps += 1
            ep_id = n_eps
            diagnosed = False
        if in_ep:
            ep_len += 1
            # help-seeking is slow - diagnosis may come late or not at all
            if not diagnosed and ep_len >= 7 and rng.random() < 0.06:
                diagnosed = True
                phq = int(np.clip(round(4 + 11 * L + rng.normal(0, 2)), 0, 27))
                encounters.append(dict(patient_id=pid, day=t, type="diagnosis",
                                       code="F32", detail="Depressive episode", phq9=phq))
                if not on_med and rng.random() < 0.65:
                    on_med, med_day = True, t
                    encounters.append(dict(patient_id=pid, day=t, type="medication_start",
                                           code="N06AB10", detail="Escitalopram 10 mg OD", phq9=np.nan))
                if rng.random() < 0.5:
                    encounters.append(dict(patient_id=pid, day=t, type="therapy_referral",
                                           code="CBT", detail="Referred for CBT", phq9=np.nan))
            if ep_len >= 14 and L < 0.78:
                in_ep = False
                ep_id_out = ep_id
                encounters.append(dict(patient_id=pid, day=t, type="episode_resolved_gt",
                                       code="", detail=f"(ground truth) episode {ep_id_out} ended",
                                       phq9=np.nan))

        # --- scheduled PHQ-9 screening (campus wellness / corporate EAP check-in) ---
        if t == next_screen:
            if rng.random() < 0.7:
                phq = int(np.clip(round(3 + 11 * L + rng.normal(0, 2)), 0, 27))
                encounters.append(dict(patient_id=pid, day=t, type="phq9_screening",
                                       code="44261-6", detail="PHQ-9 total score", phq9=phq))
            next_screen = t + int(rng.integers(25, 36))

        # --- wearable / phone signals ---
        Le = max(L - 0.15, 0.0) * reactivity
        sf, wk = stress_flag[t], weekend[t]
        sleep_delta = (-0.9 * Le) if sleep_type == "insomnia" else (1.1 * Le)
        sig = {
            "sleep_hours": base["sleep_hours"] + sleep_delta + 0.6 * wk - 0.3 * sf + rng.normal(0, 0.6),
            "sleep_onset": base["sleep_onset"] + 0.8 * Le + 0.6 * wk + 0.2 * sf + rng.normal(0, 0.5),
            "sleep_efficiency": base["sleep_efficiency"] - 6 * Le - 1.0 * sf + rng.normal(0, 2.5),
            "resting_hr": base["resting_hr"] + 4.5 * Le + 1.0 * sf + rng.normal(0, 2.0),
            "hrv_rmssd": base["hrv_rmssd"] * (1 - 0.22 * Le - 0.04 * sf) + rng.normal(0, 0.12 * base["hrv_rmssd"]),
            "steps": base["steps"] * (1 - 0.35 * Le) - (1000 if student else 800) * wk + rng.normal(0, 0.25 * base["steps"]),
            "night_screen_min": base["night_screen_min"] * (1 + 0.45 * Le) + 15 * sf + rng.normal(0, 25),
            "social_contacts": base["social_contacts"] * (1 - 0.40 * Le) + rng.normal(0, 4),
        }
        sig["sleep_hours"] = float(np.clip(sig["sleep_hours"], 3, 12.5))
        sig["sleep_efficiency"] = float(np.clip(sig["sleep_efficiency"], 55, 98))
        sig["hrv_rmssd"] = float(max(8.0, sig["hrv_rmssd"]))
        sig["steps"] = float(max(300, round(sig["steps"])))
        sig["night_screen_min"] = float(max(0, round(sig["night_screen_min"])))
        sig["social_contacts"] = float(max(0, round(sig["social_contacts"])))

        # device not worn / sync gaps - more common when low
        if rng.random() < min(0.4, 0.04 + 0.08 * Le):
            sig = {k: np.nan for k in sig}

        rows.append(dict(patient_id=pid, day=t, date=dates[t].date().isoformat(),
                         known_stressor_period=int(sf), **sig,
                         gt_latent_load=round(L, 4), gt_in_episode=int(in_ep),
                         gt_episode_id=ep_id if in_ep else 0, gt_onset=int(onset_today)))

    patient = dict(patient_id=pid, cohort=cohort, age=age, sex=sex, city=city,
                   prior_episodes=prior_episodes, family_history_depression=family_history,
                   hypothyroidism=hypothyroid, anaemia=anaemia, vitamin_d_deficiency=vit_d_def,
                   gt_vulnerability=round(vuln, 3), gt_sleep_response=sleep_type,
                   gt_reactivity=round(reactivity, 3))
    return patient, rows, encounters


def simulate_custom(pid, seed, n_days=C.N_DAYS, **overrides):
    """Simulate one person with chosen characteristics (used to enrol a demo client in the app).
    overrides: cohort, age, sex, city, prior_episodes, family_history_depression, vulnerability (0-1)."""
    rng = np.random.default_rng(seed)
    p, rows, enc = simulate_person(pid, rng, n_days, overrides=overrides)
    patient = pd.Series(p)
    daily = pd.DataFrame(rows)
    enc = pd.DataFrame(enc, columns=["patient_id", "day", "type", "code", "detail", "phq9"])
    enc["date"] = (pd.Timestamp(C.START_DATE) + pd.to_timedelta(enc["day"], unit="D")).dt.date.astype(str) if len(enc) else []
    return patient, daily, enc


def generate(n_patients=C.N_PATIENTS, n_days=C.N_DAYS, seed=C.SEED):
    rng = np.random.default_rng(seed)
    patients, daily, enc = [], [], []
    for i in range(n_patients):
        pid = f"MT{i + 1:04d}"
        p, r, e = simulate_person(pid, rng, n_days)
        patients.append(p)
        daily.extend(r)
        enc.extend(e)
    patients = pd.DataFrame(patients)
    daily = pd.DataFrame(daily)
    enc = pd.DataFrame(enc)
    enc["date"] = (pd.Timestamp(C.START_DATE) + pd.to_timedelta(enc["day"], unit="D")).dt.date.astype(str)
    return patients, daily, enc


def main():
    C.DATA_DIR.mkdir(exist_ok=True, parents=True)
    patients, daily, enc = generate()
    patients.to_csv(C.DATA_DIR / "patients.csv", index=False)
    daily.to_csv(C.DATA_DIR / "daily_wearables.csv.gz", index=False, compression="gzip")
    enc.to_csv(C.DATA_DIR / "encounters.csv", index=False)

    n_with_ep = daily.groupby("patient_id")["gt_onset"].max().sum()
    summary = {
        "patients": int(len(patients)),
        "days_per_patient": int(daily.groupby("patient_id").size().iloc[0]),
        "person_days": int(len(daily)),
        "episodes": int(daily["gt_onset"].sum()),
        "patients_with_episode": int(n_with_ep),
        "pct_patients_with_episode": round(100 * n_with_ep / len(patients), 1),
        "missing_day_rate_pct": round(100 * daily["sleep_hours"].isna().mean(), 1),
        "phq9_screenings": int((enc["type"] == "phq9_screening").sum()),
        "diagnoses": int((enc["type"] == "diagnosis").sum()),
    }
    (C.DATA_DIR / "cohort_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

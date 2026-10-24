"""
Export a patient's twin record as an HL7 FHIR R4-style Bundle (JSON).

Shows how the twin would plug into India's digital health ecosystem
(ABDM health records are exchanged as FHIR bundles). Codes used:
  * ICD-10 F32       - depressive episode
  * LOINC 44261-6    - PHQ-9 total score
  * LOINC 8867-4     - heart rate
  * ATC N06AB10      - escitalopram
Other wearable metrics use a local code system (urn:manas-twin:signal).

Run:  python -m src.fhir MT0441 120
"""
import json
import sys

import pandas as pd

from src import config as C
from src.features import EHR_EVENT_TYPES


def _obs(pid, date, system, code, display, value, unit):
    return {"resource": {
        "resourceType": "Observation", "status": "final",
        "subject": {"reference": f"Patient/{pid}"}, "effectiveDateTime": date,
        "code": {"coding": [{"system": system, "code": code, "display": display}]},
        "valueQuantity": {"value": round(float(value), 2), "unit": unit}}}


def build_bundle(pid, day, patients, daily, enc, wearable_days=7):
    p = patients.set_index("patient_id").loc[pid]
    entries = [{"resource": {
        "resourceType": "Patient", "id": pid, "gender": "female" if p["sex"] == "F" else "male",
        "extension": [{"url": "urn:manas-twin:cohort", "valueString": p["cohort"]},
                      {"url": "urn:manas-twin:age", "valueInteger": int(p["age"])}],
        "address": [{"city": p["city"], "country": "IN"}]}}]
    e = enc[(enc.patient_id == pid) & (enc.day <= day) & enc.type.isin(EHR_EVENT_TYPES)]
    for _, r in e.iterrows():
        if r["type"] in ("phq9_screening", "diagnosis") and pd.notna(r["phq9"]):
            entries.append(_obs(pid, r["date"], "http://loinc.org", "44261-6",
                                "PHQ-9 total score", r["phq9"], "{score}"))
        if r["type"] == "diagnosis":
            entries.append({"resource": {
                "resourceType": "Condition", "subject": {"reference": f"Patient/{pid}"},
                "onsetDateTime": r["date"],
                "code": {"coding": [{"system": "http://hl7.org/fhir/sid/icd-10", "code": "F32",
                                     "display": "Depressive episode"}]}}})
        if r["type"] == "medication_start":
            entries.append({"resource": {
                "resourceType": "MedicationStatement", "status": "active",
                "subject": {"reference": f"Patient/{pid}"}, "effectiveDateTime": r["date"],
                "medicationCodeableConcept": {"coding": [{"system": "http://www.whocc.no/atc",
                                                          "code": "N06AB10", "display": r["detail"]}]}}})
    w = daily[(daily.patient_id == pid) & (daily.day > day - wearable_days) & (daily.day <= day)]
    for _, r in w.dropna(subset=["resting_hr"]).iterrows():
        entries.append(_obs(pid, r["date"], "http://loinc.org", "8867-4", "Heart rate (resting)",
                            r["resting_hr"], "/min"))
        for s in ["sleep_hours", "hrv_rmssd", "steps", "night_screen_min"]:
            entries.append(_obs(pid, r["date"], "urn:manas-twin:signal", s, C.SIGNAL_LABELS[s],
                                r[s], C.SIGNAL_UNITS[s]))
    return {"resourceType": "Bundle", "type": "collection", "entry": entries}


if __name__ == "__main__":
    pid = sys.argv[1] if len(sys.argv) > 1 else "MT0441"
    day = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    P = pd.read_csv(C.DATA_DIR / "patients.csv")
    D = pd.read_csv(C.DATA_DIR / "daily_wearables.csv.gz")
    E = pd.read_csv(C.DATA_DIR / "encounters.csv")
    out = C.DATA_DIR / "fhir_samples"
    out.mkdir(exist_ok=True)
    b = build_bundle(pid, day, P, D, E)
    (out / f"{pid}_day{day}.json").write_text(json.dumps(b, indent=2))
    print(f"Wrote {len(b['entry'])} FHIR resources to {out / f'{pid}_day{day}.json'}")

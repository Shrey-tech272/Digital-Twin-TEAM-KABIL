"""Central configuration for Manas Twin."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

SEED = 42
N_PATIENTS = 600          # synthetic cohort size
N_DAYS = 240              # days of follow-up per person (~8 months)
START_DATE = "2025-01-01"

HORIZON_DAYS = 7          # predict onset of a depressive episode within the next 7 days
BASELINE_WINDOW = 45      # days used to learn each person's "normal"
BASELINE_GAP = 14         # baseline excludes the most recent 14 days
MIN_HISTORY_DAYS = 60     # twin needs this much history before it predicts

# Daily wearable / phone signals the twin tracks
SIGNALS = [
    "sleep_hours",        # total sleep time (h)
    "sleep_onset",        # clock hour of sleep onset; 24.5 = 00:30 next day
    "sleep_efficiency",   # % time asleep while in bed
    "resting_hr",         # resting heart rate (bpm)
    "hrv_rmssd",          # heart-rate variability, RMSSD (ms)
    "steps",              # daily steps
    "night_screen_min",   # phone screen time 23:00-05:00 (min)
    "social_contacts",    # calls + messages with distinct contacts
]

# Direction in which each signal typically moves before/during a depressive episode
# (+1 = goes up, -1 = goes down, 0 = can go either way -> use magnitude)
RISK_DIRECTION = {
    "sleep_hours": 0,
    "sleep_onset": +1,
    "sleep_efficiency": -1,
    "resting_hr": +1,
    "hrv_rmssd": -1,
    "steps": -1,
    "night_screen_min": +1,
    "social_contacts": -1,
}

SIGNAL_LABELS = {
    "sleep_hours": "Sleep duration",
    "sleep_onset": "Sleep onset time",
    "sleep_efficiency": "Sleep efficiency",
    "resting_hr": "Resting heart rate",
    "hrv_rmssd": "Heart-rate variability (HRV)",
    "steps": "Daily steps",
    "night_screen_min": "Late-night screen time",
    "social_contacts": "Social contacts",
}

SIGNAL_UNITS = {
    "sleep_hours": "h", "sleep_onset": "clock", "sleep_efficiency": "%",
    "resting_hr": "bpm", "hrv_rmssd": "ms", "steps": "steps",
    "night_screen_min": "min", "social_contacts": "contacts",
}

CITIES = ["Pune", "Bengaluru", "Mumbai", "Hyderabad", "Delhi NCR", "Chennai",
          "Kolkata", "Ahmedabad", "Jaipur", "Kochi", "Indore", "Lucknow"]

# Tele-MANAS: Government of India's national tele-mental-health helpline
HELPLINE = "Tele-MANAS 14416 (toll-free, 24x7)"

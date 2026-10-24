"""
Manas Twin - clinician-facing dashboard.

Run from the repository root:
    streamlit run app/streamlit_app.py
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src.fhir import build_bundle  # noqa: E402
from src.simulator import simulate_custom  # noqa: E402
from src.train import readable  # noqa: E402
from src.twin import DigitalTwin, fmt_clock, load_assets  # noqa: E402

st.set_page_config(page_title="Manas Twin", page_icon=":material/spa:", layout="wide",
                   initial_sidebar_state="expanded")

# =============================================================================
# Theme
# =============================================================================
PALETTES = {
    "light": dict(bg="#F6F2EE", surface="#FFFFFF", surface2="#F3EEE9", border="#ECE5DF", ink="#24234A",
                  muted="#8B889B", grid="#EFEAE5", navy1="#2D2C5A", navy2="#4B4985", violet="#5B57A6",
                  lav="#6F6AB8", lav_bg="#ECE9F7", rose="#C9646F", rose_bg="#FBEAEA", sky="#5A8DC2",
                  sky_bg="#E7EFF8", peach="#C98545", peach_bg="#FBEEE1", mint="#5C9468", mint_bg="#E8F1EA",
                  shadow="0 1px 2px rgba(36,35,74,.04), 0 10px 28px rgba(36,35,74,.05)"),
    "dark": dict(bg="#11111F", surface="#1A1A2E", surface2="#222238", border="#2B2B45", ink="#ECEAF7",
                 muted="#9B99B4", grid="#26263D", navy1="#2B2A63", navy2="#403D8C", violet="#8F8AE0",
                 lav="#A29DEB", lav_bg="rgba(162,157,235,.13)", rose="#EE9AA3", rose_bg="rgba(238,154,163,.13)",
                 sky="#8DB7E6", sky_bg="rgba(141,183,230,.12)", peach="#EDB37E", peach_bg="rgba(237,179,126,.13)",
                 mint="#8FCB9C", mint_bg="rgba(143,203,156,.12)", shadow="0 1px 2px rgba(0,0,0,.35)"),
}
if "dark" not in st.session_state:
    st.session_state.dark = False
MODE = "dark" if st.session_state.dark else "light"
P = PALETTES[MODE]

STATUS = {
    "Stable": dict(sky="Clear skies", icon="sun", fg=P["mint"], bg=P["mint_bg"], ring="#A8D5B3",
                   plain="Rhythm is close to their usual self."),
    "Watch": dict(sky="Clouds gathering", icon="cloud-sun", fg=P["peach"], bg=P["peach_bg"], ring="#F2C79B",
                  plain="Some signals are drifting away from their normal."),
    "Early warning": dict(sky="Storm likely", icon="cloud-lightning", fg=P["rose"], bg=P["rose_bg"], ring="#F2A3AB",
                          plain="Strong drift: a low-mood episode may be close."),
    "Learning baseline": dict(sky="Still learning", icon="sprout", fg=P["muted"], bg=P["surface2"],
                              ring="rgba(255,255,255,.5)", plain="The twin needs 60 days to learn what normal looks like."),
}
ACTIONS = {
    "Stable": "No action suggested. Routine check-ins as usual.",
    "Watch": "Suggested: a light-touch check-in; review sleep, activity and stressors with the client.",
    "Early warning": "Suggested: contact within 48 hours and administer a PHQ-9. Clinical judgement decides next steps.",
    "Learning baseline": "Keep collecting data so the twin can learn this client's normal.",
}

# Line icons (Lucide, ISC licence) as inline SVG paths
ICONS = {
    "sun": "<circle cx='12' cy='12' r='4'/><path d='M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41'/>",
    "cloud-sun": "<path d='M12 2v2'/><path d='m4.93 4.93 1.41 1.41'/><path d='M20 12h2'/><path d='m19.07 4.93-1.41 1.41'/><path d='M15.947 12.65a4 4 0 0 0-5.925-4.128'/><path d='M13 22H7a5 5 0 1 1 4.9-6H13a3 3 0 0 1 0 6Z'/>",
    "cloud-lightning": "<path d='M6 16.326A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 .5 8.973'/><path d='m13 12-3 5h4l-3 5'/>",
    "sprout": "<path d='M7 20h10'/><path d='M10 20c5.5-2.5.8-6.4 3-10'/><path d='M9.5 9.4c1.1.8 1.8 2.2 2.3 3.7-2 .4-3.5.4-4.8-.3-1.2-.6-2.3-1.9-3-4.2 2.8-.5 4.4 0 5.5.8z'/><path d='M14.1 6a7 7 0 0 0-1.1 4c1.9-.1 3.3-.6 4.3-1.4 1-1 1.6-2.3 1.7-4.6-2.7.1-4 1-4.9 2z'/>",
    "moon": "<path d='M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z'/>",
    "bed": "<path d='M2 4v16'/><path d='M2 8h18a2 2 0 0 1 2 2v10'/><path d='M2 17h20'/><path d='M6 8v9'/>",
    "heart": "<path d='M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z'/>",
    "activity": "<path d='M22 12h-4l-3 9L9 3l-3 9H2'/>",
    "footprints": "<path d='M4 16v-2.38C4 11.5 2.97 10.5 3 8c.03-2.72 1.49-6 4.5-6C9.37 2 10 3.8 10 5.5c0 3.11-2 5.66-2 8.68V16a2 2 0 1 1-4 0Z'/><path d='M20 20v-2.38c0-2.12 1.03-3.12 1-5.62-.03-2.72-1.49-6-4.5-6C14.63 6 14 7.8 14 9.5c0 3.11 2 5.66 2 8.68V20a2 2 0 1 0 4 0Z'/><path d='M16 17h4'/><path d='M4 13h4'/>",
    "phone": "<rect width='14' height='20' x='5' y='2' rx='2' ry='2'/><path d='M12 18h.01'/>",
    "message": "<path d='M7.9 20A9 9 0 1 0 4 16.1L2 22Z'/>",
    "clipboard": "<rect width='8' height='4' x='8' y='2' rx='1' ry='1'/><path d='M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2'/>",
    "stethoscope": "<path d='M11 2v2'/><path d='M5 2v2'/><path d='M5 3H4a2 2 0 0 0-2 2v4a6 6 0 0 0 12 0V5a2 2 0 0 0-2-2h-1'/><path d='M8 15a6 6 0 0 0 12 0v-3'/><circle cx='20' cy='10' r='2'/>",
    "pill": "<path d='m10.5 20.5 10-10a4.95 4.95 0 1 0-7-7l-10 10a4.95 4.95 0 1 0 7 7Z'/><path d='m8.5 8.5 7 7'/>",
    "chat": "<path d='M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z'/>",
    "book": "<path d='M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z'/><path d='M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z'/>",
    "users": "<path d='M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2'/><circle cx='9' cy='7' r='4'/><path d='M22 21v-2a4 4 0 0 0-3-3.87'/><path d='M16 3.13a4 4 0 0 1 0 7.75'/>",
    "compass": "<circle cx='12' cy='12' r='10'/><path d='m16.24 7.76-2.12 6.36-6.36 2.12 2.12-6.36 6.36-2.12z'/>",
    "watch": "<circle cx='12' cy='12' r='6'/><polyline points='12 10 12 12 13 13'/><path d='m16.13 7.66-.81-4.05a2 2 0 0 0-2-1.61h-2.68a2 2 0 0 0-2 1.61l-.78 4.05'/><path d='m7.88 16.36.8 4a2 2 0 0 0 2 1.61h2.72a2 2 0 0 0 2-1.61l.81-4.05'/>",
    "shield": "<path d='M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10'/>",
    "lock": "<rect width='18' height='11' x='3' y='11' rx='2' ry='2'/><path d='M7 11V7a5 5 0 0 1 10 0v4'/>",
    "ban": "<circle cx='12' cy='12' r='10'/><path d='m4.9 4.9 14.2 14.2'/>",
    "eye-off": "<path d='M9.88 9.88a3 3 0 1 0 4.24 4.24'/><path d='M10.73 5.08A10.43 10.43 0 0 1 12 5c7 0 10 7 10 7a13.16 13.16 0 0 1-1.67 2.68'/><path d='M6.61 6.61A13.526 13.526 0 0 0 2 12s3 7 10 7a9.74 9.74 0 0 0 5.39-1.61'/><line x1='2' x2='22' y1='2' y2='22'/>",
    "trend": "<polyline points='22 7 13.5 15.5 8.5 10.5 2 17'/><polyline points='16 7 22 7 22 13'/>",
    "bulb": "<path d='M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5'/><path d='M9 18h6'/><path d='M10 22h4'/>",
    "sliders": "<line x1='4' x2='4' y1='21' y2='14'/><line x1='4' x2='4' y1='10' y2='3'/><line x1='12' x2='12' y1='21' y2='12'/><line x1='12' x2='12' y1='8' y2='3'/><line x1='20' x2='20' y1='21' y2='16'/><line x1='20' x2='20' y1='12' y2='3'/><line x1='2' x2='6' y1='14' y2='14'/><line x1='10' x2='14' y1='8' y2='8'/><line x1='18' x2='22' y1='16' y2='16'/>",
    "check": "<path d='M22 11.08V12a10 10 0 1 1-5.93-9.14'/><polyline points='22 4 12 14.01 9 11.01'/>",
    "alert": "<path d='m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z'/><path d='M12 9v4'/><path d='M12 17h.01'/>",
    "info": "<circle cx='12' cy='12' r='10'/><path d='M12 16v-4'/><path d='M12 8h.01'/>",
    "call": "<path d='M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z'/>",
    "calendar": "<rect width='18' height='18' x='3' y='4' rx='2' ry='2'/><line x1='16' x2='16' y1='2' y2='6'/><line x1='8' x2='8' y1='2' y2='6'/><line x1='3' x2='21' y1='10' y2='10'/>",
    "history": "<path d='M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8'/><path d='M3 3v5h5'/><path d='M12 7v5l4 2'/>",
}


def icon(name, size=18, color="currentColor", sw=1.7):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' "
            f"stroke='{color}' stroke-width='{sw}' stroke-linecap='round' stroke-linejoin='round' "
            f"style='flex-shrink:0;vertical-align:-0.18em'>{ICONS[name]}</svg>")


CSS_VARS = f"""<style>
:root {{ --bg:{P['bg']}; --surface:{P['surface']}; --surface2:{P['surface2']}; --border:{P['border']};
  --ink:{P['ink']}; --muted:{P['muted']}; --navy1:{P['navy1']}; --navy2:{P['navy2']}; --violet:{P['violet']};
  --shadow:{P['shadow']}; }}
</style>"""

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,500&family=Inter:wght@400;500;600&display=swap');
html, body, .stApp, [data-testid="stAppViewContainer"] { background: var(--bg) !important; color: var(--ink);
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }
.stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp div[data-testid="stMarkdownContainer"] { color: var(--ink); font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }
.stApp div[data-testid="stMarkdownContainer"] span { font-family: inherit; }
header[data-testid="stHeader"] { background: transparent !important; }
header[data-testid="stHeader"] * { color: var(--muted) !important; }
.block-container { padding-top: 1.4rem; max-width: 1320px; }
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--ink); }
hr { border-color: var(--border) !important; }
[data-baseweb="select"], [data-baseweb="select"] div { background-color: var(--surface2) !important; color: var(--ink) !important; border-color: var(--border) !important; }
[data-baseweb="select"] input { color: var(--ink) !important; }
[data-baseweb="select"] svg { fill: var(--muted) !important; }
[data-baseweb="input"], [data-baseweb="input"] input, [data-baseweb="base-input"], [data-baseweb="textarea"] textarea {
  background: var(--surface2) !important; color: var(--ink) !important; border-color: var(--border) !important; }
[data-testid="stSelectbox"] div, [data-testid="stNumberInput"] div, [data-testid="stNumberInput"] input,
[data-testid="stMultiSelect"] div, [data-testid="stTextInput"] div, [data-testid="stTextArea"] textarea {
  background-color: var(--surface2) !important; color: var(--ink) !important; border-color: var(--border) !important; }
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] * { background-color: transparent !important; }
[data-testid="stSelectbox"] > div:first-child:has([data-testid="stWidgetLabel"]),
[data-testid="stMultiSelect"] > div:first-child:has([data-testid="stWidgetLabel"]),
[data-testid="stTextInput"] > div:first-child:has([data-testid="stWidgetLabel"]),
[data-testid="stNumberInput"] > div:first-child:has([data-testid="stWidgetLabel"]) { background-color: transparent !important; }
[data-testid="stSelectbox"] svg { fill: var(--muted) !important; color: var(--muted) !important; }
[data-testid="stSelectbox"] label, [data-testid="stSelectbox"] label *, [data-testid="stSelectbox"] div:has(> label),
[data-testid="stMultiSelect"] label, [data-testid="stMultiSelect"] label *, [data-testid="stMultiSelect"] div:has(> label),
[data-testid="stTextInput"] label, [data-testid="stTextInput"] label *, [data-testid="stTextInput"] div:has(> label),
[data-testid="stNumberInput"] label, [data-testid="stNumberInput"] label *, [data-testid="stNumberInput"] div:has(> label),
[data-testid="stTextArea"] label, [data-testid="stTextArea"] label *, [data-testid="stTextArea"] div:has(> label) { background-color: transparent !important; }
[role="listbox"], div[data-baseweb="popover"] ul { background-color: var(--surface) !important; }
[role="option"], div[data-baseweb="popover"] li { background-color: var(--surface) !important; color: var(--ink) !important; }
[role="option"] * { color: var(--ink) !important; }
[role="option"]:hover, [role="option"][aria-selected="true"], div[data-baseweb="popover"] li:hover { background-color: var(--surface2) !important; }
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { color: var(--muted) !important; }
.stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button {
  background: var(--surface) !important; color: var(--ink) !important; border: 1px solid var(--border) !important;
  border-radius: 999px !important; font-weight: 500; box-shadow: none; }
.stButton button p, .stDownloadButton button p, [data-testid="stFormSubmitButton"] button p { color: inherit !important; }
.stButton button:hover, .stDownloadButton button:hover { border-color: var(--violet) !important; color: var(--violet) !important; }
.stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"],
[data-testid="stFormSubmitButton"] button[kind="primary"] { background: var(--navy1) !important; color: #fff !important; border-color: var(--navy1) !important; }
.stButton button[kind="primary"] p, [data-testid="stFormSubmitButton"] button p { color: #fff !important; }
.stButton button:disabled { opacity: .4; }
[data-testid="stExpander"] details { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 18px; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: var(--ink) !important; }
[data-testid="stForm"] { background: var(--surface); border: none !important; border-radius: 24px; padding: 1.4rem 1.6rem; box-shadow: var(--shadow); }
[data-testid="stCode"] pre, .stCode pre { background: var(--surface2) !important; }
[data-testid="stCode"] code, [data-testid="stCode"] code span { color: var(--ink) !important; background: transparent !important; }
.st-key-nav div[role="radiogroup"] { display: inline-flex; flex-wrap: wrap; gap: 2px; background: var(--surface);
  padding: 5px; border-radius: 999px; box-shadow: var(--shadow); }
.st-key-nav [data-testid="stRadioOption"], .st-key-nav div[role="radiogroup"] > label {
  background: transparent; border-radius: 999px; padding: 0.42rem 1.05rem; margin: 0 !important; cursor: pointer; }
.st-key-nav [data-testid="stRadioOption"] > div > div:not([data-testid="stMarkdownContainer"]),
.st-key-nav div[role="radiogroup"] > label > div:first-child:not([data-testid="stMarkdownContainer"]):not(:has([data-testid="stMarkdownContainer"])) { display: none !important; }
.st-key-nav [data-testid="stRadioOption"][data-selected="true"], .st-key-nav [data-testid="stRadioOption"]:has(input:checked),
.st-key-nav div[role="radiogroup"] > label:has(input:checked) { background: var(--navy1); }
.st-key-nav [data-testid="stRadioOption"] p, .st-key-nav div[role="radiogroup"] > label p { font-size: 0.9rem; font-weight: 500; color: var(--muted) !important; }
.st-key-nav [data-testid="stRadioOption"][data-selected="true"] p, .st-key-nav [data-testid="stRadioOption"]:has(input:checked) p,
.st-key-nav div[role="radiogroup"] > label:has(input:checked) p { color: #fff !important; }
[class*="st-key-card"] { background: var(--surface); border-radius: 24px; padding: 1.1rem 1.25rem 0.6rem; box-shadow: var(--shadow); }
.serif { font-family: 'Fraunces', Georgia, 'Times New Roman', serif !important; font-weight: 400; letter-spacing: -0.01em; }
.ph { display:flex; justify-content:space-between; align-items:center; gap:1rem; margin: 0.2rem 0 1.1rem; }
.ph h1 { font-family: 'Fraunces', Georgia, serif !important; font-weight: 400 !important; font-size: 2.15rem !important;
  line-height: 1.15; margin: 0 !important; padding: 0 !important; letter-spacing: -0.015em; color: var(--ink) !important; }
.ph .sub { color: var(--muted) !important; font-size: 0.98rem; margin-top: 0.3rem; }
.avatar { width: 52px; height: 52px; border-radius: 50%; display:flex; align-items:center; justify-content:center;
  font-family: 'Fraunces', Georgia, serif; font-size: 1.25rem; color: #fff; background: linear-gradient(135deg, var(--navy2), var(--violet)); flex-shrink:0; }
.lbl { font-size: 0.86rem; color: var(--muted) !important; font-weight: 500; }
.sec { font-family: 'Fraunces', Georgia, serif !important; font-size: 1.22rem; font-weight: 400; color: var(--ink) !important; margin: 0 0 0.5rem; }
.muted, .muted * { color: var(--muted) !important; }
.card { background: var(--surface); border-radius: 24px; padding: 1.15rem 1.3rem; box-shadow: var(--shadow); height: 100%; }
.chip { display:inline-flex; align-items:center; gap:.35rem; padding: .22rem .7rem; border-radius: 999px; font-size: .8rem; font-weight: 500; white-space: nowrap; }
.hero { border-radius: 28px; padding: 1.5rem 1.7rem; display:flex; gap:1.4rem; align-items:center; color:#fff;
  background: linear-gradient(135deg, var(--navy1) 0%, var(--navy2) 100%); box-shadow: var(--shadow); }
.hero * { color: #fff !important; }
.hero .k { font-size: .9rem; opacity: .78; }
.hero .v { font-family: 'Fraunces', Georgia, serif; font-size: 3.1rem; line-height: 1.05; font-weight: 400; margin: .25rem 0 .1rem; letter-spacing: -0.02em; }
.hero .s { font-size: 1.05rem; font-weight: 500; }
.hero .p { font-size: .95rem; opacity: .86; margin-top: .45rem; line-height: 1.5; }
.hero .a { font-size: .86rem; opacity: .74; margin-top: .55rem; }
.hero .chips { display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.75rem; }
.hero .chip { background: rgba(255,255,255,.12); }
.tile { border-radius: 22px; padding: 1rem 1.05rem .8rem; height: 100%; }
.tile .h { display:flex; justify-content:space-between; align-items:center; font-size:.88rem; font-weight:500; color: var(--ink) !important; }
.tile .v { font-family: 'Fraunces', Georgia, serif; font-size: 1.85rem; line-height: 1.1; margin-top: .45rem; color: var(--ink) !important; letter-spacing: -0.01em; }
.tile .u { font-size: .8rem; color: var(--muted) !important; margin-top: .1rem; }
.tile .f { font-size: .82rem; font-weight: 500; margin-top: .35rem; }
.drv { display:grid; grid-template-columns: 2.1rem 1fr 30%; gap:.7rem; align-items:center; padding:.5rem 0; border-bottom: 1px solid var(--border); }
.drv:last-child { border-bottom: none; }
.drv .ic { width: 2.1rem; height: 2.1rem; border-radius: 50%; display:flex; align-items:center; justify-content:center; }
.drv .t { font-size: .88rem; line-height: 1.4; }
.drv .track { height: 6px; background: var(--surface2); border-radius: 6px; }
.drv .bar { height: 6px; border-radius: 6px; }
.grp { font-size: .8rem; color: var(--muted) !important; font-weight: 500; margin: .55rem 0 .1rem; }
.tl .ev { display:grid; grid-template-columns: 2.1rem 1fr; gap:.7rem; padding: .45rem 0; }
.tl .ic { width: 2.1rem; height: 2.1rem; border-radius: 50%; display:flex; align-items:center; justify-content:center; }
.tl .d { font-size: .78rem; color: var(--muted) !important; }
.tl .x { font-size: .9rem; }
.callout { border-radius: 18px; padding: .85rem 1.05rem; font-size: .93rem; display:flex; gap:.7rem; align-items:flex-start; line-height:1.5; }
.kpi .v { font-family: 'Fraunces', Georgia, serif; font-size: 2.2rem; line-height: 1.1; color: var(--c, var(--ink)) !important; letter-spacing: -0.015em; }
.kpi .l { font-size: .88rem; margin-top: .2rem; }
.kpi .c { font-size: .78rem; color: var(--muted) !important; margin-top: .25rem; }
table.mt { width:100%; border-collapse:separate; border-spacing:0; font-size:.88rem; }
table.mt th { text-align:left; font-size:.78rem; font-weight:500; color: var(--muted) !important; padding:.55rem .5rem; border-bottom:1px solid var(--border); }
table.mt td { padding:.6rem .5rem; border-bottom:1px solid var(--border); color: var(--ink); vertical-align: middle; }
table.mt tr:last-child td { border-bottom: none; }
.steps { display:flex; gap:.35rem; margin: .1rem 0 1rem; }
.steps .s { flex:1; height:4px; border-radius:4px; background: var(--border); }
.steps .s.on { background: var(--navy2); }
.story { font-size: 1.04rem; line-height: 1.7; }
.foot { font-size: .8rem; color: var(--muted) !important; line-height: 1.5; }
.foot b { color: var(--ink) !important; }
.brand { display:flex; align-items:center; gap:.65rem; margin-bottom:.4rem; }
.brand .t { font-family: 'Fraunces', Georgia, serif; font-size: 1.35rem; line-height: 1.1; }
.brand .s { font-size: .78rem; color: var(--muted) !important; }
.num-row { display:flex; gap:.6rem; flex-wrap:wrap; margin:.8rem 0; }
.num-row .tile { flex:1; min-width:130px; padding:.75rem .9rem; }
</style>"""
st.markdown(CSS_VARS, unsafe_allow_html=True)
st.markdown(CSS, unsafe_allow_html=True)


def style_fig(fig, height=None, legend=False):
    fig.update_layout(
        template="plotly_dark" if MODE == "dark" else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color=P["muted"], size=12),
        margin=dict(l=6, r=6, t=8, b=6), showlegend=legend,
        hoverlabel=dict(bgcolor=P["surface"], font_color=P["ink"], bordercolor=P["border"]),
        legend=dict(font=dict(color=P["ink"])),
    )
    fig.update_xaxes(gridcolor=P["grid"], zeroline=False, linecolor=P["border"], showline=False)
    fig.update_yaxes(gridcolor=P["grid"], zeroline=False, linecolor=P["border"], showline=False)
    if height:
        fig.update_layout(height=height)
    return fig


def show_chart(fig, key=None, target=None):
    target = target or st
    kw = dict(theme=None, config={"displayModeBar": False})
    if key:
        kw["key"] = key
    try:
        target.plotly_chart(fig, width="stretch", **kw)
    except TypeError:
        target.plotly_chart(fig, use_container_width=True, **kw)


def html(s, target=None):
    (target or st).markdown(s, unsafe_allow_html=True)


def page_header(title, sub="", right="", target=None):
    html(f"<div class='ph'><div><h1>{title}</h1><div class='sub'>{sub}</div></div>{right}</div>", target)


def callout(text, kind="info", target=None):
    ic, fg, bg = {"info": ("info", P["violet"], P["lav_bg"]), "warn": ("alert", P["peach"], P["peach_bg"]),
                  "danger": ("call", P["rose"], P["rose_bg"]), "ok": ("check", P["mint"], P["mint_bg"])}[kind]
    html(f"<div class='callout' style='background:{bg}'>{icon(ic, 18, fg)}<div>{text}</div></div>", target)


def status_chip(band, extra=""):
    s = STATUS[band]
    return (f"<span class='chip' style='background:{s['bg']};color:{s['fg']}'>{icon(s['icon'], 14, s['fg'])}"
            f"{band}{extra}</span>")


# =============================================================================
# Data + helpers
# =============================================================================
@st.cache_resource(show_spinner="Loading the twins ...")
def assets():
    return load_assets()


@st.cache_resource(show_spinner=False, max_entries=60)
def _twin_cached(pid):
    return DigitalTwin(pid, assets())


def twin(pid):
    """Twin for a stored client, or for a client enrolled live in this session."""
    custom = st.session_state.get("custom_clients", {})
    if pid in custom:
        return custom[pid]["twin"]
    return _twin_cached(pid)


A = assets()
META = A["meta"]
THR = META["threshold"]
SHOWCASE = json.loads((C.DATA_DIR / "showcase.json").read_text())["patients"]
SHOW_IDS = [s["patient_id"] for s in SHOWCASE]
TEST_IDS = META["test_patients"]
PAT = A["patients"].set_index("patient_id")

# Display names make the synthetic people relatable. They are invented, like the data.
M_NAMES = ["Arjun", "Rohan", "Aditya", "Kabir", "Vihaan", "Ishaan", "Rahul", "Siddharth", "Nikhil", "Karthik",
           "Aman", "Varun", "Pranav", "Harsh", "Yash", "Dev", "Farhan", "Aakash", "Manav", "Joel"]
F_NAMES = ["Ananya", "Priya", "Diya", "Meera", "Isha", "Kavya", "Sneha", "Riya", "Aisha", "Pooja",
           "Neha", "Tanvi", "Shreya", "Nandini", "Sara", "Aditi", "Fatima", "Lakshmi", "Mitali", "Anjali"]
FIXED = {"MT0441": "Arjun", "MT0156": "Rohan"}


def name_of(pid):
    custom = st.session_state.get("custom_clients", {})
    if pid in custom:
        return custom[pid]["name"]
    if pid in FIXED:
        return FIXED[pid]
    h = int(hashlib.md5(pid.encode()).hexdigest(), 16)
    pool = F_NAMES if PAT.loc[pid, "sex"] == "F" else M_NAMES
    return pool[h % len(pool)]


def patient_row(pid):
    custom = st.session_state.get("custom_clients", {})
    return custom[pid]["twin"].patient if pid in custom else PAT.loc[pid]


def avatar(pid):
    return f"<div class='avatar'>{name_of(pid)[:1]}</div>"


SIG_ICON = {"sleep_hours": "moon", "sleep_onset": "bed", "sleep_efficiency": "bed", "resting_hr": "heart",
            "hrv_rmssd": "activity", "steps": "footprints", "night_screen_min": "phone", "social_contacts": "message"}
SIG_TONE = {"sleep_hours": "lav", "sleep_onset": "lav", "sleep_efficiency": "sky", "resting_hr": "rose",
            "hrv_rmssd": "rose", "steps": "mint", "night_screen_min": "peach", "social_contacts": "sky"}
SIG_SHORT = {"sleep_hours": "Sleep", "sleep_onset": "Bedtime", "sleep_efficiency": "Sleep quality",
             "resting_hr": "Resting heart rate", "hrv_rmssd": "Heart-rate variability", "steps": "Steps",
             "night_screen_min": "Late-night phone", "social_contacts": "Social contact"}
SIG_HELP = {"hrv_rmssd": "Variation between heartbeats; it tends to fall under stress.",
            "sleep_efficiency": "Share of time in bed actually asleep."}


def fmt_val(sig, v):
    if v is None or pd.isna(v):
        return "–"
    if sig == "sleep_hours":
        return f"{int(v)}h {int(round((v % 1) * 60)) % 60:02d}m"
    if sig == "sleep_onset":
        return fmt_clock(v)
    if sig == "sleep_efficiency":
        return f"{v:.0f}%"
    if sig == "resting_hr":
        return f"{v:.0f} bpm"
    if sig == "hrv_rmssd":
        return f"{v:.0f} ms"
    if sig == "steps":
        return f"{v:,.0f}"
    if sig == "night_screen_min":
        return f"{v:.0f} min"
    return f"{v:.0f}/day"


def risk_score(sig, z):
    """Deviation in the 'toward risk' direction."""
    if pd.isna(z):
        return 0.0
    d = C.RISK_DIRECTION[sig]
    return abs(z) if d == 0 else d * z


def phrase(sig, now, base):
    """Everyday-language description of a change."""
    if pd.isna(now) or pd.isna(base) or not base:
        return None
    pct = 100 * (now - base) / base
    mins = (now - base) * 60
    return {
        "sleep_hours": f"sleeping {abs(mins):.0f} min {'less' if mins < 0 else 'more'} than usual",
        "sleep_onset": f"going to bed {abs(mins):.0f} min {'later' if mins > 0 else 'earlier'} than usual",
        "sleep_efficiency": "sleeping less soundly" if pct < 0 else "sleeping more soundly",
        "resting_hr": f"running a resting heart rate {abs(pct):.0f}% {'higher' if pct > 0 else 'lower'} than usual",
        "hrv_rmssd": f"showing heart-rate variability {abs(pct):.0f}% {'lower' if pct < 0 else 'higher'} than usual",
        "steps": f"walking {abs(pct):.0f}% {'less' if pct < 0 else 'more'} than usual",
        "night_screen_min": f"spending {abs(pct):.0f}% {'more' if pct > 0 else 'less'} time on the phone late at night",
        "social_contacts": f"having {abs(pct):.0f}% {'fewer' if pct < 0 else 'more'} social contacts",
    }[sig]


def verdict(score):
    if score >= 1.5:
        return "Big change", P["rose"]
    if score >= 0.75:
        return "Drifting", P["peach"]
    return "Usual", P["muted"]


RHYTHM_PREFIXES = ("z7_", "z3_", "dz7_", "raw7_")
RHYTHM_OTHER = {"twin_deviation_index", "sleep_onset_irregularity7", "missing_days7"}


def is_rhythm_feature(f):
    return f.startswith(RHYTHM_PREFIXES) or f in RHYTHM_OTHER


BACKGROUND_TEXT = {"phq9_last": "a high recent PHQ-9", "phq9_change": "a recent PHQ-9 swing", "days_since_phq9": "time since the last PHQ-9",
                   "prior_episodes": "previous episodes", "diagnosed_episodes_so_far": "a recent diagnosed episode",
                   "days_since_last_diagnosis": "a recent diagnosed episode", "family_history_depression": "family history",
                   "on_antidepressant": "current medication", "known_stressor_period": "exam / work-crunch season",
                   "age": "age", "is_student": "being a student", "is_female": "sex", "hypothyroidism": "hypothyroidism",
                   "anaemia": "anaemia", "vitamin_d_deficiency": "vitamin D deficiency"}


def background_share(tw, day, feats=None):
    """Share of the positive SHAP push that comes from history/context rather than this week's rhythm."""
    ex = tw.explain(day, k=60, feats=feats)
    pos = ex[ex.shap > 0]
    if not len(pos):
        return 0.0, []
    bg = pos[~pos.feature.map(is_rhythm_feature)]
    names = [BACKGROUND_TEXT.get(f, readable(f).lower()) for f in bg.sort_values("shap", ascending=False).feature.head(3)]
    return float(bg.shap.sum() / pos.shap.sum()), names


def summary_sentence(tw, day):
    s = tw.state(day)
    nm = name_of(tw.pid)
    sg = s["signals"].copy()
    sg["score"] = [risk_score(a, b) for a, b in zip(sg.signal, sg.z)]
    top = sg[sg.score >= 0.6].sort_values("score", ascending=False).head(3)
    parts = [p for p in (phrase(r.signal, r.current_7d, r.baseline) for r in top.itertuples()) if p]
    pron = "her" if patient_row(tw.pid)["sex"] == "F" else "his"
    share, names = background_share(tw, day) if s["band"] in ("Watch", "Early warning") else (0.0, [])
    bg_txt = ", ".join(names[:-1]) + (" and " if len(names) > 1 else "") + names[-1] if names else ""
    if not parts:
        base = f"Over the last week, {nm}'s rhythm has looked like {pron} usual self."
        if share >= 0.5 and names:
            base += f" The estimate is driven mainly by background risk ({bg_txt}), not by this week's data."
        return base
    sent = (f"Over the last week, {nm} has been {parts[0]}." if len(parts) == 1
            else f"Over the last week, {nm} has been {', '.join(parts[:-1])} and {parts[-1]}.")
    if share >= 0.6 and names:
        sent += f" Background risk ({bg_txt}) adds to this."
    return sent


def in_care(tw, day):
    e = tw.ehr_until(day)
    dx = e[e["type"] == "diagnosis"]
    if len(dx):
        return (pd.Timestamp(tw.daily.loc[day, "date"]) - pd.Timestamp(dx["date"].iloc[-1])).days <= 60
    return False


def sparkline(values, base, sd, color, w=220, h=40):
    v = np.array(values, dtype=float)
    ok = ~np.isnan(v)
    if ok.sum() < 3:
        return ""
    has_band = not (np.isnan(base) or np.isnan(sd))
    lo_b, hi_b = (base - sd, base + sd) if has_band else (np.nan, np.nan)
    lo = np.nanmin(np.r_[v[ok], lo_b]) if has_band else np.nanmin(v)
    hi = np.nanmax(np.r_[v[ok], hi_b]) if has_band else np.nanmax(v)
    rng = (hi - lo) or 1

    def y(q):
        return h - 3 - (q - lo) / rng * (h - 6)
    xs = np.linspace(2, w - 2, len(v))
    pts = " ".join(f"{x:.1f},{y(q):.1f}" for x, q in zip(xs[ok], v[ok]))
    band = (f"<rect x='0' y='{y(hi_b):.1f}' width='{w}' height='{max(1, y(lo_b) - y(hi_b)):.1f}' rx='6' "
            f"fill='{color}' opacity='0.12'/>") if has_band else ""
    last = f"<circle cx='{xs[ok][-1]:.1f}' cy='{y(v[ok][-1]):.1f}' r='3' fill='{color}'/>"
    return (f"<svg viewBox='0 0 {w} {h}' width='100%' height='{h}' preserveAspectRatio='none' style='margin-top:.35rem'>{band}"
            f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='1.8' stroke-linejoin='round' "
            f"stroke-linecap='round' vector-effect='non-scaling-stroke'/>{last}</svg>")


def ring(frac, color, icon_name, size=150):
    """Progress ring with a tick at the halfway mark (the alert line)."""
    r, sw = size * 0.39, size * 0.07
    c = 2 * np.pi * r
    f = float(np.clip(frac, 0, 1))
    cx = size / 2
    return (f"<svg width='{size}' height='{size}' viewBox='0 0 {size} {size}' style='flex-shrink:0'>"
            f"<circle cx='{cx}' cy='{cx}' r='{r}' fill='none' stroke='rgba(255,255,255,.14)' stroke-width='{sw}'/>"
            f"<circle cx='{cx}' cy='{cx}' r='{r}' fill='none' stroke='{color}' stroke-width='{sw}' stroke-linecap='round' "
            f"stroke-dasharray='{c * f:.1f} {c:.1f}' transform='rotate(-90 {cx} {cx})'/>"
            f"<line x1='{cx}' y1='{cx + r - sw}' x2='{cx}' y2='{cx + r + sw}' stroke='rgba(255,255,255,.75)' stroke-width='2'/>"
            f"<g transform='translate({cx - size * 0.16:.1f},{cx - size * 0.16:.1f})'>"
            f"<svg width='{size * 0.32:.0f}' height='{size * 0.32:.0f}' viewBox='0 0 24 24' fill='none' stroke='#fff' "
            f"stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'>{ICONS[icon_name]}</svg></g></svg>")


# =============================================================================
# Components
# =============================================================================
def hero(tw, day, target=None, compact=False):
    s = tw.state(day)
    band = s["band"]
    st_ = STATUS[band]
    risk = s["risk"]
    risk_txt = "–" if pd.isna(risk) else f"{100 * risk:.1f}%"
    chips = []
    if pd.notna(s["relative_risk"]):
        chips.append(f"{s['relative_risk']:.1f}× a typical person")
    if pd.notna(risk):
        line = 100 * tw.threshold
        if s.get("rising"):
            chips.append(f"Crossed the {line:.1f}% alert line today · shown as Watch until it holds")
        elif s.get("borderline"):
            chips.append(f"{'Just over' if risk >= tw.threshold else 'Just under'} the {line:.1f}% alert line")
        else:
            chips.append(f"Alert line {line:.1f}%")
    if in_care(tw, day):
        chips.append("Already in care · watching for relapse")
    sentence = summary_sentence(tw, day) if band != "Learning baseline" else st_["plain"]
    frac = 0 if pd.isna(risk) else risk / (2 * tw.threshold)
    date_txt = pd.Timestamp(s["date"]).strftime("%a %d %b %Y")
    rising = " · rising" if s.get("rising") else ""
    html(f"<div class='hero' title='The ring fills as the estimate approaches twice the alert line; the mark at the "
         f"bottom is the alert line.'><div style='flex:1;min-width:0'>"
         f"<div class='k'>Chance an episode begins in the next 7 days · {date_txt}</div>"
         f"<div class='v'>{risk_txt}</div>"
         f"<div class='s'>{st_['sky']} · {band}{rising}</div>"
         f"<div class='p'>{sentence}</div>"
         f"<div class='a'>{ACTIONS[band]}</div>"
         f"<div class='chips'>{''.join(f'<span class=chip>{c}</span>' for c in chips)}</div>"
         f"</div>{ring(frac, st_['ring'], st_['icon'], 118 if compact else 150)}</div>", target)


def signal_cards(tw, day, cols=4):
    s = tw.state(day)
    full = tw.daily.set_index("day")
    win = range(max(0, day - 29), day + 1)
    sg = s["signals"]
    for i0 in range(0, len(sg), cols):
        cs = st.columns(cols)
        for c, r in zip(cs, sg.iloc[i0:i0 + cols].itertuples()):
            tone = SIG_TONE[r.signal]
            fg, bg = P[tone], P[f"{tone}_bg"]
            vtxt, vcol = verdict(risk_score(r.signal, r.z))
            sm = full[r.signal].rolling(3, min_periods=1).mean().reindex(win).values
            sd = full[r.signal].shift(C.BASELINE_GAP).rolling(C.BASELINE_WINDOW, min_periods=20).std().get(day, np.nan)
            ph = phrase(r.signal, r.current_7d, r.baseline)
            detail = "" if (vtxt == "Usual" or not ph) else f"<span class='muted'> · {ph}</span>"
            tip = f" title='{SIG_HELP[r.signal]}'" if r.signal in SIG_HELP else ""
            html(f"<div class='tile' style='background:{bg}'{tip}>"
                 f"<div class='h'><span>{SIG_SHORT[r.signal]}</span>{icon(SIG_ICON[r.signal], 18, fg)}</div>"
                 f"<div class='v'>{fmt_val(r.signal, r.current_7d)}</div>"
                 f"<div class='u'>last 7 days · usual {fmt_val(r.signal, r.baseline)}</div>"
                 f"{sparkline(sm, r.baseline, sd, fg)}"
                 f"<div class='f' style='color:{vcol}'>{vtxt}{detail}</div></div>", c)
        st.write("")


def risk_chart(tw, upto, start=C.MIN_HISTORY_DAYS, end=None, height=310, gt=True, mark_day=True):
    rs = tw.risk_series()
    end = end if end is not None else upto
    rs = rs[(rs.day >= start - 2) & (rs.day <= upto)]
    dates = pd.to_datetime(rs.date)
    r = 100 * rs.risk
    t = 100 * THR
    ymax = float(min(max(1.7 * t, (r.max() if r.notna().any() else 0) * 1.15), 22))
    x0 = pd.Timestamp(tw.daily.loc[max(start - 2, 0), "date"])
    x1 = pd.Timestamp(tw.daily.loc[min(max(end, upto) + 2, tw.n_days - 1), "date"])
    fig = go.Figure()
    if gt:
        flags = rs.in_episode.values
        a = None
        for i, f in enumerate(flags):
            if f and a is None:
                a = i
            if a is not None and (not f or i == len(flags) - 1):
                b = i if f else i - 1
                fig.add_vrect(x0=dates.iloc[a], x1=dates.iloc[b], fillcolor=P["lav"], opacity=0.10, line_width=0,
                              annotation_text="actual episode", annotation_position="top left",
                              annotation_font=dict(size=10, color=P["muted"]))
                a = None
    fig.add_hline(y=t, line_dash="dash", line_color=P["rose"], line_width=1.2,
                  annotation_text=f"alert line {t:.1f}%", annotation_position="top right",
                  annotation_font=dict(size=10, color=P["rose"]))
    fig.add_hline(y=t / 2, line_dash="dot", line_color=P["peach"], line_width=1.2,
                  annotation_text="watch", annotation_position="top right", annotation_font=dict(size=10, color=P["peach"]))
    e = tw.ehr_until(upto)
    for _, ev in e[e["type"] == "diagnosis"].iterrows():
        if pd.Timestamp(ev["date"]) >= x0:
            fig.add_vline(x=pd.Timestamp(ev["date"]), line_dash="dot", line_color=P["muted"], line_width=1)
            fig.add_annotation(x=pd.Timestamp(ev["date"]), y=ymax * 0.92, text=" diagnosed", showarrow=False,
                               xanchor="left", font=dict(size=10, color=P["muted"]))
    line_col = P["violet"] if MODE == "dark" else P["navy1"]
    fill_col = "rgba(143,138,224,.10)" if MODE == "dark" else "rgba(45,44,90,.06)"
    fig.add_trace(go.Scatter(x=dates, y=r.clip(upper=ymax * 0.98), mode="lines", customdata=r, fill="tozeroy",
                             fillcolor=fill_col, line=dict(color=line_col, width=2, shape="spline", smoothing=0.6),
                             hovertemplate="%{x|%d %b}<br>%{customdata:.1f}% chance<extra></extra>"))
    if mark_day and len(rs) and pd.notna(rs.iloc[-1].risk):
        last = rs.iloc[-1]
        b = tw.band(last.risk, tw.prev_risk(int(last.day)))
        fig.add_trace(go.Scatter(x=[pd.Timestamp(last.date)], y=[min(100 * last.risk, ymax * 0.98)], mode="markers",
                                 marker=dict(size=12, color=STATUS[b]["fg"], line=dict(color=P["surface"], width=3)),
                                 hoverinfo="skip"))
    fig.update_yaxes(range=[0, ymax], ticksuffix="%", nticks=5)
    fig.update_xaxes(range=[x0, x1], tickformat="%d %b", showgrid=False)
    return style_fig(fig, height)


def chart_card(title, fig, key, sub="", target=None):
    target = target or st
    with target.container(key=f"card_{key}"):
        html(f"<div class='sec' style='margin-bottom:0'>{title}</div>"
             + (f"<div class='lbl'>{sub}</div>" if sub else ""))
        show_chart(fig, key=key)


DRV_ICON = {f"{p}{s}": SIG_ICON[s] for s in C.SIGNALS for p in ("z7_", "z3_", "dz7_", "raw7_")}
DRV_ICON.update({"phq9_last": "clipboard", "phq9_change": "clipboard", "days_since_phq9": "clipboard",
                 "known_stressor_period": "book", "prior_episodes": "history", "diagnosed_episodes_so_far": "history",
                 "days_since_last_diagnosis": "history", "twin_deviation_index": "compass",
                 "sleep_onset_irregularity7": "bed", "missing_days7": "watch", "family_history_depression": "users",
                 "on_antidepressant": "pill", "age": "calendar"})


def driver_text(feature, value, row_now, row_base):
    """Turn a model feature into an everyday sentence."""
    for pre in ("z7_", "z3_", "dz7_", "raw7_"):
        if feature.startswith(pre):
            sig = feature[len(pre):]
            if pre == "raw7_":
                return f"{SIG_SHORT[sig]} this week: {fmt_val(sig, row_now.get(sig))}"
            ph = phrase(sig, row_now.get(sig), row_base.get(sig))
            when = "in the last 3 days" if pre == "z3_" else "this week" if pre == "z7_" else "vs. last week"
            if not ph:
                return f"{SIG_SHORT[sig]} changing ({when})"
            return f"{ph[0].upper() + ph[1:]} ({when})"
    v = value
    if feature == "phq9_last":
        return f"Latest mood check-in (PHQ-9) score: {v:.0f}"
    if feature == "phq9_change":
        return (f"Mood check-in score {'dropped' if v < 0 else 'rose'} {abs(v):.0f} points at the last check-in "
                "(big swings often follow a recent episode)")
    if feature == "days_since_phq9":
        return f"{v:.0f} days since the last mood check-in"
    if feature == "known_stressor_period":
        return "Exam or work-crunch season" if v else "Not in an exam / crunch period"
    if feature == "prior_episodes":
        return f"Has had {v:.0f} depressive episode{'s' if v != 1 else ''} before"
    if feature == "twin_deviation_index":
        return "Overall drift away from their own normal"
    if feature == "sleep_onset_irregularity7":
        return "Irregular bedtimes this week"
    if feature == "missing_days7":
        return f"Watch not worn on {v:.0f} of the last 7 days"
    if feature == "family_history_depression":
        return "Family history of depression"
    if feature == "on_antidepressant":
        return "Currently on an antidepressant"
    return readable(feature)


def drivers_html(ex, max_items=6, tw=None, day=None, feats=None):
    ex = ex[ex.shap > 0].head(max_items)
    if not len(ex):
        return "<div class='muted' style='padding:.5rem 0'>Nothing is pushing the risk up right now.</div>"
    mx = ex.shap.max()
    now, base = {}, {}
    if tw is not None:
        fx = feats if feats is not None else tw.feats
        row_ = fx.loc[fx.day == day].iloc[0]
        now = {s_: row_[f"raw7_{s_}"] for s_ in C.SIGNALS}
        base = {s_: row_[f"base_{s_}"] for s_ in C.SIGNALS}

    def row(r, fg, bg):
        txt = driver_text(r.feature, r.value, now, base) if tw is not None else r.driver
        return (f"<div class='drv'><div class='ic' style='background:{bg}'>{icon(DRV_ICON.get(r.feature, 'compass'), 16, fg)}</div>"
                f"<div class='t'>{txt}</div><div class='track'><div class='bar' style='width:{max(6, 100 * r.shap / mx):.0f}%;"
                f"background:{fg}'></div></div></div>")
    week = [r for r in ex.itertuples() if is_rhythm_feature(r.feature)]
    bg = [r for r in ex.itertuples() if not is_rhythm_feature(r.feature)]
    out = ""
    if week:
        out += "<div class='grp'>This week's rhythm</div>" + "".join(row(r, P["rose"], P["rose_bg"]) for r in week)
    if bg:
        out += "<div class='grp'>Background risk · history and context</div>" + "".join(row(r, P["peach"], P["peach_bg"]) for r in bg)
    return out


def drivers_card(title, inner, foot="", target=None):
    html(f"<div class='card'><div class='sec'>{title}</div>{inner}"
         + (f"<div class='foot' style='margin-top:.6rem'>{foot}</div>" if foot else "") + "</div>", target)


def ehr_timeline(tw, day):
    e = tw.ehr_until(day)
    if not len(e):
        return "<div class='muted'>No health-record events yet.</div>"
    look = {"phq9_screening": ("clipboard", "lav"), "diagnosis": ("stethoscope", "rose"),
            "medication_start": ("pill", "peach"), "therapy_referral": ("chat", "mint")}
    out = "<div class='tl'>"
    for _, r in e.iloc[::-1].head(8).iterrows():
        if r["type"] == "phq9_screening":
            sc = int(r["phq9"])
            sev = ("minimal" if sc < 5 else "mild" if sc < 10 else "moderate" if sc < 15
                   else "moderately severe" if sc < 20 else "severe")
            label = f"PHQ-9 check-in · <b>{sc}</b> <span class='muted'>({sev})</span>"
        elif r["type"] == "diagnosis":
            label = f"<b>Diagnosed</b> · depressive episode (ICD-10 F32), PHQ-9 {int(r['phq9'])}"
        elif r["type"] == "medication_start":
            label = f"Started <b>{r['detail']}</b>"
        else:
            label = "Referred for <b>talk therapy (CBT)</b>"
        ic, tone = look.get(r["type"], ("info", "lav"))
        out += (f"<div class='ev'><div class='ic' style='background:{P[tone + '_bg']}'>{icon(ic, 15, P[tone])}</div>"
                f"<div><div class='d'>{pd.Timestamp(r['date']).strftime('%d %b %Y')}</div><div class='x'>{label}</div></div></div>")
    return out + "</div>"


def kpi(col, value, label, caption="", color=None):
    style = f" style='--c:{color}'" if color else ""
    html(f"<div class='card kpi'><div class='v'{style}>{value}</div><div class='l'>{label}</div>"
         + (f"<div class='c'>{caption}</div>" if caption else "") + "</div>", col)


# =============================================================================
# Sidebar
# =============================================================================
# Apply navigation requests made at the end of the previous run (widgets can't be
# changed once they are drawn, so we set their state here, before drawing).
if "pending_client" in st.session_state:
    st.session_state["client_select"] = st.session_state.pop("pending_client")
if "pending_page" in st.session_state:
    st.session_state["nav_page"] = st.session_state.pop("pending_page")

BRAND_MARK = ("<svg width='38' height='38' viewBox='0 0 38 38'><circle cx='15' cy='19' r='10' fill='none' "
              f"stroke='{P['violet']}' stroke-width='2'/><circle cx='23' cy='19' r='10' fill='{P['lav_bg']}' "
              f"stroke='{P['navy2'] if MODE == 'light' else P['lav']}' stroke-width='2'/></svg>")

with st.sidebar:
    html(f"<div class='brand'>{BRAND_MARK}<div><div class='t'>Manas Twin</div>"
         "<div class='s'>Early warning for depressive episodes</div></div></div>")
    st.toggle("Dark mode", key="dark")
    st.divider()
    only_test = st.checkbox("Only clients unseen during training", value=True,
                            help="Keeps the demo honest: these synthetic clients were never used to train the model.")
    pool = TEST_IDS if only_test else A["patients"]["patient_id"].tolist()
    custom_ids = list(st.session_state.get("custom_clients", {}).keys())
    options = custom_ids + SHOW_IDS + [p for p in pool if p not in SHOW_IDS]

    def _label(p):
        if p in custom_ids:
            return f"{name_of(p)} · {p} · new"
        return f"{name_of(p)} · {p} · demo" if p in SHOW_IDS else f"{name_of(p)} · {p}"
    pid = st.selectbox("Client", options, format_func=_label, key="client_select")
    tw = twin(pid)
    default_day = next((s_["suggested_day"] for s_ in SHOWCASE if s_["patient_id"] == pid), 150)
    key = f"day_{pid}"
    if key not in st.session_state:
        st.session_state[key] = default_day
    day = st.slider("Date", C.MIN_HISTORY_DAYS, tw.n_days - 1, key=key, format="day %d",
                    help="Move through time to watch the twin update.")
    html(f"<div class='lbl' style='margin-top:-0.5rem'>{icon('calendar', 14, P['muted'])} "
         f"{pd.Timestamp(tw.daily.loc[day, 'date']).strftime('%A, %d %B %Y')}</div>")
    show_gt = st.checkbox("Show actual episodes", value=True,
                          help="Ground truth from the synthetic simulator, shown for evaluation. The twin never sees it.")
    st.divider()
    if st.button("New tester · reset the app", width="stretch",
                 help="Clears all selections so the next person starts fresh."):
        keep = st.session_state.get("dark", False)
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.session_state.dark = keep
        st.rerun()
    html(f"<div class='foot' style='margin-top:.7rem'>Research prototype. Clients are <b>synthetic</b>. "
         f"Not a diagnostic tool.<br><br>If you or someone you know is struggling: <b>{C.HELPLINE}</b></div>")

# =============================================================================
# Navigation
# =============================================================================
P_GUIDE, P_STORY, P_TWIN, P_WHATIF, P_CLIN, P_LIVE, P_PERF, P_ABOUT = PAGES = [
    "Guide", "Story", "Twin", "What-if", "Clinician", "Live", "Evidence", "About"]
with st.container(key="nav"):
    page = st.radio("Navigate", PAGES, horizontal=True, label_visibility="collapsed", key="nav_page")
st.write("")

# =============================================================================
# Story content
# =============================================================================
STORY_PID = "MT0441"
STORY = [
    dict(day=95, title="Meet Arjun",
         text="Arjun is 18, an engineering student in Kochi. His smartwatch and phone have been quietly feeding his "
              "<b>digital twin</b>, a living virtual copy of his daily rhythm, for three months. By now the twin has "
              "learned what <b>normal looks like for him</b>: how long he sleeps, when he goes to bed, how much he "
              "moves and how his heart behaves.", cards=True),
    dict(day=108, title="Exams begin",
         text="End-semester exams start. Arjun sleeps a bit less, stays up later and spends more time on his phone at "
              "night. On their own these changes look small, and a one-size-fits-all rule would ignore them. But the "
              "twin compares him with <b>his own normal</b>, and the clouds start gathering.", cards=True),
    dict(day=110, title="An early warning, seven days ahead",
         text="The twin's estimate crosses the alert line. Nothing dramatic has happened and nobody around Arjun has "
              "noticed anything. With his consent, his campus counsellor gets a gentle alert to check in and repeat a "
              "short PHQ-9 questionnaire. The panel below shows <b>why</b> the twin is concerned.", drivers=True),
    dict(day=117, title="The episode begins",
         text="Seven days later, a depressive episode begins (the shaded band, which comes from the simulator's hidden "
              "ground truth). This is exactly the window the twin was warning about."),
    dict(day=136, title="Usual care catches up, 19 days late",
         text="Without the twin, Arjun is only diagnosed on 17 May, 19 days into the episode, when he scores 16 on the "
              "PHQ-9. He starts medication and therapy. With the twin, support could have started on 21 April: "
              "<b>26 days earlier</b>."),
    dict(day=112, title="What if he had a better week?",
         text="The twin can also run experiments safely on the virtual Arjun. Suppose his last week had included an "
              "hour more sleep, a 45-minute earlier bedtime, 3,000 more steps a day, less late-night phone time and a "
              "few more conversations with friends. Here is how his outlook changes.", whatif=True),
]


def whatif_card(res, labels=("Actual week", "Your scenario")):
    sb, sa = STATUS[res["band_before"]], STATUS[res["band_after"]]
    change = (res["after"] / res["before"] - 1) * 100 if res["before"] else 0

    def side(lbl, val, s):
        return (f"<div style='flex:1;text-align:center'><div class='lbl'>{lbl}</div>"
                f"<div style='width:64px;height:64px;border-radius:50%;background:{s['bg']};display:flex;align-items:center;"
                f"justify-content:center;margin:.6rem auto .4rem'>{icon(s['icon'], 30, s['fg'])}</div>"
                f"<div class='serif' style='font-size:2.3rem;color:{s['fg']}'>{100 * val:.1f}%</div>"
                f"<div class='lbl'>{s['sky']}</div></div>")
    verb_col = P["mint"] if change <= 0 else P["rose"]
    return (f"<div class='card'><div style='display:flex;align-items:center;gap:.5rem'>"
            f"{side(labels[0], res['before'], sb)}<div class='muted'>{icon('trend', 26, P['muted'])}</div>"
            f"{side(labels[1], res['after'], sa)}</div>"
            f"<div style='text-align:center;margin-top:.9rem;font-size:.98rem'>Chance of an episode in the next 7 days "
            f"<b style='color:{verb_col}'>{'falls' if change <= 0 else 'rises'} by {abs(change):.0f}%</b></div></div>")


# =============================================================================
# Guide
# =============================================================================
if page == P_GUIDE:
    page_header("A guide for psychologists and counsellors",
                "About 13 minutes: three to read, ten to explore. Thank you for taking a look.")
    c1, c2 = st.columns([1.12, 1])
    with c1:
        html("<div class='card story'><div class='sec'>What you are looking at</div>"
             "<b>Manas Twin</b> is a prototype <b>digital twin</b> for depression: a continuously updated model of one "
             "client's daily rhythm (sleep, bedtime, heart-rate variability, activity, late-night phone use, social "
             "contact) joined with their record (PHQ-9 screenings, diagnoses, medication). It learns what <b>normal "
             "looks like for that client</b> and estimates the chance that a depressive episode <b>starts within the "
             "next 7 days</b>.<div class='sec' style='margin-top:1rem'>What it is not</div>"
             "It does <b>not</b> diagnose, does <b>not</b> assess suicide or self-harm risk, and does <b>not</b> "
             "replace your judgement. It is designed as a <b>prompt for a human check-in</b>. All clients in this "
             "prototype are <b>synthetic</b> (computer-generated); no real person's data is shown.</div>")
        st.write("")
        tasks = [("Story", "Click Next through the six steps (3 min).", "Does the sequence make clinical sense?"),
                 ("Twin", "Client Arjun. In the sidebar, drag the date from day 100 to 125.",
                  "Note the day you would first have been concerned, and when the twin flagged."),
                 ("Clinician", "Open two records from the caseload. Try Enrol a demo client.",
                  "Do the explanations and the PHQ-9 timeline read sensibly?"),
                 ("What-if", "Press Full healthy week, then move one slider yourself.",
                  "Are these realistic levers for a client? Is the caveat clear enough?"),
                 ("Evidence", "Skim the numbers, including the real-data check.", "Is the performance framed honestly?")]
        rows = "".join(
            f"<div class='ev'><div class='ic' style='background:{P['lav_bg']};color:{P['lav']};font-family:Fraunces,serif'>"
            f"{i + 1}</div><div class='x'><b>{t}</b> · {d}<div class='muted' style='font-size:.85rem'>{q}</div></div></div>"
            for i, (t, d, q) in enumerate(tasks))
        html(f"<div class='card'><div class='sec'>Please try these five tasks</div><div class='tl'>{rows}</div>"
             "</div>")
    with c2:
        def srow(band, txt):
            return f"<tr><td>{status_chip(band)}</td><td>{txt}</td></tr>"
        html(f"<div class='card'><div class='sec'>How to read the screen</div><table class='mt'>"
             f"<tr><th>Element</th><th>Meaning</th></tr>"
             + srow("Stable", f"Clear skies. 7-day episode risk below {100 * THR / 2:.1f}%. Close to the client's own normal.")
             + srow("Watch", f"Clouds gathering. Risk {100 * THR / 2:.1f}–{100 * THR:.1f}%, or a single-day jump over the "
                             "alert line. Light-touch check-in.")
             + srow("Early warning", f"Storm likely. Risk ≥ {100 * THR:.1f}% (about 3× the cohort base rate) and holding. "
                                     "Contact within 48 h and a PHQ-9.")
             + f"<tr><td>The ring</td><td>Fills as the estimate approaches twice the alert line. The mark at the bottom is "
               f"the alert line.</td></tr>"
             f"<tr><td>× a typical person</td><td>Relative to the cohort base rate ({100 * tw.base_rate:.1f}% per week).</td></tr>"
             f"<tr><td>Usual · Drifting · Big change</td><td>How far the last 7 days sit from the client's own 45-day "
             f"baseline (about 0.75 and 1.5 SD).</td></tr>"
             f"<tr><td>HRV</td><td>Heart-rate variability (RMSSD). Lower under sustained stress.</td></tr>"
             f"<tr><td>PHQ-9</td><td>0–4 minimal · 5–9 mild · 10–14 moderate · 15–19 moderately severe · 20–27 severe.</td></tr>"
             f"<tr><td>Shaded band</td><td>Actual episode from the simulator, shown only to evaluate. The twin never "
             f"sees it.</td></tr></table></div>")
        st.write("")
        faq = [("Why no diagnosis?", "Onset prediction is a screening signal; diagnosis needs a clinician and a conversation."),
               ("How many false alarms?", "About one false-alarm day a month per client at the current threshold. See Evidence."),
               ("Who sees the data?", "The client, and with their revocable consent, their counsellor. Institutions see aggregates only."),
               ("No smartwatch?", "A phone-only mode (sleep, screen time, social contact) is on the roadmap."),
               ("What about crisis?", "Out of scope by design. The app signposts Tele-MANAS 14416 and relies on the human in the loop."),
               ("Is this real data?", "The clients are synthetic. The rhythm features were checked on real wrist-actigraphy "
                                      "from 55 people (see Evidence).")]
        html("<div class='card'><div class='sec'>Questions clinicians usually ask</div>"
             + "".join(f"<div style='padding:.4rem 0;border-bottom:1px solid var(--border);font-size:.9rem'><b>{q}</b> "
                       f"<span class='muted'>{a}</span></div>" for q, a in faq) + "</div>")
    st.write("")
    callout("Use <b>New tester · reset the app</b> in the sidebar to start a fresh walkthrough.", "info")

# =============================================================================
# Story
# =============================================================================
elif page == P_STORY:
    st.session_state.setdefault("story_step", 0)
    i = st.session_state.story_step
    stp = STORY[i]
    stw = twin(STORY_PID)
    html("<div class='steps'>" + "".join(f"<div class='s {'on' if k <= i else ''}'></div>" for k in range(len(STORY))) + "</div>")
    page_header(stp["title"], f"Step {i + 1} of {len(STORY)} · "
                f"{pd.Timestamp(stw.daily.loc[stp['day'], 'date']).strftime('%d %B %Y')}", avatar(STORY_PID))
    left, right = st.columns([1, 1.1])
    with left:
        html(f"<div class='card story'>{stp['text']}</div>")
        st.write("")
        b1, b2, _ = st.columns([1, 1, 1.4])
        if b1.button("Back", disabled=i == 0, width="stretch"):
            st.session_state.story_step -= 1
            st.rerun()
        if i < len(STORY) - 1:
            if b2.button("Next", type="primary", width="stretch"):
                st.session_state.story_step += 1
                st.rerun()
        elif b2.button("Restart", type="primary", width="stretch"):
            st.session_state.story_step = 0
            st.rerun()
    with right:
        if stp.get("whatif"):
            res = stw.what_if(stp["day"], sleep_hours=1, sleep_onset_shift_h=-0.75, steps=3000,
                              night_screen_min=-45, social_contacts=5)
            html(whatif_card(res, ("His actual week", "With a better week")))
            st.write("")
            callout("A model-based experiment: it shows how the twin's estimate responds, not a guaranteed effect. "
                    "Try your own changes in the What-if tab.", "info")
        else:
            hero(stw, stp["day"], compact=True)
    st.write("")
    if stp.get("drivers"):
        c1, c2 = st.columns([1.4, 1])
        chart_card("Outlook over time", risk_chart(stw, stp["day"], 60, 140, 290), f"story_chart_{i}", target=c1)
        drivers_card("Why the twin is concerned", drivers_html(stw.explain(stp["day"]), tw=stw, day=stp["day"]), target=c2)
    elif not stp.get("whatif"):
        chart_card("Outlook over time", risk_chart(stw, stp["day"], 60, 140, 290), f"story_chart_{i}")
    if stp.get("cards"):
        st.write("")
        html("<div class='sec'>Arjun's body rhythm</div><div class='lbl' style='margin:-.35rem 0 .6rem'>Last 30 days. "
             "The shaded band is his normal range.</div>")
        signal_cards(stw, stp["day"])

# =============================================================================
# Twin
# =============================================================================
elif page == P_TWIN:
    p = patient_row(pid)
    nm = name_of(pid)
    meta = [f"{p['cohort'].title()}, {p['age']}", p["city"], f"{p['prior_episodes']} prior episode"
            + ("s" if p["prior_episodes"] != 1 else ""), f"family history: {'yes' if p['family_history_depression'] else 'no'}"]
    if tw.state(day)["stressor"]:
        meta.append("exam / work-crunch period")
    page_header(f"{nm}'s twin", " · ".join(meta), avatar(pid))
    hero(tw, day)
    st.write("")
    c1, c2 = st.columns([1.55, 1])
    chart_card("Outlook over time", risk_chart(tw, day, gt=show_gt), "twin_chart",
               "Chance an episode begins in the next 7 days", target=c1)
    drivers_card("What's pushing the risk up", drivers_html(tw.explain(day), tw=tw, day=day),
                 "Longer bar = stronger push on the estimate (SHAP).", target=c2)
    st.write("")
    html(f"<div class='sec'>Body rhythm</div><div class='lbl' style='margin:-.35rem 0 .6rem'>Last 30 days. The shaded "
         f"band is {nm}'s normal range.</div>")
    signal_cards(tw, day)

# =============================================================================
# What-if
# =============================================================================
elif page == P_WHATIF:
    nm = name_of(pid)
    WI = dict(wi_sleep=0.0, wi_bed=0.0, wi_steps=0, wi_screen=0, wi_social=0)
    for k, v in WI.items():
        st.session_state.setdefault(k, v)

    def preset(**kv):
        for k, v in WI.items():
            st.session_state[k] = kv.get(k, v)

    page_header(f"Try a different week for {nm}",
                f"Change the last 7 days and watch the twin re-run. Date: "
                f"{pd.Timestamp(tw.daily.loc[day, 'date']).strftime('%d %b %Y')} (change it in the sidebar).", avatar(pid))
    b = st.columns(4)
    b[0].button("Sleep more", on_click=preset, kwargs=dict(wi_sleep=1.0, wi_bed=-0.75), width="stretch")
    b[1].button("Move more", on_click=preset, kwargs=dict(wi_steps=3000), width="stretch")
    b[2].button("Full healthy week", on_click=preset, type="primary",
                kwargs=dict(wi_sleep=1.0, wi_bed=-0.75, wi_steps=3000, wi_screen=-45, wi_social=5), width="stretch")
    b[3].button("Reset", on_click=preset, width="stretch")
    st.write("")
    c1, c2 = st.columns([1, 1.15])
    with c1:
        with st.container(key="card_sliders"):
            html("<div class='sec'>Adjust the week</div>")
            st.slider("Sleep per night (hours)", -2.0, 2.0, step=0.25, key="wi_sleep")
            st.slider("Bedtime (hours, minus = earlier)", -2.0, 2.0, step=0.25, key="wi_bed")
            st.slider("Steps per day", -4000, 6000, step=500, key="wi_steps")
            st.slider("Late-night phone (minutes)", -120, 120, step=5, key="wi_screen")
            st.slider("Social contacts per day", -15, 15, step=1, key="wi_social")
    res = tw.what_if(day, sleep_hours=st.session_state.wi_sleep, sleep_onset_shift_h=st.session_state.wi_bed,
                     steps=st.session_state.wi_steps, night_screen_min=st.session_state.wi_screen,
                     social_contacts=st.session_state.wi_social)
    with c2:
        html(whatif_card(res))
        st.write("")
        drivers_card("What would still push risk up",
                     drivers_html(res["drivers_after"], 4, tw=tw, day=day, feats=res["feats_after"]))
    st.write("")
    callout("This is a <b>model-based experiment</b> on the twin: it shows how the estimate responds if these signals "
            "had looked different. It is not a guaranteed causal effect, and it is meant to support a conversation "
            "with a counsellor, not replace one.", "warn")

# =============================================================================
# Clinician
# =============================================================================
elif page == P_CLIN:
    date_txt = pd.Timestamp(tw.daily.loc[day, "date"]).strftime("%d %b %Y")
    r = A["risk"][A["risk"].day == day].merge(A["patients"][["patient_id", "cohort", "age", "sex", "city"]], on="patient_id")
    r = r[r.patient_id.isin(pool)].copy()
    for cid, c in st.session_state.get("custom_clients", {}).items():
        ct = c["twin"]
        f = ct.feats.loc[ct.feats.day == day].iloc[0]
        r = pd.concat([r, pd.DataFrame([dict(patient_id=cid, day=day, risk=f["risk"], twin_deviation_index=f["twin_deviation_index"],
                                            cohort=ct.patient["cohort"], age=ct.patient["age"], sex=ct.patient["sex"],
                                            city=ct.patient["city"])])], ignore_index=True)
    r = r.dropna(subset=["risk"])
    prev = A["risk"][A["risk"].day == day - 1][["patient_id", "risk"]].rename(columns={"risk": "prev"})
    r = r.merge(prev, on="patient_id", how="left")
    for cid, c in st.session_state.get("custom_clients", {}).items():
        r.loc[r.patient_id == cid, "prev"] = c["twin"].prev_risk(day)
    enc = A["enc"]
    phq = enc[(enc.day <= day) & enc.phq9.notna()].sort_values("day").groupby("patient_id").agg(last_phq9=("phq9", "last"))
    r = r.merge(phq, on="patient_id", how="left")
    r["band"] = [tw.band(a, b) for a, b in zip(r.risk, r.prev)]
    page_header(f"Caseload · {date_txt}", "Everyone the service supports, sorted by today's outlook. "
                "Change the date in the sidebar.")
    with st.expander("Enrol a demo client (synthetic) and watch the twin learn them"):
        html("<div class='lbl' style='margin-bottom:.4rem'>Choose a profile. A synthetic client with 240 days of wearable "
             "data and a health record is generated, and their twin is built on the spot.</div>")
        e1, e2, e3, e4 = st.columns(4)
        nm_new = e1.text_input("Display name", value="Meera")
        coh = e2.selectbox("Profile", ["student", "professional"])
        age_new = e3.number_input("Age", 18, 40, 24)
        sex_new = e4.selectbox("Sex", ["F", "M"])
        e1, e2, e3, e4 = st.columns(4)
        city_new = e1.selectbox("City", C.CITIES)
        prior_new = e2.number_input("Prior episodes", 0, 5, 1)
        fam_new = e3.checkbox("Family history of depression", value=False)
        vul_new = e4.slider("Underlying vulnerability", 0.0, 1.0, 0.45, 0.05,
                            help="Simulator parameter: how easily stress tips this person into an episode.")
        if st.button("Enrol client and build twin", type="primary"):
            cc = st.session_state.setdefault("custom_clients", {})
            new_id = f"NEW{len(cc) + 1:04d}"
            pr, dd, ee = simulate_custom(new_id, seed=int(time.time()) % 100000, cohort=coh, age=int(age_new), sex=sex_new,
                                         city=city_new, prior_episodes=int(prior_new), family_history_depression=bool(fam_new),
                                         vulnerability=float(vul_new))
            cc[new_id] = {"twin": DigitalTwin(new_id, assets(), frames=dict(patient=pr, daily=dd, enc=ee)),
                          "name": (nm_new.strip() or "New client")}
            st.session_state[f"day_{new_id}"] = 150
            st.session_state["pending_client"] = new_id
            st.session_state["pending_page"] = P_TWIN
            st.rerun()
    st.write("")
    k = st.columns(4)
    kpi(k[0], len(r), "Clients supported", color=P["ink"])
    kpi(k[1], int((r.band == "Early warning").sum()), "Early warning", "contact within 48 h", P["rose"])
    kpi(k[2], int((r.band == "Watch").sum()), "Watch", "light-touch check-in", P["peach"])
    kpi(k[3], int((r.band == "Stable").sum()), "Stable", "routine", P["mint"])
    st.write("")
    top = r.sort_values("risk", ascending=False).head(12)
    rows = ""
    for t in top.itertuples():
        phq_txt = "–" if pd.isna(t.last_phq9) else f"{t.last_phq9:.0f}"
        rows += (f"<tr><td><div style='display:flex;align-items:center;gap:.6rem'><div class='avatar' style='width:32px;"
                 f"height:32px;font-size:.9rem'>{name_of(t.patient_id)[:1]}</div><div><b>{name_of(t.patient_id)}</b><br>"
                 f"<span class='muted' style='font-size:.78rem'>{t.patient_id}</span></div></div></td>"
                 f"<td>{status_chip(t.band)}</td><td class='serif' style='font-size:1.1rem'>{100 * t.risk:.1f}%</td>"
                 f"<td>{t.twin_deviation_index:+.2f}</td><td>{phq_txt}</td>"
                 f"<td class='muted'>{t.cohort.title()}, {t.age} · {t.city}</td></tr>")
    c1, c2 = st.columns([1.5, 1])
    html(f"<div class='card'><div class='sec'>Who needs attention today</div><table class='mt'><tr><th>Client</th>"
         f"<th>Status</th><th>Risk</th><th>Drift</th><th>PHQ-9</th><th>Profile</th></tr>{rows}</table>"
         f"<div class='foot' style='margin-top:.5rem'>Top 12 of {len(r)} shown. Drift = how far the last week is from the "
         "client's own normal (0 = usual).</div></div>", c1)
    with c2:
        ids = top.patient_id.tolist()
        rec = st.selectbox("Open a health record", ids, index=ids.index(pid) if pid in ids else 0,
                           format_func=lambda x: f"{name_of(x)} · {x}")
        rtw = twin(rec)
        rs_ = rtw.state(day)
        s_ = STATUS[rs_["band"]]
        html(f"<div class='card'><div style='display:flex;gap:.8rem;align-items:center'>{avatar(rec)}"
             f"<div style='flex:1'><div class='serif' style='font-size:1.35rem'>{name_of(rec)}</div>"
             f"<div class='lbl'>{s_['sky']} · {100 * rs_['risk']:.1f}% risk</div></div>{status_chip(rs_['band'])}</div>"
             f"<div class='muted' style='margin:.8rem 0 .3rem;font-size:.92rem;line-height:1.55'>{summary_sentence(rtw, day)}</div>"
             f"<div style='border-top:1px solid var(--border);margin:.7rem 0 .3rem'></div>"
             f"<div class='sec' style='font-size:1.05rem'>Health record</div>{ehr_timeline(rtw, day)}</div>")
        st.write("")
        bundle = build_bundle(rec, day, A["patients"], A["daily"], A["enc"])
        with st.expander(f"FHIR record ({len(bundle['entry'])} resources) for ABDM-style exchange"):
            st.download_button("Download FHIR JSON", json.dumps(bundle, indent=2), file_name=f"{rec}_day{day}_fhir.json")
            st.code(json.dumps(bundle, indent=2)[:3000] + "\n...", language="json")

# =============================================================================
# Live
# =============================================================================
elif page == P_LIVE:
    nm = name_of(pid)
    page_header(f"Live feed · {nm}", "Replays the watch and phone data day by day, as if it were arriving now. "
                "The twin re-learns and re-scores every day.", avatar(pid))
    c1, c2, c3 = st.columns(3)
    start = c1.number_input("Start at day", C.MIN_HISTORY_DAYS, tw.n_days - 2, max(C.MIN_HISTORY_DAYS, day - 15))
    n = c2.slider("Days to replay", 5, 40, 20)
    speed = c3.select_slider("Speed", ["slow", "normal", "fast"], value="normal")
    delay = {"slow": 0.8, "normal": 0.35, "fast": 0.12}[speed]
    go_btn = st.button("Start live replay", type="primary")
    hero_ph, tick_ph, chart_ph = st.empty(), st.empty(), st.empty()
    lo = max(C.MIN_HISTORY_DAYS, int(start) - 20)
    if not go_btn:
        hero(tw, int(start), hero_ph)
        show_chart(risk_chart(tw, int(start), lo, int(start) + n, 290, show_gt), key="live_idle", target=chart_ph)
    else:
        for t in range(int(start), min(int(start) + n, tw.n_days)):
            d = tw.daily.loc[t]
            hero(tw, t, hero_ph)
            items = [("sleep_hours", "Sleep"), ("sleep_onset", "Bedtime"), ("hrv_rmssd", "HRV"), ("steps", "Steps"),
                     ("night_screen_min", "Late-night phone")]
            html("<div class='num-row'>" + "".join(
                f"<div class='tile' style='background:{P[SIG_TONE[s] + '_bg']}'><div class='h'><span>{lab} today</span>"
                f"{icon(SIG_ICON[s], 16, P[SIG_TONE[s]])}</div><div class='v' style='font-size:1.45rem'>{fmt_val(s, d[s])}</div></div>"
                for s, lab in items) + "</div>", tick_ph)
            show_chart(risk_chart(tw, t, lo, int(start) + n, 290, show_gt), key=f"live_{t}", target=chart_ph)
            time.sleep(delay)
        callout("Replay finished. Move the sidebar date to explore any day.", "ok")

# =============================================================================
# Evidence
# =============================================================================
elif page == P_PERF:
    m = json.loads((C.REPORT_DIR / "metrics.json").read_text())
    t, p = m["twin"], m["population"]
    page_header("Does it work?", "Part 1: tested on 180 synthetic clients the model never saw during training, against a "
                "one-size-fits-all model that ignores each person's own normal.")
    k = st.columns(4)
    kpi(k[0], f"{t['episodes_detected']} of {t['episodes_evaluated']}", "episodes flagged before they began",
        f"one-size-fits-all: {p['episodes_detected']} of {p['episodes_evaluated']}")
    kpi(k[1], f"{t['median_lead_time_days']:.0f} days", "typical warning before onset", "time for a counsellor to act")
    kpi(k[2], f"{t['false_alert_days_per_30_quiet_days']:.1f}", "false-alarm days per month",
        f"one-size-fits-all: {p['false_alert_days_per_30_quiet_days']:.1f}")
    kpi(k[3], f"{t['auroc']:.2f}", "AUROC (0.5 = coin flip, 1 = perfect)", f"one-size-fits-all: {p['auroc']:.2f}")
    st.write("")
    c1, c2 = st.columns(2)
    comp = [("Episodes flagged early (%)", 100 * t["episode_detection_rate"], 100 * p["episode_detection_rate"], "{:.0f}%"),
            ("False-alarm days / month ×10", 10 * t["false_alert_days_per_30_quiet_days"], 10 * p["false_alert_days_per_30_quiet_days"], "{:.0f}"),
            ("AUROC ×100", 100 * t["auroc"], 100 * p["auroc"], "{:.0f}")]
    twin_col = P["violet"] if MODE == "dark" else P["navy2"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Manas Twin (personal normal)", y=[c[0] for c in comp], x=[c[1] for c in comp], orientation="h",
                         marker_color=twin_col, text=[c[3].format(c[1]) for c in comp], textposition="outside"))
    fig.add_trace(go.Bar(name="One-size-fits-all", y=[c[0] for c in comp], x=[c[2] for c in comp], orientation="h",
                         marker_color=P["border"] if MODE == "light" else "#3A3A58",
                         text=[c[3].format(c[2]) for c in comp], textposition="outside"))
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=-0.12), bargap=0.35)
    fig.update_xaxes(visible=False, range=[0, 105])
    fig.update_yaxes(autorange="reversed", showgrid=False)
    chart_card("Personal normal vs. one-size-fits-all", style_fig(fig, 320, legend=True), "perf_cmp", target=c1)
    imp = pd.read_csv(C.REPORT_DIR / "shap_global_importance.csv", index_col=0)["mean_abs_shap"].head(10)[::-1]
    fig2 = go.Figure(go.Bar(x=imp.values, y=[readable(f) for f in imp.index], orientation="h", marker_color=P["lav"]))
    fig2.update_xaxes(title="average influence (mean |SHAP|)")
    fig2.update_yaxes(showgrid=False)
    chart_card("What the twin pays most attention to", style_fig(fig2, 320), "perf_shap", target=c2)
    st.write("")
    callout("<b>Reading this honestly:</b> about half of episodes are flagged in advance, typically 6 days ahead, with "
            "roughly one false-alarm day a month. That makes the twin a <b>prompt for a human check-in</b>, not a "
            f"diagnosis. All results come from a synthetic cohort ({m['data']['patients']} people × "
            f"{m['data']['days_per_patient']} days); see the real-data check below.", "warn")
    st.write("")
    html("<div class='ph' style='margin-top:1.4rem'><div><h1 style='font-size:1.7rem !important'>Real-data check</h1>"
         "<div class='sub'>Part 2: the public DEPRESJON dataset (Garcia-Ceja et al., 2018). Minute-level wrist activity from "
         "23 people with depression (MADRS-scored) and 32 controls, about 13 days each. No health record and no episode "
         "onsets, so it cannot train the full twin, but it can test whether the twin's rhythm features carry real "
         "signal.</div></div></div>")
    dv_path = C.REPORT_DIR / "depresjon_validation.json"
    if dv_path.exists():
        dv = json.loads(dv_path.read_text())
        q1 = dv["q1_person_level"]
        k = st.columns(4)
        kpi(k[0], f"{q1['auroc_lopo']:.2f}", "AUROC: depressed vs control",
            f"95% CI {q1['ci95'][0]:.2f}–{q1['ci95'][1]:.2f}, leave-one-person-out")
        kpi(k[1], f"{100 * q1['sensitivity_at_spec80']:.0f}%", "of depressed people identified", "at 80% specificity")
        kpi(k[2], f"{dv['people']}", "real people", f"{dv['depressed']} depressed · {dv['people'] - dv['depressed']} controls")
        kpi(k[3], f"{dv['complete_days']:,}", "complete days of actigraphy",
            f"median {dv['days_per_person_median']:.0f} days per person")
        st.write("")
        c1, c2 = st.columns([1.1, 1])
        uni = dv["q1_univariate"]
        rows = ""
        for f, lab in [("mean_activity", "Mean daily activity"), ("rest_minutes", "Minutes at rest per day"),
                       ("day_to_day_var", "Day-to-day variability"), ("night_share", "Night-time activity share"),
                       ("rel_amplitude", "Day–night contrast"), ("intradaily_var", "Fragmentation of rhythm")]:
            u = uni[f]
            sig = icon("check", 15, P["mint"]) if u["p_value"] < 0.05 else ""
            rows += (f"<tr><td>{lab}</td><td>{u['depressed_median']:.3g}</td><td>{u['control_median']:.3g}</td>"
                     f"<td>{u['p_value']:.3f} {sig}</td></tr>")
        html(f"<div class='card'><div class='sec'>Rhythm features by group</div><div class='lbl' style='margin:-.3rem 0 .4rem'>"
             f"Medians</div><table class='mt'><tr><th>Feature</th><th>Depressed</th><th>Control</th><th>p (Mann-Whitney)</th>"
             f"</tr>{rows}</table><div class='foot' style='margin-top:.5rem'>Ticked: p &lt; 0.05. People with depression moved "
             "less, rested more and varied less from day to day, in line with the digital-phenotyping literature.</div></div>", c1)
        ex = dv["examples"]
        fig3 = go.Figure()
        for pid_, colr, lab in [("control_3", P["mint"], "Control (real person)"), ("condition_2", P["rose"], "Depressed (real person)")]:
            e = ex[pid_]
            fig3.add_trace(go.Scatter(x=list(range(1, len(e["mean_activity"]) + 1)), y=e["mean_activity"], mode="lines+markers",
                                      name=lab, line=dict(color=colr, width=2, shape="spline", smoothing=0.5), marker=dict(size=5)))
        fig3.update_xaxes(title="day of recording", showgrid=False)
        fig3.update_yaxes(title="mean activity count")
        fig3.update_layout(legend=dict(orientation="h", y=-0.3))
        chart_card("Two real people, two weeks of wrist activity", style_fig(fig3, 300, legend=True), "dep_examples", target=c2)
        st.write("")
        callout("<b>Honest reading:</b> real wearable data carries a clear between-person depression signal, which supports the "
                "features the twin is built on. What DEPRESJON cannot test is the <i>within-person</i> early-warning step "
                "(no episode onsets, about 13 days per person), and symptom-severity correlations were not significant with "
                "23 people. That is exactly what the planned longitudinal pilot is for.", "info")
    else:
        callout("Run <code>python -m src.validate_depresjon</code> after downloading DEPRESJON to generate this section.", "warn")

# =============================================================================
# About
# =============================================================================
elif page == P_ABOUT:
    page_header("What is Manas Twin?", "Manas (मनस्) means mind.")
    html("<div class='card story'>A <b>digital twin</b> is a living virtual copy of something real that stays in sync "
         "with it. Manas Twin is a virtual copy of one person's daily rhythm (sleep, heart, movement, phone use and "
         "social contact), paired with their health record. Like a weather forecast for the mind, it notices when the "
         "person drifts away from <b>their own normal</b> and warns about a week before a depressive episode is likely "
         "to begin, while there is still time to help.</div>")
    st.write("")
    steps = [("sprout", "mint", "Learn", "Learns your normal from 60+ days of watch and phone data."),
             ("compass", "lav", "Mirror", "Compares your last week with your own normal, every day."),
             ("trend", "rose", "Predict", "Estimates the chance an episode starts in the next 7 days."),
             ("bulb", "peach", "Explain", "Shows the reasons in plain language."),
             ("sliders", "sky", "Simulate", "Lets you test what a better week would change.")]
    for c, (ic, tone, h, d) in zip(st.columns(5), steps):
        html(f"<div class='tile' style='background:{P[tone + '_bg']};min-height:170px'><div class='h'><span>{h}</span>"
             f"{icon(ic, 20, P[tone])}</div><div class='muted' style='font-size:.88rem;margin-top:.7rem;line-height:1.5'>"
             f"{d}</div></div>", c)
    st.write("")
    html("<div class='sec'>Built to be safe</div>")
    guard = [("lock", "Consent first", "Explicit, revocable consent in line with India's DPDP Act 2023 and ABDM."),
             ("users", "Human in the loop", "The twin alerts. A counsellor or clinician decides. It never diagnoses."),
             ("ban", "Narrow scope", "Predicts depressive-episode onset only. It does not assess suicide or self-harm risk."),
             ("eye-off", "Privacy", "Pseudonymous IDs. Counts, never message content. Aggregate-only views for institutions.")]
    for c, (ic, h, d) in zip(st.columns(4), guard):
        html(f"<div class='card' style='min-height:160px'><div style='width:38px;height:38px;border-radius:50%;"
             f"background:{P['lav_bg']};display:flex;align-items:center;justify-content:center'>{icon(ic, 18, P['lav'])}</div>"
             f"<div style='font-weight:600;margin:.6rem 0 .2rem'>{h}</div><div class='muted' style='font-size:.88rem;"
             f"line-height:1.5'>{d}</div></div>", c)
    st.write("")
    callout(f"If you are in distress or thinking about harming yourself, please reach out now: <b>{C.HELPLINE}</b>, "
            "or go to the nearest emergency department.", "danger")

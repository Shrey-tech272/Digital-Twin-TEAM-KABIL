#!/usr/bin/env bash
# Regenerate data, retrain, export a sample FHIR bundle and launch the dashboard.
set -e
python -m src.simulator
python -m src.train
python -m src.fhir MT0441 140
streamlit run app/streamlit_app.py

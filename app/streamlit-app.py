import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺")
st.title("Diabetes Risk Predictor")
st.caption("Enter patient measurements, then send them to the prediction API.")

with st.form("patient_form"):
    glucose = st.number_input("Glucose (mg/dL)", min_value=0.1, max_value=500.0, value=120.0)
    blood_pressure = st.number_input("Blood pressure (mmHg)", min_value=0.1, max_value=250.0, value=70.0)
    skin_thickness = st.number_input("Skin thickness (mm)", min_value=0.0, max_value=100.0, value=20.0)
    insulin = st.number_input("Insulin (µU/mL)", min_value=0.0, max_value=1000.0, value=80.0)
    bmi = st.number_input("BMI (kg/m²)", min_value=0.1, max_value=100.0, value=25.0)
    pedigree = st.number_input("Diabetes pedigree function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.number_input("Age (years)", min_value=0.1, max_value=120.0, value=35.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()
        risk = result.get("risk", "Unknown")
        badge = result.get("badge_color", "gray")
        advice = result.get("advice", "")

        if risk == "High":
            st.error(f"Risk: {risk}")
        else:
            st.success(f"Risk: {risk}")

        st.write(advice)
        st.caption(f"Badge color: {badge}")
    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {API_URL}: {exc}")

import streamlit as st
import numpy as np
import joblib

model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.markdown("Fill in the details to check your heart disease risk.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120, 45)
        resting_bp = st.number_input("Resting Blood Pressure", 50, 200, 120)
        cholesterol = st.number_input("Cholesterol", 100, 400, 200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
        max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
        oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0, 1.0)

    with col2:
        sex = st.selectbox("Sex", ["Female", "Male"])
        sex_m = 1 if sex == "Male" else 0

        cp_type = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
        cp_asym = 1 if cp_type == "ASY" else 0
        cp_nap = 1 if cp_type == "NAP" else 0
        cp_ta = 1 if cp_type == "TA" else 0

        restecg = st.selectbox("Resting ECG", ["Normal", "ST"])
        restecg_st = 1 if restecg == "ST" else 0
        restecg_normal = 1 if restecg == "Normal" else 0

        exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        exercise_angina_y = 1 if exercise_angina == "Yes" else 0

        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
        slope_flat = 1 if st_slope == "Flat" else 0
        slope_down = 1 if st_slope == "Down" else 0

    submit = st.form_submit_button("Predict")

if submit:
    features = np.array([[age, resting_bp, cholesterol, fasting_bs, max_hr, oldpeak,
                          sex_m, cp_asym, cp_nap, cp_ta,restecg_normal, restecg_st,
                          exercise_angina_y, slope_flat, slope_down]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    st.subheader("Result:")
    if prediction == 1:
        st.error("⚠️ High risk of heart disease. Please consult a doctor.")
    else:
        st.success("✅ Low risk of heart disease.")

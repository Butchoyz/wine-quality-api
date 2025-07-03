import streamlit as st
import joblib
import numpy as np

# Load model and imputers
model = joblib.load("xgb_wine_model.pkl")
imputer_mean = joblib.load("imputer_mean_density.pkl")
imputer_median = joblib.load("imputer_median_citric_pH.pkl")

st.set_page_config(page_title="Wine Quality Predictor 🍷", layout="centered")
st.title("🍷 Wine Quality Predictor")

with st.form("wine_form"):
    fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0)
    volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0)
    citric_acid = st.number_input("Citric Acid", min_value=0.0)
    residual_sugar = st.number_input("Residual Sugar", min_value=0.0)
    chlorides = st.number_input("Chlorides", min_value=0.0)
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0)
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0)
    density = st.number_input("Density", min_value=0.0)
    pH = st.number_input("pH", min_value=0.0)
    sulphates = st.number_input("Sulphates", min_value=0.0)
    alcohol = st.number_input("Alcohol", min_value=0.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                                chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                                density, pH, sulphates, alcohol]])
        input_data[:, 2:3] = imputer_median.transform(input_data[:, 2:3])
        input_data[:, 7:8] = imputer_mean.transform(input_data[:, 7:8])
        input_data[:, 8:9] = imputer_median.transform(input_data[:, 8:9])

        prediction = model.predict(input_data)[0]
        confidence = model.predict_proba(input_data)[0][prediction]

        result = "Good" if prediction == 1 else "Not Good"
        st.success(f"Prediction: **{result}**")
        st.info(f"Confidence: **{round(float(confidence) * 100, 2)}%**")
    except Exception as e:
        st.error(f"Error: {e}")

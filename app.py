
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Predicción Precio Casas - Boston", layout="centered")
st.title("Predicción - Precio de Casas (Boston)")

model = joblib.load("best_model.pkl")
scaler = joblib.load("minmax_scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")
feature_ranges = joblib.load("feature_ranges.pkl")

st.write("Introduce las características para predecir el precio (medv):")

vals = {}
for col in feature_columns:
    lo = float(feature_ranges[col]['min'])
    hi = float(feature_ranges[col]['max'])
    default = float((lo + hi) / 2)
    vals[col] = st.number_input(col, value=default, min_value=lo, max_value=hi, format="%.5f")

input_df = pd.DataFrame([vals], columns=feature_columns)
input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_columns)

if st.button("Predecir precio"):
    pred = model.predict(input_scaled)[0]
    st.success(f"Precio estimado (medv): {pred:.2f} (miles de USD)")


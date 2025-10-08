
mport streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==============================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================
st.set_page_config(page_title="Predicción Precio Casas - Boston", layout="centered")
st.title("🏠 Predicción del Precio de Casas (Boston)")

st.write("Ingrese las características para predecir el precio medio de una vivienda (MEDV):")

# ==============================
# CARGA DE ARCHIVOS DEL MODELO
# ==============================
try:
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("minmax_scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    feature_ranges = joblib.load("feature_ranges.pkl")
except FileNotFoundError:
    st.error("❌ No se encontraron los archivos del modelo. Verifique que los .pkl estén en el mismo directorio.")
    st.stop()

# ==============================
# NOMBRES AMIGABLES PARA VARIABLES
# ==============================
nombres_amigables = {
    'crim': 'Tasa de criminalidad per cápita',
    'zn': 'Proporción de zonas residenciales (ZN)',
    'indus': 'Proporción de zonas industriales (%)',
    'chas': 'Límite con río Charles (1 = sí, 0 = no)',
    'nox': 'Contaminación por óxidos nítricos (NOX)',
    'rm': 'Promedio de habitaciones por vivienda (RM)',
    'age': 'Casas construidas antes de 1940 (%)',
    'dis': 'Distancia ponderada a centros de empleo (DIS)',
    'rad': 'Índice de acceso a autopistas radiales (RAD)',
    'tax': 'Tasa de impuestos sobre la propiedad (TAX)',
    'ptratio': 'Relación alumno/profesor (PTRATIO)',
    'black': 'Índice población afroamericana (BLACK)',
    'lstat': 'Porcentaje de población de bajo estatus (LSTAT)'
}

# ==============================
# ENTRADA DE USUARIO
# ==============================
vals = {}
for col in feature_columns:
    lo = float(feature_ranges[col]['min'])
    hi = float(feature_ranges[col]['max'])
    default = float((lo + hi) / 2)
    label = nombres_amigables.get(col, col)
    vals[col] = st.number_input(
        label,
        min_value=lo,
        max_value=hi,
        value=default,
        format="%.3f"
    )

# Convertir a DataFrame
input_df = pd.DataFrame([vals], columns=feature_columns)

# Escalar las entradas antes de predecir
input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_columns)

# ==============================
# PREDICCIÓN
# ==============================
if st.button("🔮 Predecir precio"):
    pred = model.predict(input_scaled)[0]

    st.markdown("---")
    st.markdown("### 🏡 **Resultado de la predicción:**")
    st.success(f"**Precio estimado:** ${pred * 1000:,.2f} USD")

    st.markdown(
        "<p style='text-align:center; color:gray;'>"
        "El precio está expresado en miles de dólares (MEDV × 1000).<br>"
        "Modelo entrenado con el conjunto de datos de Boston Housing."
        "</p>",
        unsafe_allow_html=True
    )

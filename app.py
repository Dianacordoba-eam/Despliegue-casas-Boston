import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==============================
# CONFIGURACIÓN DE LA APP
# ==============================
st.set_page_config(page_title="Predicción Precio Casas - Boston", layout="centered")
st.title("🏠 Predicción del Precio de Casas (Boston)")
st.write("Ingrese las características para predecir el precio medio de una vivienda (MEDV):")

# ==============================
# CARGA DE MODELOS Y ESCALADOR
# ==============================
model = joblib.load("best_model.pkl")
scaler = joblib.load("minmax_scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")
feature_ranges = joblib.load("feature_ranges.pkl")

# ==============================
# CAPTURA DE ENTRADAS
# ==============================
vals = {}

st.sidebar.header("💡 Guía de valores típicos")
st.sidebar.write("""
**Rangos aproximados del dataset:**
- crim: 0.0 – 88.0  
- zn: 0.0 – 100.0  
- indus: 0.5 – 27.0  
- chas: 0 o 1  
- nox: 0.38 – 0.87  
- rm: 3.5 – 8.8  
- age: 2 – 100  
- dis: 1.1 – 12.1  
- rad: 1 – 24  
- tax: 190 – 711  
- ptratio: 12 – 22  
- black: 0 – 396  
- lstat: 1 – 38  
""")

for col in feature_columns:
    # Convertir correctamente los rangos (por seguridad)
    lo = float(feature_ranges[col]['min'])
    hi = float(feature_ranges[col]['max'])
    default = float(np.mean([lo, hi]))
    
    vals[col] = st.number_input(
        f"{col}",
        value=default,
        min_value=lo,
        max_value=hi,
        format="%.3f"
    )

# ==============================
# PREDICCIÓN
# ==============================
input_df = pd.DataFrame([vals], columns=feature_columns)
input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_columns)

if st.button("🔮 Predecir precio"):
    pred = model.predict(input_scaled)[0]
    st.success(f"🏡 Precio estimado: ${pred*1000:,.2f} USD")
    st.caption("(*Los valores están expresados en miles de USD según el dataset original*)")

    # Mostrar resumen de entrada
    with st.expander("Ver valores ingresados"):
        st.dataframe(input_df)

!pip install streamlit pyngrok joblib scikit-learn pandas numpy -q
from pyngrok import ngrok

# Crear túnel a Streamlit
public_url = ngrok.connect(8501)
print("🌍 URL pública de tu app:", public_url)

# Ejecutar app
!streamlit run app.py --server.port 8501

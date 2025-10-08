# =====================================================
# ENTRENAMIENTO MODELO PRECIO CASAS BOSTON - FINAL
# =====================================================

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =====================================================
# 1. Cargar dataset
# =====================================================
df = pd.read_csv("/content/Precios_Casas_Boston.csv")

print("✅ Datos cargados correctamente.")
print("Columnas disponibles:", df.columns.tolist())
print(df.head())

# =====================================================
# 2. Definir variable objetivo y predictoras
# =====================================================
target = 'medv'
X = df.drop(columns=[target])
y = df[target]

# =====================================================
# 3. División de datos
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================================================
# 4. Escalado
# =====================================================
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# 5. Entrenar modelo
# =====================================================
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# =====================================================
# 6. Evaluación del modelo
# =====================================================
y_pred = model.predict(X_test_scaled)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"✅ Modelo entrenado con éxito:")
print(f"RMSE: {rmse:.3f}")
print(f"R²: {r2:.3f}")

# =====================================================
# 7. Guardar archivos
# =====================================================
joblib.dump(model, "best_model.pkl")
joblib.dump(scaler, "minmax_scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")
feature_ranges = X.agg(['min', 'max']).to_dict()
joblib.dump(feature_ranges, "feature_ranges.pkl")

print("✅ Archivos guardados correctamente:")
print("- best_model.pkl")
print("- minmax_scaler.pkl")
print("- feature_columns.pkl")
print("- feature_ranges.pkl")
✅ Al final de este paso, descarga los archivos:

best_model.pkl

minmax_scaler.pkl

feature_columns.pkl

feature_ranges.pkl

🏠 2️⃣ — CÓDIGO DE TU APLICACIÓN STREAMLIT (app.py)
Guarda este archivo junto con los .pkl.
👉 Este es el código corregido, limpio y funcional para desplegar.

python
Copiar código
import streamlit as st
import pandas as pd
import joblib

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
    lo = float(feature_ranges[col]['min'])
    hi = float(feature_ranges[col]['max'])
    default = float((lo + hi) / 2)
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

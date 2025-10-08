
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Cargar dataset
df = pd.read_csv("/content/Precios_Casas_Boston.csv")

# Verificar nombres de columnas
print("Columnas del dataset:", df.columns.tolist())

# Definir variable objetivo y predictoras
target = 'medv'  # precio medio de la vivienda
X = df.drop(columns=[target])
y = df[target]

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalado
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar modelo
model = RandomForestRegressor(random_state=42, n_estimators=200)
model.fit(X_train_scaled, y_train)

# Evaluar modelo
y_pred = model.predict(X_test_scaled)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"✅ Modelo entrenado correctamente")
print(f"RMSE: {rmse:.3f}")
print(f"R²: {r2:.3f}")

# Guardar archivos correctamente sincronizados
joblib.dump(model, "best_model.pkl")
joblib.dump(scaler, "minmax_scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

# Guardar rangos originales (sin escalar)
feature_ranges = X.agg(['min', 'max']).to_dict()
joblib.dump(feature_ranges, "feature_ranges.pkl")

print("✅ Archivos guardados correctamente:")
print("- best_model.pkl")
print("- minmax_scaler.pkl")
print("- feature_columns.pkl")
print("- feature_ranges.pkl")

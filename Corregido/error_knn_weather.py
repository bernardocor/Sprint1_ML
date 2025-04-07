
"""
Recomendación: utilizar un linter como 'pycodestyle' o 'black' 
para validar el cumplimiento del estándar PEP8 antes de entregas.
Instalación sugerida: pip install pycodestyle black
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Carga de los datos
data_path = 'SeoulBi.csv'
df = pd.read_csv(data_path, encoding='ISO-8859-1')

# Renombrar columnas para simplificar el manejo
df.rename(
    columns={
        'Temperature(°C)': 'Temperature_C',
        'Dew point temperature(°C)': 'Dew_point_temp_C'
    },
    inplace=True
)

# Selección de variables relevantes
features = [
    'Temperature_C', 'Humidity(%)', 'Wind speed (m/s)',
    'Visibility (10m)', 'Dew_point_temp_C',
    'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)'
]
target = 'Rented Bike Count'

X = df[features]
y = df[target]

# División en conjunto entrenamiento-prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Definición de valores de k a evaluar
k_values = [3, 5, 10, 15, 20, 50, 100, 300, 500, 1000]
errors = []

# Entrenamiento y evaluación del modelo KNN para cada k
for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    errors.append(rmse)

# Identificación del mejor valor de k
best_k = k_values[np.argmin(errors)]
best_rmse = min(errors)

# Gráfica RMSE vs. Número de Vecinos (k)
plt.figure(figsize=(10, 5))
plt.plot(k_values, errors, marker='o', linestyle='-', color='b')
plt.xlabel('Número de Vecinos (k)')
plt.ylabel('RMSE en el conjunto de prueba')
plt.title('Error RMSE vs Número de Vecinos (k)')
plt.grid(True)
plt.tight_layout()
plt.savefig('error_knn_weather.png')
plt.show()

# Mostrar resultados
print(f"El menor error RMSE es {best_rmse:.2f} obtenido con k={best_k}")

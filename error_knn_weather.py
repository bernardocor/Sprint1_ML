import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Cargar los datos
data_path = 'SeoulBi.csv'
df = pd.read_csv(data_path, encoding='ISO-8859-1')

# Renombrar columnas para evitar caracteres especiales
df.rename(columns={
    'Temperature(°C)': 'Temperature_C',
    'Dew point temperature(°C)': 'Dew_point_temperature_C'
}, inplace=True)

# Seleccionar las variables relevantes
features = ['Temperature_C', 'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)', 'Dew_point_temperature_C', 
            'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)']
target = 'Rented Bike Count'

X = df[features]
y = df[target]

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir los valores de k
k_values = [3, 5, 10, 15, 20, 50, 100, 300, 500, 1000]
errors = []

# Entrenar y evaluar modelos KNN
for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    errors.append(rmse)

# Encontrar el menor error y su k correspondiente
best_k = k_values[np.argmin(errors)]
best_rmse = min(errors)

# Graficar los resultados
plt.figure(figsize=(10, 5))
plt.plot(k_values, errors, marker='o', linestyle='-', color='b')
plt.xlabel('Número de Vecinos (k)')
plt.ylabel('RMSE en el conjunto de prueba')
plt.title('Error RMSE vs Número de Vecinos (k)')
plt.grid()
plt.savefig('error_knn_weather.png')
plt.show()

print(f"El menor error RMSE es {best_rmse:.2f} y se obtiene con k={best_k}")

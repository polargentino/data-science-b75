"""
Explicación Detallada:

1-Cargar los Datos:

Utiliza pandas para leer el archivo CSV y cargar los datos en un DataFrame.

2-Preparar los Datos:

Selecciona las columnas predictoras relevantes.
Selecciona la variable objetivo (precio de la vivienda).

3-Dividir los Datos en Conjuntos de Entrenamiento y Validación:

Divide los datos en conjuntos de entrenamiento y validación para evaluar 
el rendimiento del modelo.

4-Crear y Entrenar el Modelo XGBoost:

Crea un modelo XGBRegressor con los parámetros ajustados para un 
mejor rendimiento y eficiencia.
Entrena el modelo utilizando la función fit(), con parada temprana para 
evitar el sobreajuste y optimizar el número de árboles.

5-Realizar Predicciones:

Realiza predicciones utilizando el modelo entrenado sobre el conjunto de validación.

6-Evaluar el Modelo:

Calcula el error absoluto medio (MAE) para evaluar la precisión de las 
predicciones del modelo.
Imprime el valor del MAE.
Este código te proporciona un modelo XGBoost completo y ajustado, listo 
para ser utilizado. Recuerda que puedes ajustar aún más los parámetros 
para adaptarlos a tus necesidades específicas.
"""
# import xgboost
# print(xgboost.__version__)

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping  # Importación específica
from sklearn.metrics import mean_absolute_error

# 1. Cargar datos
data = pd.read_csv('/home/pol/Downloads/melb_data.csv')

# 2. Preparación de datos
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
target = 'Price'

# Eliminar filas con valores faltantes
data_clean = data[cols_to_use + [target]].dropna()
X = data_clean[cols_to_use]
y = data_clean[target]

# 3. Dividir datos
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
)

# 4. Crear modelo
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    n_jobs=4,
    objective='reg:squarederror',
    eval_metric='mae',  # Métrica definida aquí
    random_state=0
)

# 5. Configurar parada temprana (con callback)
early_stop = EarlyStopping(
    rounds=5,  # Rondas sin mejora
    metric_name='mae',  # Métrica a monitorear
    data_name='validation_0',  # Nombre del conjunto de validación
    save_best=True  # Guardar mejor modelo
)

# 6. Entrenar con callback
model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    callbacks=[early_stop],  # Usar callback aquí
    verbose=False
)

# 7. Evaluar
predictions = model.predict(X_valid)
mae = mean_absolute_error(y_valid, predictions)
print(f"Error Absoluto Medio: ${mae:,.0f}")

# 8. Resultados finales
print(f"Iteraciones óptimas: {model.best_iteration}")
print(f"Mejor puntuación: {model.best_score:.2f}")

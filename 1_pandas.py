import pandas as pd
# pandas es una biblioteca fundamental para la manipulación y análisis de datos en Python.
# Proporciona estructuras de datos como DataFrames, que son tablas bidimensionales con 
# filas y columnas, y Series, que son arrays unidimensionales etiquetados.
# En este contexto, pandas se utiliza para cargar el archivo CSV (melb_data.csv), 
# manipular los datos, seleccionar columnas y realizar operaciones de limpieza de datos.
# as pd se usa para crear un alias, permitiendo que te refieras a la biblioteca 
# como pd en el resto del código, lo que hace que el código sea más conciso.

from sklearn.model_selection import train_test_split
# sklearn (scikit-learn) es una biblioteca de aprendizaje automático muy popular en Python.
# model_selection es un módulo dentro de sklearn que proporciona herramientas para la 
# selección de modelos y la evaluación del rendimiento.
# train_test_split es una función que divide un conjunto de datos en dos subconjuntos: 
# un conjunto de entrenamiento y un conjunto de prueba (o validación).
# Esto es esencial para evaluar el rendimiento de un modelo de aprendizaje automático, 
# ya que permite entrenar el modelo en un subconjunto de datos y probar su rendimiento 
# en un subconjunto diferente.

from sklearn.ensemble import RandomForestRegressor
# ensemble es un módulo dentro de sklearn que contiene algoritmos de aprendizaje 
# automático de conjunto, que combinan múltiples modelos para mejorar el rendimiento.
# RandomForestRegressor es una clase que implementa el algoritmo de bosque aleatorio 
# para problemas de regresión (predicción de valores numéricos).
# En este caso, se utiliza para construir un modelo que predice el precio de las viviendas

from sklearn.metrics import mean_absolute_error
# metrics es un módulo dentro de sklearn que proporciona funciones para evaluar 
# el rendimiento de los modelos de aprendizaje automático.
# mean_absolute_error (MAE) es una función que calcula el error absoluto medio 
# entre las predicciones del modelo y los valores reales.
# Se utiliza para medir la precisión del modelo de regresión.

from sklearn.impute import SimpleImputer
# impute es un módulo dentro de sklearn que proporciona herramientas para 
# imputar (rellenar) valores faltantes en los datos.
# SimpleImputer es una clase que implementa estrategias de imputación simples, 
# como reemplazar los valores faltantes con la media, la mediana o el valor más frecuente.
# Se utiliza para manejar los valores faltantes en el conjunto de datos.

# Ruta del archivo CSV
file_path = '/home/pol/Downloads/melb_data.csv'

# Cargar los datos
data = pd.read_csv(file_path)

# Seleccionar la variable objetivo (el precio de la vivienda)
y = data.Price

# Seleccionar las variables predictoras numéricas (excluyendo la variable objetivo y las columnas no numéricas)
melb_predictors = data.drop(['Price'], axis=1)
X = melb_predictors.select_dtypes(exclude=['object'])

# Dividir los datos en conjuntos de entrenamiento (80%) y validación (20%)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Función para comparar diferentes enfoques (calcula el Error Absoluto Medio - MAE)
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0) # Modelo de Bosque Aleatorio con 10 árboles
    model.fit(X_train, y_train) # Entrena el modelo con los datos de entrenamiento
    preds = model.predict(X_valid) # Realiza predicciones con los datos de validación
    return mean_absolute_error(y_valid, preds) # Calcula el MAE entre las predicciones y los valores reales

# Enfoque 1: Eliminar columnas con valores faltantes
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()] # Identifica las columnas con valores faltantes
reduced_X_train = X_train.drop(cols_with_missing, axis=1) # Elimina las columnas faltantes del conjunto de entrenamiento
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1) # Elimina las columnas faltantes del conjunto de validación
print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid)) # Calcula y muestra el MAE

# Enfoque 2: Imputación (reemplazar valores faltantes con la media)
my_imputer = SimpleImputer() # Crea un objeto SimpleImputer
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train)) # Imputa los valores faltantes en el conjunto de entrenamiento
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid)) # Imputa los valores faltantes en el conjunto de validación
imputed_X_train.columns = X_train.columns # Restaura los nombres de las columnas
imputed_X_valid.columns = X_valid.columns # Restaura los nombres de las columnas
print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid)) # Calcula y muestra el MAE

# Enfoque 3: Extensión de la imputación (imputación + columnas indicadoras de valores faltantes)
X_train_plus = X_train.copy() # Crea copias de los conjuntos de entrenamiento y validación
X_valid_plus = X_valid.copy()
for col in cols_with_missing: # Para cada columna con valores faltantes
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull() # Crea una nueva columna indicadora de valores faltantes
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()
my_imputer = SimpleImputer() # Crea un objeto SimpleImputer
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus)) # Imputa los valores faltantes en el conjunto de entrenamiento modificado
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus)) # Imputa los valores faltantes en el conjunto de validación modificado
imputed_X_train_plus.columns = X_train_plus.columns # Restaura los nombres de las columnas
imputed_X_valid_plus.columns = X_valid_plus.columns # Restaura los nombres de las columnas
print("MAE from Approach 3 (An Extension to Imputation):")
print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid)) # Calcula y muestra el MAE

# Imprimir información sobre los valores faltantes
print(X_train.shape) # Imprime la forma del conjunto de entrenamiento (número de filas, número de columnas)
missing_val_count_by_column = (X_train.isnull().sum()) # Calcula el número de valores faltantes en cada columna
print(missing_val_count_by_column[missing_val_count_by_column > 0]) # Imprime las columnas con valores faltantes y su conteo


# MAE from Approach 1 (Drop columns with missing values):
# 183550.22137772635

# MAE from Approach 2 (Imputation):
# 178166.46269899711

# MAE from Approach 3 (An Extension to Imputation):
# 178927.503183954

# (10864, 12)
# Car               49
# BuildingArea    5156
# YearBuilt       4307
# dtype: int64

# Análisis de los Resultados:

# MAE from Approach 1 (Drop columns with missing values): 183550.22137772635

# Este es el Error Absoluto Medio (MAE) cuando se eliminan las columnas 
# con valores faltantes.
# Un MAE de 183550 significa que, en promedio, las predicciones del modelo se 
# desvían en 183550 unidades de la variable objetivo (el precio de la vivienda).

# MAE from Approach 2 (Imputation): 178166.46269899711

# Este es el MAE cuando se imputan los valores faltantes con la media.
# El MAE es menor que en el Enfoque 1 (178166 vs. 183550), lo que indica 
# que la imputación mejora el rendimiento del modelo.

# MAE from Approach 3 (An Extension to Imputation): 178927.503183954

# Este es el MAE cuando se imputan los valores faltantes y se agregan 
# columnas indicadoras de valores faltantes.
# El MAE es ligeramente mayor que en el Enfoque 2 (178927 vs. 178166), 
# lo que indica que agregar columnas indicadoras no mejoró el rendimiento en este caso.
# (10864, 12)

# Esta es la forma del conjunto de entrenamiento: 10864 filas y 12 columnas.
# Car 49, BuildingArea 5156, YearBuilt 4307

# Estas son las columnas con valores faltantes y el número de valores 
# faltantes en cada columna.
# Esto muestra que 'BuildingArea' y 'YearBuilt' tienen una gran cantidad 
# de valores faltantes.

# Conclusión:

# La imputación (Enfoque 2) fue el mejor enfoque en este caso, ya que 
# produjo el MAE más bajo.
# Eliminar columnas con valores faltantes (Enfoque 1) resultó en un 
# rendimiento peor debido a la pérdida de información importante.
# Agregar columnas indicadoras de valores faltantes (Enfoque 3) 
# no mejoró el rendimiento, lo que indica que esta técnica no siempre es beneficiosa.
# En resumen, la imputación de valores faltantes con la media resultó 
# ser la estrategia más efectiva para este conjunto de datos, 
# minimizando el error en las predicciones del modelo de bosque aleatorio.                  
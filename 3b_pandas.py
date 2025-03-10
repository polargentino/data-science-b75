"""
Explicación Detallada:

1-Cargar los Datos:

Utiliza pandas para leer el archivo CSV y cargar los datos en un DataFrame.

2-Preparar los Datos:

Separa la variable objetivo (Price) de las variables predictoras.
Divide los datos en conjuntos de entrenamiento y validación para evaluar el 
rendimiento del modelo.

3-Seleccionar Columnas Relevantes:

Identifica las columnas categóricas y numéricas que se utilizarán en el modelo.
Crea copias de los conjuntos de entrenamiento y validación con solo las 
columnas seleccionadas.

4-Preprocesamiento de Datos Numéricos:

Crea un transformador (SimpleImputer) para manejar los valores faltantes en 
las columnas numéricas.

5-Preprocesamiento de Datos Categóricos:

Crea un transformador (Pipeline) que incluye:
SimpleImputer para manejar los valores faltantes en las columnas categóricas.
OneHotEncoder para convertir las columnas categóricas en columnas numéricas 
usando codificación one-hot.

6-Combinar Preprocesamiento:

Combina los transformadores numéricos y categóricos en un solo 
preprocesador (ColumnTransformer) para aplicar diferentes 
transformaciones a diferentes columnas.

7-Definir el Modelo:

Crea un modelo de bosque aleatorio (RandomForestRegressor) para la regresión.

8-Crear la Pipeline:

Crea una pipeline (Pipeline) que encadena el preprocesador y el modelo.

9-Entrenar el Modelo:

Entrena la pipeline con los datos de entrenamiento. La pipeline aplica 
automáticamente el preprocesamiento y entrena el modelo.

10-Realizar Predicciones:

Realiza predicciones con los datos de validación. La pipeline 
aplica automáticamente el preprocesamiento a los datos de validación
antes de realizar las predicciones.

11-Evaluar el Modelo:

Calcula el error absoluto medio (MAE) para evaluar el rendimiento del modelo.

"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. Cargar los Datos
# Lee el archivo CSV con los datos de viviendas de Melbourne
data = pd.read_csv('/home/pol/Downloads/melb_data.csv')

# 2. Preparar los Datos
# Separa la variable objetivo (precio de la vivienda) de las variables predictoras
y = data.Price
X = data.drop(['Price'], axis=1)

# Divide los datos en conjuntos de entrenamiento y validación (80% entrenamiento, 20% validación)
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# 3. Seleccionar Columnas Relevantes
# Identifica las columnas categóricas con baja cardinalidad (menos de 10 valores únicos)
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"]

# Identifica las columnas numéricas
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Combina las columnas categóricas y numéricas seleccionadas
my_cols = categorical_cols + numerical_cols

# Crea copias de los conjuntos de entrenamiento y validación con solo las columnas seleccionadas
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

# 4. Preprocesamiento de Datos Numéricos
# Crea un transformador para imputar valores faltantes en columnas numéricas con una constante (puedes cambiar la estrategia si lo deseas)
numerical_transformer = SimpleImputer(strategy='constant')

# 5. Preprocesamiento de Datos Categóricos
# Crea un transformador para imputar valores faltantes en columnas categóricas con el valor más frecuente
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Aplica codificación one-hot a las columnas categóricas
])

# 6. Combinar Preprocesamiento
# Combina los transformadores numéricos y categóricos en un solo preprocesador usando ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# 7. Definir el Modelo
# Crea un modelo de bosque aleatorio para la regresión
model = RandomForestRegressor(n_estimators=100, random_state=0)

# 8. Crear la Pipeline
# Crea una pipeline que combina el preprocesador y el modelo
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# 9. Entrenar el Modelo
# Entrena la pipeline con los datos de entrenamiento
my_pipeline.fit(X_train, y_train)

# 10. Realizar Predicciones
# Realiza predicciones con los datos de validación
preds = my_pipeline.predict(X_valid)

# 11. Evaluar el Modelo
# Calcula el error absoluto medio (MAE) para evaluar el rendimiento del modelo
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)

# salida : MAE: 160679.18917034855

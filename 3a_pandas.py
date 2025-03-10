"""
Mejoras implementadas:

Optimización del modelo:
Aumentamos n_estimators a 200
Agregamos max_depth=15 para controlar el sobreajuste
Nuevo enfoque (Target Encoding):
Usa información de la variable objetivo para codificar
Mejora el rendimiento con variables categóricas
Automatización:
Detección automática de columnas categóricas
Manejo consistente de índices en transformaciones
"""

# Importar bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from category_encoders import TargetEncoder  # Necesitarás instalar: pip install category_encoders

# 1. CARGA DE DATOS
# Leer los datos desde un archivo CSV
data = pd.read_csv('/home/pol/Downloads/melb_data.csv')

# 2. PREPARACIÓN INICIAL
# Separar variable objetivo (Precio) de las características predictoras
y = data.Price  # Variable a predecir
X = data.drop(['Price'], axis=1)  # Todas las demás columnas como predictores

# Dividir datos en entrenamiento (80%) y validación (20%)
X_train_full, X_valid_full, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
)

# 3. LIMPIEZA DE DATOS
# Eliminar columnas con valores faltantes (enfoque simple)
cols_with_missing = [col for col in X_train_full.columns 
                    if X_train_full[col].isnull().any()]
X_train_full.drop(cols_with_missing, axis=1, inplace=True)
X_valid_full.drop(cols_with_missing, axis=1, inplace=True)

# 4. SELECCIÓN DE CARACTERÍSTICAS
# Seleccionar columnas categóricas con baja cardinalidad (<10 categorías)
low_cardinality_cols = [col for col in X_train_full.columns 
                        if X_train_full[col].nunique() < 10 
                        and X_train_full[col].dtype == "object"]

# Seleccionar columnas numéricas
numerical_cols = [col for col in X_train_full.columns 
                  if X_train_full[col].dtype in ['int64', 'float64']]

# Combinar columnas seleccionadas
my_cols = low_cardinality_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

# 5. ANÁLISIS EXPLORATORIO
# Mostrar primeras filas de los datos de entrenamiento
print("Datos de entrenamiento procesados:")
print(X_train.head())

# Identificar variables categóricas
object_cols = list(X_train.select_dtypes(include='object').columns)
print("\nVariables categóricas:", object_cols)

# 6. FUNCIÓN DE EVALUACIÓN
def score_dataset(X_train, X_valid, y_train, y_valid):
    """Entrena un RandomForest y devuelve el MAE (Error Absoluto Medio)"""
    model = RandomForestRegressor(
        n_estimators=200,  # Aumentamos de 100 a 200 árboles
        max_depth=15,      # Profundidad máxima de los árboles
        random_state=0
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

# 7. ENFOQUES DE CODIFICACIÓN
# === Enfoque 1: Eliminar variables categóricas ===
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])
print("\nMAE (Eliminar categóricas):", 
      score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))

# === Enfoque 2: Codificación Ordinal ===
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])
print("MAE (Codificación Ordinal):", 
      score_dataset(label_X_train, label_X_valid, y_train, y_valid))

# === Enfoque 3: One-Hot Encoding ===
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# Mantener índices originales
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Combinar con variables numéricas
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# Asegurar nombres de columnas válidos
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)

print("MAE (One-Hot Encoding):", 
      score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))

# === Enfoque 4: Target Encoding (Nueva mejora) ===
target_encoder = TargetEncoder()
encoded_X_train = target_encoder.fit_transform(X_train, y_train)
encoded_X_valid = target_encoder.transform(X_valid)

print("MAE (Target Encoding):", 
      score_dataset(encoded_X_train, encoded_X_valid, y_train, y_valid))


# Datos de entrenamiento procesados:
#       Type Method             Regionname  Rooms  Distance  ...  Bathroom  Landsize  Lattitude  Longtitude  Propertycount
# 12167    u      S  Southern Metropolitan      1       5.0  ...       1.0       0.0  -37.85984    144.9867        13240.0
# 6524     h     SA   Western Metropolitan      2       8.0  ...       2.0     193.0  -37.85800    144.9005         6380.0
# 8413     h      S   Western Metropolitan      3      12.6  ...       1.0     555.0  -37.79880    144.8220         3755.0
# 2919     u     SP  Northern Metropolitan      3      13.0  ...       1.0     265.0  -37.70830    144.9158         8870.0
# 6043     h      S   Western Metropolitan      3      13.3  ...       1.0     673.0  -37.76230    144.8272         4217.0

# [5 rows x 12 columns]

# Variables categóricas: ['Type', 'Method', 'Regionname']

# MAE (Eliminar categóricas): 176668.37004911096
# MAE (Codificación Ordinal): 166807.33732477558
# MAE (One-Hot Encoding): 166387.02438765
# MAE (Target Encoding): 166240.23329934917
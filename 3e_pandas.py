import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# 1. Cargar los Datos
data = pd.read_csv('/home/pol/Downloads/melb_data.csv')

# 2. Preparar los Datos
y = data.Price
X = data.drop(['Price'], axis=1)

# 3. Dividir los Datos en Conjuntos de Entrenamiento y Validación
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Eliminar columnas con valores faltantes (enfoque simple)
cols_with_missing = [col for col in X_train_full.columns if X_train_full[col].isnull().any()]
X_train_full.drop(cols_with_missing, axis=1, inplace=True)
X_valid_full.drop(cols_with_missing, axis=1, inplace=True)

# Seleccionar columnas categóricas con baja cardinalidad
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"]

# Seleccionar columnas numéricas
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Mantener solo las columnas seleccionadas
my_cols = low_cardinality_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

# 4. Preprocesamiento de Datos
# Preprocesamiento para datos numéricos
numerical_transformer = SimpleImputer(strategy='mean')

# Preprocesamiento para datos categóricos
categorical_transformer = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore')
)

# Combinar preprocesamiento usando ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, low_cardinality_cols)
    ])

# 5. Crear la Pipeline Completa
my_pipeline = make_pipeline(preprocessor, RandomForestRegressor(n_estimators=100, random_state=0))

# 6. Evaluar el Modelo
cv_scores = cross_val_score(my_pipeline, X_train, y_train,
                            cv=5,
                            scoring='neg_mean_absolute_error')

print("Cross-validation MAE: %f" % (-1 * cv_scores.mean()))

# Cross-validation MAE: 172334.877627
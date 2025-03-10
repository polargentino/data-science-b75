import pandas as pd
from sklearn.model_selection import train_test_split

# Leer los datos
data = pd.read_csv('/home/pol/Downloads/melb_data.csv')


# Separar la variable objetivo de las predictoras
y = data.Price
X = data.drop(['Price'], axis=1)

# Dividir los datos en conjuntos de entrenamiento y validación
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Eliminar columnas con valores faltantes (enfoque simple)
cols_with_missing = [col for col in X_train_full.columns if X_train_full[col].isnull().any()]
X_train_full.drop(cols_with_missing, axis=1, inplace=True)
X_valid_full.drop(cols_with_missing, axis=1, inplace=True)

# Seleccionar columnas categóricas con baja cardinalidad (conveniente pero arbitrario)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"]

# Seleccionar columnas numéricas
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Mantener solo las columnas seleccionadas
my_cols = low_cardinality_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

# Mostrar las primeras filas de los datos de entrenamiento
print(X_train.head())

# Obtener la lista de variables categóricas
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)

# Definir la función para medir la calidad de cada enfoque (MAE)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

# Enfoque 1: Eliminar variables categóricas
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])
print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))

# Enfoque 2: Codificación ordinal
from sklearn.preprocessing import OrdinalEncoder
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])
print("MAE from Approach 2 (Ordinal Encoding):")
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))

# Enfoque 3: Codificación one-hot
from sklearn.preprocessing import OneHotEncoder
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)
print("MAE from Approach 3 (One-Hot Encoding):")
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))

#       Type Method             Regionname  ...  Lattitude  Longtitude  Propertycount
# 12167    u      S  Southern Metropolitan  ...  -37.85984    144.9867        13240.0
# 6524     h     SA   Western Metropolitan  ...  -37.85800    144.9005         6380.0
# 8413     h      S   Western Metropolitan  ...  -37.79880    144.8220         3755.0
# 2919     u     SP  Northern Metropolitan  ...  -37.70830    144.9158         8870.0
# 6043     h      S   Western Metropolitan  ...  -37.76230    144.8272         4217.0

# [5 rows x 12 columns]
# Categorical variables:
# ['Type', 'Method', 'Regionname']
# MAE from Approach 1 (Drop categorical variables):
# 175703.48185157913
# MAE from Approach 2 (Ordinal Encoding):
# 165936.40548390493
# MAE from Approach 3 (One-Hot Encoding):
# 166089.4893009678

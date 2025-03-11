import numpy as np  # Librería para álgebra lineal
import pandas as pd  # Librería para procesamiento de datos (CSV, I/O)
from sklearn.ensemble import RandomForestClassifier  # Modelo de Random Forest

# 1. Cargar los datos de entrenamiento
train_data = pd.read_csv("~/Downloads/titanic/train.csv")
# Mostrar las primeras 5 filas del DataFrame train_data
print("Primeras 5 filas de train.csv:")
print(train_data.head())

# 2. Cargar los datos de prueba
test_data = pd.read_csv("~/Downloads/titanic/test.csv")
# Mostrar las primeras 5 filas del DataFrame test_data
print("\nPrimeras 5 filas de test.csv:")
print(test_data.head())

# 3. Análisis exploratorio de datos (EDA) - Patrón de género
# Calcular la tasa de supervivencia de las mujeres
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women) / len(women)
print("\n% de mujeres que sobrevivieron:", rate_women)

# Calcular la tasa de supervivencia de los hombres
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men) / len(men)
print("% de hombres que sobrevivieron:", rate_men)

# 4. Preparación de los datos para el modelo
# Seleccionar la columna 'Survived' como variable objetivo (y)
y = train_data["Survived"]

# Seleccionar las características (features) para el modelo
features = ["Pclass", "Sex", "SibSp", "Parch"]

# Convertir las características categóricas (Sex) en variables dummy (one-hot encoding)
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# 5. Crear y entrenar el modelo de Random Forest
# Crear una instancia del modelo RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

# Entrenar el modelo con los datos de entrenamiento
model.fit(X, y)

# 6. Generar las predicciones para los datos de prueba
predictions = model.predict(X_test)

# 7. Crear el archivo de envío (submission.csv)
# Crear un DataFrame con los PassengerId y las predicciones
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

# Guardar el DataFrame en un archivo CSV llamado submission.csv
output.to_csv('submission.csv', index=False)

# Imprimir un mensaje de confirmación
print("\nTu envío fue guardado exitosamente!")
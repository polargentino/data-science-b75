import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ruta al archivo train.csv
file_path = "/home/pol/Downloads/train.csv"

# Carga de datos
titanic_data = pd.read_csv(file_path)

# --- Análisis básico por pol---

# Número de pasajeros totales
total = len(titanic_data)
print(f"Total passengers: {total}")

# Número de pasajeros que sobrevivieron
survived = (titanic_data.Survived == 1).sum()
print(f"Survived: {survived}")

# Número de pasajeros menores de 18 años
minors = (titanic_data.Age < 18).sum()
print(f"Minors: {minors}")

# Fracción de sobrevivientes
survived_fraction = survived / total
print(f"Survived fraction: {survived_fraction}")

# Fracción de menores
minors_fraction = minors / total
print(f"Minors fraction: {minors_fraction}")

# --- Visualizaciones ---

# 1. Distribución de sobrevivientes
sns.countplot(x='Survived', data=titanic_data)
plt.title('Survival Count')
plt.show()

# 2. Distribución de sobrevivientes por sexo
sns.countplot(x='Sex', hue='Survived', data=titanic_data)
plt.title('Survival by Sex')
plt.show()

# 3. Distribución de la edad
sns.histplot(titanic_data['Age'].dropna(), bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

# 4. Distribución de la clase del pasajero (Pclass)
sns.countplot(x='Pclass', data=titanic_data)
plt.title('Passenger Class Distribution')
plt.show()

# 5. Tasa de supervivencia por clase de pasajero
sns.barplot(x='Pclass', y='Survived', data=titanic_data)
plt.title('Survival Rate by Pclass')
plt.show()

# 6. Tasa de supervivencia por sexo y clase de pasajero
sns.barplot(x='Sex', y='Survived', hue='Pclass', data=titanic_data)
plt.title('Survival Rate by Sex and Pclass')
plt.show()

# 7. Distribución de tarifas (Fare)
sns.histplot(titanic_data['Fare'], bins=40, kde=True)
plt.title('Fare Distribution')
plt.show()

# 8. Distribución del puerto de embarque (Embarked)
sns.countplot(x='Embarked', data=titanic_data)
plt.title('Embarked Distribution')
plt.show()

# 9. Tasa de supervivencia por puerto de embarque
sns.barplot(x='Embarked', y='Survived', data=titanic_data)
plt.title('Survival Rate by Embarked')
plt.show()
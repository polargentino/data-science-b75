import pandas as pd

# Ruta al archivo train.csv
file_path = "/home/pol/Downloads/train.csv"  # Replace with your actual path if different.

# Carga de datos
titanic_data = pd.read_csv(file_path)

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

# Funciones de ayuda (q5) - Placeholder
class q5:
    @staticmethod
    def check():
        print("Checking...")

    @staticmethod
    def hint():
        print("Hint: Divide the count by the total.")

    @staticmethod
    def solution():
        print("Solution:")
        print(f"survived_fraction = {survived / total}")
        print(f"minors_fraction = {minors / total}")

# Llamadas a las funciones de ayuda (si las tienes)
q5.check()
q5.hint()
q5.solution()
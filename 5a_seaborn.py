"""
Comentarios Detallados:

Cargar y examinar los datos:

insurance_filepath = "~/Downloads/insurance.csv": Define la ruta del archivo insurance.csv. 
¡Asegúrate de ajustar esta ruta a la ubicación real del archivo en tu sistema!

insurance_data = pd.read_csv(insurance_filepath): Lee el archivo CSV y lo carga en un 
DataFrame llamado insurance_data.

insurance_data.head(): Muestra las primeras 5 filas del DataFrame para verificar la carga.
I've added to_markdown to display the dataframe in a more readable format in the output.

Diagrama de dispersión básico:
------------------------------

sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges']): Crea un 
diagrama de dispersión con 'bmi' en el eje x e 'charges' en el eje y.
plt.figure(figsize=(8, 6)): Adjust the figure size to make the plot more readable.
plt.title(): Add relevant titles to the plot, and x and y axis.

Diagrama de dispersión con línea de regresión:
----------------------------------------------

sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges']): Crea un diagrama 
de dispersión con una línea de regresión que muestra la tendencia general de la 
relación entre 'bmi' y 'charges'.
plt.figure(figsize=(8, 6)): Adjust the figure size to make the plot more readable.
plt.title(): Add relevant titles to the plot, and x and y axis.

Diagrama de dispersión con codificación de color:
-------------------------------------------------

sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], 
hue=insurance_data['smoker']): Crea un diagrama de dispersión donde los 
puntos se colorean según la columna 'smoker', permitiendo visualizar la 
influencia del tabaquismo en la relación entre 'bmi' y 'charges'.
plt.figure(figsize=(8, 6)): Adjust the figure size to make the plot more readable.
plt.title(): Add relevant titles to the plot, and x and y axis.
plt.legend(title="Smoker"): Add a title to the legend.

Diagrama de dispersión con líneas de regresión separadas:
--------------------------------------------------------

sns.lmplot(x="bmi", y="charges", hue="smoker", data=insurance_data): Crea un 
diagrama de dispersión con líneas de regresión separadas para fumadores y no 
fumadores, mostrando cómo la relación entre 'bmi' y 'charges' difiere según el tabaquismo.
plt.figure(figsize=(8, 6)): Adjust the figure size to make the plot more readable.
plt.title(): Add relevant titles to the plot, and x and y axis.

Diagrama de dispersión categórico (Swarm Plot):
-----------------------------------------------

sns.swarmplot(x=insurance_data['smoker'], y=insurance_data['charges']): Crea un diagrama de dispersión categórico (swarm plot) que muestra la distribución de 'charges' para fumadores y no fumadores, permitiendo comparar las distribuciones y observar la presencia de outliers.
plt.figure(figsize=(8, 6)): Adjust the figure size to make the plot more readable.
plt.title(): Add relevant titles to the plot, and x and y axis.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Cargar y examinar los datos
# Ruta del archivo insurance.csv
insurance_filepath = "~/Downloads/insurance.csv"  # ¡Asegúrate de que este sea el path correcto!

# Leer el archivo CSV en un DataFrame llamado insurance_data
insurance_data = pd.read_csv(insurance_filepath)

# Mostrar las primeras 5 filas del DataFrame para verificar la carga
print("Primeras 5 filas del DataFrame:")
print(insurance_data.head().to_markdown(index=False, numalign="left", stralign="left"))  # Use to_markdown for better output

# 2. Diagrama de dispersión básico
# Crear un diagrama de dispersión para visualizar la relación entre 'bmi' y 'charges'
plt.figure(figsize=(8, 6))  # Adjust figure size for better visualization
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])
plt.title("Relación entre BMI y Charges")  # Add a title for clarity
plt.xlabel("BMI (Body Mass Index)")  # Add x-axis label
plt.ylabel("Charges (Insurance Costs)")  # Add y-axis label
plt.show()

# 3. Diagrama de dispersión con línea de regresión
# Crear un diagrama de dispersión con una línea de regresión para visualizar la relación entre 'bmi' y 'charges'
plt.figure(figsize=(8, 6))  # Adjust figure size
sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges'])
plt.title("Relación entre BMI y Charges con Línea de Regresión")  # Add a title
plt.xlabel("BMI (Body Mass Index)")  # Add x-axis label
plt.ylabel("Charges (Insurance Costs)")  # Add y-axis label
plt.show()

# 4. Diagrama de dispersión con codificación de color
# Crear un diagrama de dispersión con codificación de color para visualizar la relación entre 'bmi', 'charges' y 'smoker'
plt.figure(figsize=(8, 6))  # Adjust figure size
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker'])
plt.title("Relación entre BMI, Charges y Smoker")  # Add a title
plt.xlabel("BMI (Body Mass Index)")  # Add x-axis label
plt.ylabel("Charges (Insurance Costs)")  # Add y-axis label
plt.legend(title="Smoker")  # Add legend title
plt.show()

# 5. Diagrama de dispersión con líneas de regresión separadas
# Crear un diagrama de dispersión con líneas de regresión separadas para fumadores y no fumadores
plt.figure(figsize=(8, 6))  # Adjust figure size
sns.lmplot(x="bmi", y="charges", hue="smoker", data=insurance_data)
plt.title("Relación entre BMI y Charges por Smoker con Líneas de Regresión")  # Add a title
plt.xlabel("BMI (Body Mass Index)")  # Add x-axis label
plt.ylabel("Charges (Insurance Costs)")  # Add y-axis label
plt.show()

# 6. Diagrama de dispersión categórico (Swarm Plot)
# Crear un diagrama de dispersión categórico (swarm plot) para visualizar la distribución de 'charges' por 'smoker'
plt.figure(figsize=(8, 6))  # Adjust figure size
sns.swarmplot(x=insurance_data['smoker'], y=insurance_data['charges'])
plt.title("Distribución de Charges por Smoker (Swarm Plot)")  # Add a title
plt.xlabel("Smoker")  # Add x-axis label
plt.ylabel("Charges (Insurance Costs)")  # Add y-axis label
plt.show()

# Salidas : (más gráficos)
# Primeras 5 filas del DataFrame:
# | age   | sex    | bmi    | children   | smoker   | region    | charges   |
# |:------|:-------|:-------|:-----------|:---------|:----------|:----------|
# | 19    | female | 27.9   | 0          | yes      | southwest | 16884.9   |
# | 18    | male   | 33.77  | 1          | no       | southeast | 1725.55   |
# | 28    | male   | 33     | 3          | no       | southeast | 4449.46   |
# | 33    | male   | 22.705 | 0          | no       | northwest | 21984.5   |
# | 32    | male   | 28.88  | 0          | no       | northwest | 3866.86   |
# /home/pol/Desktop/data-science-b75/env/lib/python3.10/site-packages/seaborn/categorical.py:3399: UserWarning: 14.6% of the points cannot be placed; you may want to decrease the size of the markers or use stripplot.
#   warnings.warn(msg, UserWarning)
# /home/pol/Desktop/data-science-b75/env/lib/python3.10/site-packages/seaborn/categorical.py:3399: UserWarning: 45.5% of the points cannot be placed; you may want to decrease the size of the markers or use stripplot.
#   warnings.warn(msg, UserWarning)
# /home/pol/Desktop/data-science-b75/env/lib/python3.10/site-packages/seaborn/categorical.py:3399: UserWarning: 45.6% of the points cannot be placed; you may want to decrease the size of the markers or use stripplot.
#   warnings.warn(msg, UserWarning)

# Analísis de Gemini foto insurance_1.png: Conclusiones:

# El swarmplot muestra claramente que el tabaquismo está fuertemente 
# asociado con costos de seguro más altos.
# Los fumadores no solo pagan más en promedio, sino que también tienen 
# una mayor variabilidad en sus costos.
# Este gráfico es una herramienta efectiva para visualizar la 
# distribución de una variable numérica en relación con una variable 
# categórica y para identificar diferencias significativas entre grupos.
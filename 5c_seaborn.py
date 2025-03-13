"""
Comentarios Detallados:

Cargar y examinar los datos:

iris_filepath = "~/Downloads/iris.csv": Define la ruta del archivo iris.csv. 
¡Asegúrate de ajustar esta ruta a la ubicación real del archivo en tu sistema!
iris_data = pd.read_csv(iris_filepath, index_col="Id"): Lee el archivo CSV y lo 
carga en un DataFrame llamado iris_data, usando la columna 'Id' como índice.
iris_data.head(): Muestra las primeras 5 filas del DataFrame.
I've added to_markdown to display the dataframe in a more readable format in the output.

Histograma de la longitud del pétalo:
-------------------------------------

sns.histplot(iris_data['Petal Length (cm)']): Crea un histograma de la 
columna 'Petal Length (cm)'.
plt.title(), plt.xlabel(), plt.ylabel(): Añaden títulos y etiquetas a los ejes.

Gráfico de densidad de la longitud del pétalo:
---------------------------------------------

sns.kdeplot(data=iris_data['Petal Length (cm)'], fill=True): Crea un gráfico de 
densidad de la columna 'Petal Length (cm)'.
Nota: Se utiliza fill=True en lugar de shade=True porque shade está obsoleto en 
versiones recientes de seaborn.

Gráfico de densidad 2D:
-----------------------

sns.jointplot(x=iris_data['Petal Length (cm)'], y=iris_data['Sepal Width (cm)'], kind="kde"): Crea un gráfico de densidad 2D que muestra la relación entre 'Petal Length (cm)' y 'Sepal Width (cm)'.
kind="kde": Especifica que se debe usar un gráfico de densidad 2D.
plt.suptitle(): Añade un título al jointplot.

Histogramas por especie:
-----------------------

sns.histplot(data=iris_data, x='Petal Length (cm)', hue='Species'): Crea histogramas 
de 'Petal Length (cm)' para cada especie de Iris.
hue='Species': Separa los histogramas por la variable 'Species'.

Gráficos de densidad por especie:
---------------------------------

sns.kdeplot(data=iris_data, x='Petal Length (cm)', hue='Species', fill=True): Crea 
gráficos de densidad de 'Petal Length (cm)' para cada especie de Iris.
hue='Species': Separa los gráficos de densidad por la variable 'Species'.
fill=True: Rellena el área bajo la curva de densidad.

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Cargar y examinar los datos
iris_filepath = "~/Downloads/iris.csv"  # ¡Asegúrate de ajustar esta ruta!
iris_data = pd.read_csv(iris_filepath, index_col="Id")

print("Primeras 5 filas del DataFrame:")
print(iris_data.head().to_markdown(index=False, numalign="left", stralign="left"))  # Use to_markdown for better output

# 2. Histograma de la longitud del pétalo
plt.figure(figsize=(8, 6))
sns.histplot(iris_data['Petal Length (cm)'])
plt.title("Histograma de Longitud del Pétalo")
plt.xlabel("Longitud del Pétalo (cm)")
plt.ylabel("Conteo")
plt.show()

# 3. Gráfico de densidad de la longitud del pétalo
plt.figure(figsize=(8, 6))
sns.kdeplot(data=iris_data['Petal Length (cm)'], fill=True)  # Usamos fill=True en lugar de shade=True
plt.title("Gráfico de Densidad de Longitud del Pétalo")
plt.xlabel("Longitud del Pétalo (cm)")
plt.ylabel("Densidad")
plt.show()

# 4. Gráfico de densidad 2D
plt.figure(figsize=(8, 6))
sns.jointplot(x=iris_data['Petal Length (cm)'], y=iris_data['Sepal Width (cm)'], kind="kde")
plt.suptitle("Gráfico de Densidad 2D: Longitud del Pétalo vs. Ancho del Sépalo", y=1.02)  # Añadir título al jointplot
plt.show()

# 5. Histogramas por especie
plt.figure(figsize=(10, 6))
sns.histplot(data=iris_data, x='Petal Length (cm)', hue='Species')
plt.title("Histograma de Longitud del Pétalo por Especie")
plt.xlabel("Longitud del Pétalo (cm)")
plt.ylabel("Conteo")
plt.show()

# 6. Gráficos de densidad por especie
plt.figure(figsize=(10, 6))
sns.kdeplot(data=iris_data, x='Petal Length (cm)', hue='Species', fill=True)
plt.title("Gráfico de Densidad de Longitud del Pétalo por Especie")
plt.xlabel("Longitud del Pétalo (cm)")
plt.ylabel("Densidad")
plt.show()

# Salidas : Primeras 5 filas del DataFrame:
# | Sepal Length (cm)   | Sepal Width (cm)   | Petal Length (cm)   | Petal Width (cm)   | Species     |
# |:--------------------|:-------------------|:--------------------|:-------------------|:------------|
# | 5.1                 | 3.5                | 1.4                 | 0.2                | Iris-setosa |
# | 4.9                 | 3                  | 1.4                 | 0.2                | Iris-setosa |
# | 4.7                 | 3.2                | 1.3                 | 0.2                | Iris-setosa |
# | 4.6                 | 3.1                | 1.5                 | 0.2                | Iris-setosa |
# | 5                   | 3.6                | 1.4                 | 0.2                | Iris-setosa |


# Analísis de Gemini foto iris_1.png: ¡Claro! Analicemos la imagen del gráfico de densidad de la longitud del pétalo por especie para complementar nuestro comentario anterior.

# Análisis del Gráfico de Densidad de Longitud del Pétalo por Especie:

# Título: "Gráfico de Densidad de Longitud del Pétalo por Especie"

# El título indica claramente que el gráfico muestra la distribución de la longitud del pétalo para cada una de las tres especies de Iris (setosa, versicolor y virginica).
# Eje X (Longitud del Pétalo (cm)):

# El eje horizontal representa la longitud del pétalo en centímetros, con valores que van desde 1 hasta 8.
# Eje Y (Densidad):

# El eje vertical representa la densidad de probabilidad, que indica la probabilidad relativa de observar una longitud de pétalo particular.
# Distribución de las especies:

# Iris setosa (azul):
# La distribución de setosa es muy estrecha y centrada alrededor de 1.5 cm.
# La densidad es alta en este rango, lo que indica que la mayoría de las flores de setosa tienen longitudes de pétalo muy similares y cortas.
# La distribución cae rápidamente a cero fuera de este rango estrecho.
# Iris versicolor (naranja):
# La distribución de versicolor es más amplia y centrada alrededor de 4.2 cm.
# La densidad es menor en comparación con setosa, lo que indica una mayor variabilidad en la longitud del pétalo.
# La distribución se extiende desde aproximadamente 3 cm hasta 5.5 cm.
# Iris virginica (verde):
# La distribución de virginica es la más amplia y centrada alrededor de 5.5 cm.
# La densidad es aún menor en comparación con versicolor, lo que indica la mayor variabilidad en la longitud del pétalo.
# La distribución se extiende desde aproximadamente 4.5 cm hasta 7 cm.
# Observaciones clave:

# Separación clara: Las tres especies muestran distribuciones de longitud del pétalo claramente separadas.
# Solapamiento mínimo: Hay un solapamiento mínimo entre las distribuciones de versicolor y virginica, pero setosa está completamente separada.
# Identificación: La longitud del pétalo es un buen predictor para identificar la especie de Iris, especialmente para setosa.
# Información adicional:

# El gráfico utiliza áreas sombreadas para representar la densidad, lo que facilita la comparación de las distribuciones.
# La leyenda identifica claramente cada especie con su color correspondiente.
# Conclusiones:

# El gráfico de densidad muestra claramente las diferencias en la distribución de la longitud del pétalo entre las tres especies de Iris.
# Iris setosa se distingue fácilmente por su longitud de pétalo corta y consistente.
# Iris versicolor y virginica tienen longitudes de pétalo más variables y se solapan ligeramente.
# Este gráfico es una herramienta efectiva para visualizar y comparar distribuciones de datos entre diferentes grupos.
# Con este análisis, podemos agregar información valiosa a nuestro comentario sobre el tutorial de distribuciones utilizando el conjunto de datos de Iris.


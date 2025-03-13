"""
Conceptos clave:

Histograma: Un histograma divide los datos en intervalos (bins) y muestra 
----------
la frecuencia (o conteo) de los datos que caen en cada intervalo. Es útil 
para visualizar la distribución de una variable numérica.

Gráfico de densidad: Un gráfico de densidad estima y muestra la función de 
--------------------
densidad de probabilidad de una variable numérica. Proporciona una representación 
suave de la distribución de los datos.

Explicación detallada:

Histograma básico:
------------------

sns.histplot(insurance_data['bmi'], kde=False): Crea un histograma de 
la columna 'bmi'. kde=False evita que se muestre el gráfico de densidad superpuesto.
plt.title(), plt.xlabel(), plt.ylabel(): Añaden títulos y etiquetas a los ejes.

Gráfico de densidad:
-------------------

sns.kdeplot(insurance_data['bmi']): Crea un gráfico de densidad de la columna 'bmi'.

Histograma con densidad:
-----------------------

sns.histplot(insurance_data['bmi'], kde=True): Crea un histograma y superpone un gráfico de densidad.

Histograma por categoría:
------------------------

sns.histplot(data=insurance_data, x='charges', hue='smoker', element='step'): 
Crea histogramas de 'charges' para cada categoría de 'smoker'.
hue='smoker': Separa los histogramas por la variable 'smoker'.
element='step': Muestra el histograma como un contorno.

Gráfico de densidad por categoría:
---------------------------------

sns.kdeplot(data=insurance_data, x='charges', hue='smoker', fill=True): Crea 
gráficos de densidad de 'charges' para cada categoría de 'smoker'.
fill=True: Rellena el área bajo la curva de densidad.

Puntos clave:
------------

Los histogramas y gráficos de densidad son útiles para comprender la distribución 
de datos numéricos.
seaborn proporciona funciones convenientes para crear estos gráficos.
Con los histogramas y los graficos de densidad, es posible visualizar la diferencia 
de distribucion de una variable numerica, a traves de una variable categorica, 
como se muestra en los ultimos 2 ejemplos del codigo.
Asegúrate de ajustar la ruta del archivo insurance.csv en tu sistema.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar los datos
insurance_filepath = "~/Downloads/insurance.csv"  # ¡Asegúrate de ajustar esta ruta!
insurance_data = pd.read_csv(insurance_filepath)

# 1. Histograma de la variable 'bmi'
plt.figure(figsize=(8, 6))
sns.histplot(insurance_data['bmi'], kde=False)  # kde=False para solo el histograma
plt.title("Histograma de BMI")
plt.xlabel("BMI (Body Mass Index)")
plt.ylabel("Frecuencia")
plt.show()

# 2. Gráfico de densidad de la variable 'bmi'
plt.figure(figsize=(8, 6))
sns.kdeplot(insurance_data['bmi'])
plt.title("Gráfico de Densidad de BMI")
plt.xlabel("BMI (Body Mass Index)")
plt.ylabel("Densidad")
plt.show()

# 3. Histograma con gráfico de densidad superpuesto
plt.figure(figsize=(8, 6))
sns.histplot(insurance_data['bmi'], kde=True)  # kde=True para superponer el gráfico de densidad
plt.title("Histograma y Densidad de BMI")
plt.xlabel("BMI (Body Mass Index)")
plt.ylabel("Frecuencia/Densidad")
plt.show()

# 4. Histograma de 'charges' diferenciado por 'smoker'
plt.figure(figsize=(10, 6))
sns.histplot(data=insurance_data, x='charges', hue='smoker', element='step')
plt.title('Histograma de Charges por Smoker')
plt.xlabel('Charges (Costos de Seguro)')
plt.ylabel('Frecuencia')
plt.show()

# 5. Gráfico de densidad de 'charges' diferenciado por 'smoker'
plt.figure(figsize=(10, 6))
sns.kdeplot(data=insurance_data, x='charges', hue='smoker', fill=True)
plt.title('Gráfico de Densidad de Charges por Smoker')
plt.xlabel('Charges (Costos de Seguro)')
plt.ylabel('Densidad')
plt.show()

# Analísis de https://gemini.google.com/; gráfico distribuciones_1.png: 
# ¡Claro! Analicemos el histograma de la imagen que has compartido.

# Análisis del Histograma de BMI:

# Título: "Histograma de BMI"

# El título indica claramente que el gráfico muestra la distribución del Índice de Masa Corporal (BMI).
# Eje X (BMI):

# El eje horizontal representa el BMI (Body Mass Index), con valores que van desde aproximadamente 15 hasta 50.
# Eje Y (Frecuencia):

# El eje vertical representa la frecuencia, que indica cuántos individuos tienen un BMI dentro de cada intervalo (bin) del histograma.
# Forma de la distribución:

# La distribución parece ser aproximadamente simétrica, con una ligera asimetría positiva (cola más larga hacia la derecha).
# Hay un pico central alrededor de 30-35, lo que indica que la mayoría de los individuos en el conjunto de datos tienen un BMI en este rango.
# La distribución se extiende hacia valores más altos de BMI, lo que sugiere la presencia de individuos con sobrepeso u obesidad.
# La distribución también se extiende hacia valores más bajos de BMI, aunque en menor medida.
# Interpretación:

# El histograma muestra que la distribución del BMI en el conjunto de datos es bastante centrada, con una concentración de individuos en el rango de BMI considerado como sobrepeso.
# La ligera asimetría positiva sugiere que hay más individuos con BMI alto que con BMI bajo.
# La presencia de individuos con BMI muy alto o muy bajo podría indicar la presencia de outliers o casos especiales en el conjunto de datos.
# Información adicional:

# Los bins (intervalos) del histograma parecen tener un ancho constante, lo que facilita la comparación de las frecuencias entre los diferentes rangos de BMI.
# Los bordes de los bins están claramente definidos, lo que permite una interpretación precisa de los datos.
# Conclusiones:

# El histograma proporciona una visualización clara de la distribución del BMI en el conjunto de datos.
# La distribución es aproximadamente simétrica, con una ligera asimetría positiva.
# La mayoría de los individuos tienen un BMI en el rango de sobrepeso.
# La presencia de outliers o casos especiales podría ser investigada más a fondo.
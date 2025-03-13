"""
Comentarios Detallados:

1- Configuración de la ruta del archivo:
----------------------------------------

flight_filepath = "~/Downloads/flight_delays.csv": Establece la ruta 
del archivo flight_delays.csv en tu directorio de descargas.

2-Carga del conjunto de datos:
------------------------------
flight_data = pd.read_csv(flight_filepath, index_col="Month"): Carga el 
archivo CSV en un DataFrame llamado flight_data.
index_col="Month": Especifica que la columna "Month" se utilizará como índice del DataFrame.

3-Inspección de los datos:
--------------------------
print(flight_data.to_markdown()): Muestra el DataFrame completo en formato 
Markdown para verificar que se cargó correctamente.

4-Creación del gráfico de barras:
---------------------------------
plt.figure(figsize=(10, 6)): Crea una figura para el gráfico de barras con 
un tamaño específico.
plt.title("Retraso promedio de llegada para vuelos de Spirit Airlines, por mes"): 
Agrega un título al gráfico.
sns.barplot(x=flight_data.index, y=flight_data['NK']): Crea el gráfico de barras 
utilizando la función barplot de Seaborn.
x=flight_data.index: Utiliza los valores del índice (meses) como el eje x.
y=flight_data['NK']: Utiliza los valores de la columna "NK" (Spirit Airlines) como el eje y.
plt.ylabel("Retraso de llegada (en minutos)"): Agrega una etiqueta al eje y.
plt.show(): Muestra el gráfico.

5-Creación del mapa de calor:

plt.figure(figsize=(14, 7)): Crea una figura para el mapa de calor con un tamaño específico.
plt.title("Retraso promedio de llegada para cada aerolínea, por mes"): Agrega 
un título al gráfico.
sns.heatmap(data=flight_data, annot=True): Crea el mapa de calor utilizando la 
función heatmap de Seaborn.
data=flight_data: Utiliza todos los datos del DataFrame para crear el mapa de calor.
annot=True: Muestra los valores numéricos en cada celda del mapa de calor.
plt.xlabel("Aerolínea"): Agrega una etiqueta al eje x.
plt.show(): Muestra el gráfico.

6-Puntos clave:

Este código te muestra cómo crear gráficos de barras y mapas de calor utilizando Seaborn.
Los comentarios detallados te ayudarán a comprender cada paso del proceso.
Recuerda que debes tener instaladas las bibliotecas pandas, matplotlib y seaborn para que este código funcione.
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Configuración de la ruta del archivo
# Ruta del archivo flight_delays.csv en tu directorio de descargas
flight_filepath = "~/Downloads/flight_delays.csv"

# 2. Carga del conjunto de datos
# Utiliza pandas para leer el archivo CSV y cargarlo en un DataFrame llamado flight_data
# index_col="Month" establece la columna "Month" como índice del DataFrame
flight_data = pd.read_csv(flight_filepath, index_col="Month")

# 3. Inspección de los datos
# Muestra el DataFrame completo para verificar que se cargó correctamente
print(flight_data.to_markdown())

# 4. Creación del gráfico de barras (Bar Chart) para Spirit Airlines (NK)
# Crea una figura con un tamaño específico (10 pulgadas de ancho, 6 pulgadas de alto)
plt.figure(figsize=(10, 6))

# Agrega un título al gráfico
plt.title("Retraso promedio de llegada para vuelos de Spirit Airlines, por mes")

# Utiliza seaborn para crear un gráfico de barras
# x=flight_data.index especifica el índice (meses) como el eje x
# y=flight_data['NK'] especifica la columna 'NK' (Spirit Airlines) como el eje y
# sns.barplot(x=flight_data.index, y=flight_data['NK']) # color azul
# Utiliza una paleta de colores diferente (por ejemplo, "husl")
sns.barplot(x=flight_data.index, y=flight_data['NK'], palette="colorblind")
# Otras paletas de colores:

# seaborn ofrece muchas paletas de colores diferentes. Algunas opciones populares son:
# "husl"
# "deep"
# "muted"
# "pastel"
# "bright"
# "dark"
# "colorblind"


# Agrega una etiqueta al eje y (eje vertical)
plt.ylabel("Retraso de llegada (en minutos)")

# Muestra el gráfico
plt.show()

# 5. Creación del mapa de calor (Heatmap) para todos los datos
# Crea una figura con un tamaño específico (14 pulgadas de ancho, 7 pulgadas de alto)
plt.figure(figsize=(14, 7))

# Agrega un título al gráfico
plt.title("Retraso promedio de llegada para cada aerolínea, por mes")

# Utiliza seaborn para crear un mapa de calor
# data=flight_data especifica el DataFrame que contiene los datos
# annot=True muestra los valores numéricos en cada celda
sns.heatmap(data=flight_data, annot=True)

# Agrega una etiqueta al eje x (eje horizontal)
plt.xlabel("Aerolínea")

# Muestra el gráfico
plt.show()

# Salidas: 
# |   Month |       AA |        AS |        B6 |         DL |        EV |         F9 |         HA |        MQ |       NK |       OO |        UA |         US |         VX |        WN |
# |--------:|---------:|----------:|----------:|-----------:|----------:|-----------:|-----------:|----------:|---------:|---------:|----------:|-----------:|-----------:|----------:|
# |       1 |  6.95584 | -0.320888 |  7.34728  | -2.04385   |  8.5375   | 18.3572    |  3.51264   | 18.165    | 11.3981  | 10.8899  |  6.35273  |   3.10746  |  1.4207    |  3.38947  |
# |       2 |  7.5302  | -0.782923 | 18.6577   |  5.61475   | 10.4172   | 27.4242    |  6.02997   | 21.3016   | 16.4745  |  9.58889 |  7.26066  |   7.11446  |  7.78441   |  3.50136  |
# |       3 |  6.69359 | -0.544731 | 10.7413   |  2.07797   |  6.7301   | 20.0749    |  3.46838   | 11.0184   | 10.0391  |  3.18169 |  4.89221  |   3.33079  |  5.34821   |  3.26334  |
# |       4 |  4.93178 | -3.009    |  2.78011  |  0.0833426 |  4.82125  | 12.6404    |  0.0110215 |  5.13123  |  8.76622 |  3.2238  |  4.37609  |   2.66029  |  0.995507  |  2.9964   |
# |       5 |  5.17388 | -1.7164   | -0.709019 |  0.149333  |  7.72429  | 13.0076    |  0.826426  |  5.46679  | 22.3973  |  4.14116 |  6.82769  |   0.681605 |  7.10202   |  5.68078  |
# |       6 |  8.19102 | -0.220621 |  5.04715  |  4.41959   | 13.9528   | 19.713     |  0.882786  |  9.63932  | 35.5615  |  8.33848 | 16.9327   |   5.7663   |  5.77941   | 10.7435   |
# |       7 |  3.87044 |  0.377408 |  5.84145  |  1.20486   |  6.92642  | 14.4645    |  2.00159   |  3.98029  | 14.3524  |  6.79033 | 10.2626   | nan        |  7.13577   | 10.5049   |
# |       8 |  3.19391 |  2.5039   |  9.28095  |  0.653114  |  5.15442  |  9.17574   |  7.44803   |  1.89657  | 20.519   |  5.60669 |  5.01404  | nan        |  5.10622   |  5.53211  |
# |       9 | -1.43273 | -1.8138   |  3.53915  | -3.70338   |  0.851062 |  0.97846   |  3.69692   | -2.16727  |  8.0001  |  1.5309  | -1.79426  | nan        |  0.0709979 | -1.33626  |
# |      10 | -0.58093 | -2.99362  |  3.67679  | -5.01152   |  2.30376  |  0.0821274 |  0.467074  | -3.73505  |  6.81074 |  1.7509  | -2.45654  | nan        |  2.25428   | -0.688851 |
# |      11 |  0.77263 | -1.91652  |  1.4183   | -3.17541   |  4.41593  | 11.1645    | -2.71989   |  0.220061 |  7.54388 |  4.92555 |  0.281064 | nan        |  0.11637   |  0.995684 |
# |      12 |  4.14968 | -1.84668  | 13.8393   |  2.5046    |  6.68518  |  9.34622   | -1.70647   |  0.662486 | 12.7331  | 10.9476  |  7.01208  | nan        | 13.4987    |  6.72089  |
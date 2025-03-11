"""
¡Excelente! ¡Parece que has encontrado el dataset correcto! 
La tabla que muestras, junto con la imagen del gráfico de líneas, 
confirman que este es el conjunto de datos que se utilizó en el tutorial original.

Confirmación de que es el dataset correcto:

1-Columnas de fecha y canciones:

La tabla muestra claramente la columna "Date" y las columnas para cada 
canción ("Shape of You", "Despacito", etc.), que son las columnas 
esenciales para el gráfico de líneas.

2-Datos de transmisiones diarias:

Los valores numéricos en las columnas de las canciones representan las 
transmisiones diarias, que es el tipo de datos que necesitamos para visualizar 
las tendencias a lo largo del tiempo.

2-Gráfico de líneas:

La imagen del grafico muestra las tendencias de las canciones en el tiempo, 
justo lo que se espera de este dataset.

3-Valores NaN:

La presencia de valores "nan" (Not a Number) al principio de algunas columnas 
es esperada, ya que algunas canciones se lanzaron después de la primera fecha 
en el dataset.
Ahora puedes proceder con el código que te proporcioné anteriormente, y 
debería funcionar correctamente para generar el gráfico de líneas.

4-Resumen de los próximos pasos:

Asegúrate de que el archivo spotify.csv esté guardado en tu directorio de descargas.
Ejecuta el código que te di anteriormente.
Observa el gráfico de líneas generado.
Con este dataset, podrás seguir el tutorial original y explorar las 
tendencias de las reproducciones diarias de estas canciones populares.

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Configuración de la ruta del archivo
# Ruta del archivo spotify.csv en tu directorio de descargas
spotify_filepath = "~/Downloads/spotify.csv"

# 2. Carga del conjunto de datos
# Utiliza pandas para leer el archivo CSV y cargarlo en un DataFrame llamado spotify_data
# index_col="Date" establece la columna "Date" como índice del DataFrame
# parse_dates=True asegura que las fechas se interpreten correctamente
spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)

# 3. Inspección de los datos
# Muestra las primeras 5 filas del DataFrame para verificar que se cargó correctamente
print(spotify_data.head().to_markdown(index=True, numalign="left", stralign="left"))

# 4. Creación del gráfico de líneas
# Crea una figura con un tamaño específico (14 pulgadas de ancho, 6 pulgadas de alto)
plt.figure(figsize=(14, 6))

# Agrega un título al gráfico
plt.title("Reproducciones diarias globales de canciones populares en 2017-2018")

# Utiliza seaborn para crear un gráfico de líneas
# data=spotify_data especifica el DataFrame que contiene los datos
sns.lineplot(data=spotify_data)

# Agrega una etiqueta al eje x (eje horizontal)
plt.xlabel("Fecha")

# Muestra el gráfico
plt.show()

# 5. Creación de un subconjunto del gráfico de líneas
# Crea una figura con un tamaño específico (14 pulgadas de ancho, 6 pulgadas de alto)
plt.figure(figsize=(14, 6))

# Agrega un título al gráfico
plt.title("Reproducciones diarias globales de 'Shape of You' y 'Despacito'")

# Utiliza seaborn para crear un gráfico de líneas para la columna 'Shape of You'
# label="Shape of You" agrega una etiqueta a la leyenda
sns.lineplot(data=spotify_data['Shape of You'], label="Shape of You")

# Utiliza seaborn para crear un gráfico de líneas para la columna 'Despacito'
# label="Despacito" agrega una etiqueta a la leyenda
sns.lineplot(data=spotify_data['Despacito'], label="Despacito")

# Agrega una etiqueta al eje x (eje horizontal)
plt.xlabel("Fecha")

# Muestra el gráfico
plt.show()

# Salidas: 
# | Date                | Shape of You   | Despacito   | Something Just Like This   | HUMBLE.   | Unforgettable   |
# |:--------------------|:---------------|:------------|:---------------------------|:----------|:----------------|
# | 2017-01-06 00:00:00 | 1.22871e+07    | nan         | nan                        | nan       | nan             |
# | 2017-01-07 00:00:00 | 1.31903e+07    | nan         | nan                        | nan       | nan             |
# | 2017-01-08 00:00:00 | 1.30999e+07    | nan         | nan                        | nan       | nan             |
# | 2017-01-09 00:00:00 | 1.45064e+07    | nan         | nan                        | nan       | nan             |
# | 2017-01-10 00:00:00 | 1.42756e+07    | nan         | nan                        | nan       | nan             |

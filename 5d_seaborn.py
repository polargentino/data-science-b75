"""
Explicación detallada:

Cargar los datos:
-----------------

spotify_filepath = "~/Downloads/spotify.csv": Define la ruta del archivo spotify.csv. 
¡Asegúrate de ajustar esta ruta a la ubicación real del archivo en tu sistema!
spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True):
Lee el archivo CSV y lo carga en un DataFrame llamado spotify_data, usando 
la columna 'Date' como índice y parseándola como fechas.

Gráfico de líneas con estilo predeterminado:
--------------------------------------------

sns.lineplot(data=spotify_data): Crea un gráfico de líneas con el estilo 
predeterminado de Seaborn.

Gráfico de líneas con estilo personalizado:
------------------------------------------

sns.set_style("dark"): Cambia el estilo a "dark".
sns.lineplot(data=spotify_data): Crea el gráfico de líneas con el nuevo estilo.

Experimentar con otros estilos:
-------------------------------

Se repite el proceso para los estilos "whitegrid" y "ticks".

Restablecer el estilo:
----------------------

sns.set_style("white"): Restablece el estilo predeterminado de Seaborn.

Puntos clave:
------------

La elección del tipo de gráfico depende del tipo de datos y del mensaje 
que quieras transmitir.
Seaborn ofrece varios temas predefinidos para personalizar la apariencia de tus gráficos.
Puedes cambiar el estilo de tus gráficos con la función sns.set_style().
Experimenta con diferentes estilos para encontrar el que mejor se adapte a tus necesidades.

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar los datos de Spotify
spotify_filepath = "~/Downloads/spotify.csv"  # ¡Asegúrate de ajustar esta ruta!
spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)

# 1. Gráfico de líneas con el estilo predeterminado
plt.figure(figsize=(12, 6))
sns.lineplot(data=spotify_data)
plt.title("Gráfico de Líneas con Estilo Predeterminado")
plt.show()

# 2. Gráfico de líneas con el estilo "dark"
sns.set_style("dark")
plt.figure(figsize=(12, 6))
sns.lineplot(data=spotify_data)
plt.title("Gráfico de Líneas con Estilo 'dark'")
plt.show()

# 3. Gráfico de líneas con el estilo "whitegrid"
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))
sns.lineplot(data=spotify_data)
plt.title("Gráfico de Líneas con Estilo 'whitegrid'")
plt.show()

# 4. Gráfico de líneas con el estilo "ticks"
sns.set_style("ticks")
plt.figure(figsize=(12, 6))
sns.lineplot(data=spotify_data)
plt.title("Gráfico de Líneas con Estilo 'ticks'")
plt.show()

# 5. Restablecer el estilo predeterminado
sns.set_style("white") # Para restablecer el estilo original de seaborn.

# ¡Perfecto! Vamos a explorar cómo elegir los tipos de gráficos adecuados y cómo personalizar los estilos para hacer que tus visualizaciones sean más efectivas y atractivas.

# Resumen de Tipos de Gráficos:

# Como mencionaste, los tipos de gráficos se pueden clasificar en tres categorías principales:

# Tendencias:

# sns.lineplot: Ideal para mostrar tendencias a lo largo del tiempo o para comparar tendencias entre grupos.
# Relaciones:

# ns.barplot: Útil para comparar cantidades entre diferentes grupos.
# sns.heatmap: Permite visualizar patrones de color en tablas de datos numéricos.
# sns.scatterplot: Muestra la relación entre dos variables continuas.
# sns.regplot: Similar a scatterplot pero con una línea de regresión para resaltar la relación lineal.
# sns.lmplot: Permite dibujar múltiples líneas de regresión para diferentes grupos en un scatterplot.
# sns.swarmplot: Muestra la relación entre una variable continua y una variable categórica.
# Distribuciones:

# sns.histplot: Muestra la distribución de una variable numérica.
# sns.kdeplot: Muestra una estimación suavizada de la distribución de una variable numérica (o dos en el caso de gráficos KDE 2D).
# sns.jointplot: Combina un gráfico KDE 2D con gráficos KDE para las variables individuales.
# Personalización de Estilos con Seaborn:

# Seaborn ofrece cinco temas predefinidos para personalizar la apariencia de tus gráficos:

# "darkgrid"
# "whitegrid"
# "dark"
# "white"
# "ticks"
# Puedes cambiar el estilo de tus gráficos utilizando la función sns.set_style() y especificando el tema deseado.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ruta del archivo a leer (modificada)
fifa_filepath = "~/Downloads/fifa.csv"

# Lee el archivo en una variable fifa_data
fifa_data = pd.read_csv(fifa_filepath, index_col="Date", parse_dates=True)

# Imprime las primeras 5 filas de los datos
print(fifa_data.head().to_markdown(index=False, numalign="left", stralign="left"))

# Establece el ancho y la altura de la figura
plt.figure(figsize=(16, 6))

# Gráfico de líneas que muestra cómo evolucionaron las clasificaciones de la FIFA a lo largo del tiempo
sns.lineplot(data=fifa_data) # "lineplot" se refiere a un tipo de gráfico llamado gráfico de líneas
plt.show()
# Salidas: (.to_markdown)
# (env) ┌──(env)─(pol㉿kali)-[~/Desktop/data-science-b75]
# └─$ /home/pol/Desktop/data-science-b75/env/bin/python /home/pol/Desktop/data-science-b75/4a_seaborn.py
# | ARG   | BRA   | ESP   | FRA   | GER   | ITA   |
# |:------|:------|:------|:------|:------|:------|
# | 5     | 8     | 13    | 12    | 1     | 2     |
# | 12    | 1     | 14    | 7     | 5     | 2     |
# | 9     | 1     | 7     | 14    | 4     | 3     |
# | 9     | 4     | 7     | 15    | 3     | 1     |
# | 8     | 3     | 5     | 15    | 1     | 2     |

#Descripción de la imagen:

# La imagen muestra una parte de un gráfico de líneas. El gráfico representa 
# la evolución de las clasificaciones de la FIFA a lo largo del tiempo.

# El eje vertical (eje Y) representa la clasificación de la FIFA, con 
# valores que van de 0 a 25.
# El eje horizontal (eje X) representa la fecha, con marcas que muestran 
# los años 1996, 2000, 2004, 2008, 2012 y 2016.
# Se puede ver una línea específica etiquetada como "ITA", que representa 
# la clasificación de Italia a lo largo del tiempo.
# la imagen muestra que la clasificacion de ITA varia en el tiempo.
# la imagen es una porcion de una grafica mas grande, donde se pueden ver mas paises.
# En resumen, la imagen es una porción de un gráfico que visualiza la 
# clasificación de la FIFA de la selección italiana de fútbol a lo largo del tiempo.
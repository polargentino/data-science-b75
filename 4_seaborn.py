"""
Explicación del código:

1-Importar bibliotecas:

*-pandas para la manipulación de datos.
*-matplotlib.pyplot para la creación de gráficos.
*-seaborn para la visualización de datos estadísticos.

2-Cargar datos:

*-Se especifica la ruta del archivo melbourne_data.csv.
*-Se utiliza pd.read_csv() para leer el archivo en un DataFrame llamado 
melbourne_data.

3-Mostrar datos:

*-Se utiliza melbourne_data.head() para mostrar las primeras 5 filas del DataFrame.
*-.to_markdown(index=False, numalign="left", stralign="left") formatea la salida 
para que sea legible en Markdown.

4-Seleccionar columnas:

*-Se crea una lista columns_to_plot con los nombres de las columnas que se 
utilizarán en el gráfico de líneas.

5-Crear gráfico:

*-Se utiliza plt.figure(figsize=(16, 6)) para establecer el tamaño de la 
figura del gráfico.
*-Se utiliza sns.lineplot() para crear un gráfico de líneas con los datos 
de las columnas seleccionadas.
*-plt.show() muestra el gráfico.

6-Resumen:

Este código carga el conjunto de datos melbourne_data.csv, muestra las 
primeras 5 filas y crea un gráfico de líneas utilizando las columnas 
'Rooms', 'Distance', 'Landsize', 'BuildingArea' y 'YearBuilt'.
"""

import tabulate
print(tabulate.__version__)
# tabulate: Es la biblioteca que pandas necesita para formatear las tablas en Markdown.


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ruta del archivo a leer
melbourne_data_filepath = "/home/pol/Downloads/melb_data.csv"

# Lee el archivo en una variable melbourne_data
melbourne_data = pd.read_csv(melbourne_data_filepath)

# Imprime las primeras 5 filas de los datos
print(melbourne_data.head().to_markdown(index=False, numalign="left", stralign="left"))

# Selecciona las columnas para el gráfico de líneas
columns_to_plot = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']

# Establece el ancho y la altura de la figura
plt.figure(figsize=(16, 6))

# Gráfico de líneas que muestra cómo evolucionaron las variables seleccionadas a lo largo del tiempo
sns.lineplot(data=melbourne_data[columns_to_plot])
plt.show()
# Salidas:
# 0.9.0
# | Suburb     | Address          | Rooms   | Type   | Price     | Method   | SellerG   | Date      | Distance   | Postcode   | Bedroom2   | Bathroom   | Car   | Landsize   | BuildingArea   | YearBuilt   | CouncilArea   | Lattitude   | Longtitude   | Regionname            | Propertycount   |
# |:-----------|:-----------------|:--------|:-------|:----------|:---------|:----------|:----------|:-----------|:-----------|:-----------|:-----------|:------|:-----------|:---------------|:------------|:--------------|:------------|:-------------|:----------------------|:----------------|
# | Abbotsford | 85 Turner St     | 2       | h      | 1.48e+06  | S        | Biggin    | 3/12/2016 | 2.5        | 3067       | 2          | 1          | 1     | 202        | nan            | nan         | Yarra         | -37.7996    | 144.998      | Northern Metropolitan | 4019            |
# | Abbotsford | 25 Bloomburg St  | 2       | h      | 1.035e+06 | S        | Biggin    | 4/02/2016 | 2.5        | 3067       | 2          | 1          | 0     | 156        | 79             | 1900        | Yarra         | -37.8079    | 144.993      | Northern Metropolitan | 4019            |
# | Abbotsford | 5 Charles St     | 3       | h      | 1.465e+06 | SP       | Biggin    | 4/03/2017 | 2.5        | 3067       | 3          | 2          | 0     | 134        | 150            | 1900        | Yarra         | -37.8093    | 144.994      | Northern Metropolitan | 4019            |
# | Abbotsford | 40 Federation La | 3       | h      | 850000    | PI       | Biggin    | 4/03/2017 | 2.5        | 3067       | 3          | 2          | 1     | 94         | nan            | nan         | Yarra         | -37.7969    | 144.997      | Northern Metropolitan | 4019            |
# | Abbotsford | 55a Park St      | 4       | h      | 1.6e+06   | VB       | Nelson    | 4/06/2016 | 2.5        | 3067       | 3          | 1          | 2     | 120        | 142            | 2014        | Yarra         | -37.8072    | 144.994      | Northern Metropolitan | 4019            |


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
import os
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.animation import FuncAnimation

sns.set_style("whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

def apply_kmeans_clustering(data_path):
    df = pd.read_csv(data_path)
    X = df.loc[:, ["MedInc", "Latitude", "Longitude"]].values  # Convertir a array NumPy

    # Animación K-medias
    def animate(i):
        kmeans = KMeans(n_clusters=6, n_init=1, max_iter=i, random_state=0)  # max_iter para animación
        kmeans.fit(X[:, 1:3])  # Solo Latitud y Longitud para animación
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        plt.clf()
        plt.scatter(X[:, 1], X[:, 2], c=labels, cmap='viridis', alpha=0.5)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200)
        plt.title(f"Iteración {i}")

    fig = plt.figure()
    ani = FuncAnimation(fig, animate, frames=range(1, 21), repeat=True)  # Animar 20 iteraciones
    plt.show()

    kmeans = KMeans(n_clusters=6, n_init='auto', random_state=0)
    labels = kmeans.fit_predict(X[:, 1:3])  # Solo Latitud y Longitud para Voronoi

    # Tessellación Voronoi
    vor = Voronoi(kmeans.cluster_centers_)
    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_width=2)
    plt.scatter(X[:, 1], X[:, 2], c=labels, cmap='viridis', alpha=0.5)
    plt.title("Tessellación de Voronoi")
    plt.show()

    X_df = pd.DataFrame(X, columns=["MedInc", "Latitude", "Longitude"])
    X_df["Cluster"] = labels
    X_df["Cluster"] = X_df["Cluster"].astype("category")
    X_df["MedHouseVal"] = df["MedHouseVal"]

    print("Primeros registros del DataFrame transformado:")
    print(X_df.head())

if __name__ == "__main__":
    downloads_path = os.path.expanduser("~/Downloads")
    data_path = os.path.join(downloads_path, "housing.csv")
    apply_kmeans_clustering(data_path)

# Salidas: Primeros registros del DataFrame transformado:
#    MedInc  Latitude  Longitude Cluster  MedHouseVal
# 0  8.3252     37.88    -122.23       1        4.526
# 1  8.3014     37.86    -122.22       1        3.585
# 2  7.2574     37.85    -122.24       1        3.521
# 3  5.6431     37.85    -122.25       1        3.413
# 4  3.8462     37.85    -122.25       1        3.422

# Análisis por Gemini 2.0 Flash: (características_7.png)
# Claro, vamos a realizar un análisis detallado de la imagen que has obtenido:

# Título: Tessellación de Voronoi

# El título nos indica que estamos viendo una representación de una 
# tessellación de Voronoi. Esta técnica se utiliza para dividir un plano 
# en regiones basadas en la distancia a un conjunto de puntos (en este caso, 
# los centroides de los clústeres generados por K-medias).

# Ejes:

# Eje X (Horizontal): Representa la longitud (Longitude), variando aproximadamente 
# de 34 a 40.
# Eje Y (Vertical): Representa la latitud (Latitude), variando aproximadamente 
# de -123 a -118.
# Puntos de Datos:

# Los puntos dispersos en el gráfico representan los datos originales.
# Cada punto está coloreado según el clúster al que ha sido asignado por el 
# algoritmo K-medias.
# La transparencia de los puntos (alpha) permite ver la densidad de los datos 
# en diferentes áreas.
# Regiones de Voronoi:

# Las líneas sólidas negras y las líneas discontinuas negras delimitan las 
# regiones de Voronoi.
# Cada región de Voronoi contiene todos los puntos que están más cerca de un 
# centroide particular que de cualquier otro centroide.
# Las líneas representan los límites donde la distancia a dos centroides es igual.
# Análisis Detallado:

# Agrupamiento Espacial:

# La imagen muestra cómo el algoritmo K-medias ha agrupado los datos espacialmente.
# Las regiones de Voronoi delimitan claramente los clústeres geográficos.
# Se pueden observar áreas de alta densidad de puntos dentro de cada región, 
# lo que indica una fuerte cohesión dentro de los clústeres.
# Distribución de Clústeres:

# Los clústeres están distribuidos de manera no uniforme a lo largo del plano.
# Algunos clústeres son más grandes y abarcan una mayor área geográfica, 
# mientras que otros son más pequeños y concentrados.
# Se puede observar que hay una mayor densidad de datos en la parte inferior 
# izquierda del grafico.
# Límites de Voronoi:

# Los límites de Voronoi proporcionan una visualización clara de cómo se 
# dividen los datos en regiones basadas en la proximidad a los centroides.
# Las líneas sólidas indican límites más definidos, mientras que las líneas 
# discontinuas pueden indicar límites menos precisos o áreas de transición.
# Interpretación Geográfica:

# Dado que los ejes representan latitud y longitud, podemos interpretar la 
# imagen como una representación geográfica de los clústeres.
# Los clústeres pueden representar diferentes regiones geográficas con características 
# similares, como densidad de población, ingresos medios o valores de las viviendas.
# Posibles Aplicaciones:

# Esta visualización puede ser útil para identificar patrones geográficos en los datos.
# Puede ayudar a comprender cómo se agrupan los datos espacialmente y a identificar 
# áreas de interés.
# Puede ser útil para aplicaciones como planificación urbana, análisis de mercado 
# o segmentación de clientes.
# En resumen:

# La imagen proporciona una visualización clara y efectiva de la tessellación 
# de Voronoi generada por el algoritmo K-medias. Muestra cómo los datos se 
# agrupan espacialmente y cómo se delimitan las regiones de clústeres. Esta 
# visualización puede ser útil para identificar patrones geográficos y comprender 
# la distribución de los datos en el espacio.
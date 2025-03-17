import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
import os

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
    """
    Aplica el agrupamiento K-medias a los datos de vivienda de California y visualiza 
    los resultados.

    Args:
        data_path (str): Ruta al archivo CSV que contiene los datos de vivienda.

    Returns:
        None. Muestra gráficos y imprime los primeros registros del DataFrame transformado.
    """

    # 1. Cargar los datos
    df = pd.read_csv(data_path)

    # 2. Seleccionar las características para el agrupamiento
    X = df.loc[:, ["MedInc", "Latitude", "Longitude"]]

    # 3. Inicializar y entrenar el modelo K-medias
    kmeans = KMeans(n_clusters=6, n_init='auto', random_state=0)
    X["Cluster"] = kmeans.fit_predict(X)
    X["Cluster"] = X["Cluster"].astype("category")

    # 4. Visualizar la distribución geográfica de los clústeres
    plt.figure()
    sns.relplot(
        x="Longitude", y="Latitude", hue="Cluster", data=X, height=6
    )
    plt.title("Distribución Geográfica de Clústeres")
    plt.show()

    # 5. Agregar la variable objetivo 'MedHouseVal' al DataFrame
    X["MedHouseVal"] = df["MedHouseVal"]

    # 6. Visualizar la distribución de 'MedHouseVal' dentro de cada clúster
    plt.figure()
    sns.catplot(x="MedHouseVal", y="Cluster", data=X, kind="boxen", height=6)
    plt.title("Distribución de MedHouseVal por Clúster")
    plt.show()

    # 7. Imprimir los primeros registros del DataFrame transformado
    print("Primeros registros del DataFrame transformado:")
    print(X.head())

if __name__ == "__main__":
    # Obtener la ruta del directorio de descargas
    downloads_path = os.path.expanduser("~/Downloads")
    data_path = os.path.join(downloads_path, "housing.csv")

    apply_kmeans_clustering(data_path)

# Salidas: (env) ┌──(env)─(pol㉿kali)-[~/Desktop/data-science-b75]
# └─$ /home/pol/Desktop/data-science-b75/env/bin/python /home/pol/Desktop/data-science-b75/6d_caracte.py
# Primeros registros del DataFrame transformado:
#    MedInc  Latitude  Longitude Cluster  MedHouseVal
# 0  8.3252     37.88    -122.23       4        4.526
# 1  8.3014     37.86    -122.22       4        3.585
# 2  7.2574     37.85    -122.24       4        3.521
# 3  5.6431     37.85    -122.25       1        3.413
# 4  3.8462     37.85    -122.25       1        3.422
                                                     
# Analisis por Gemini 2.0 Flash:
# ¡Excelente! El script se ha ejecutado correctamente y ha producido la salida esperada. Veamos qué significa esta salida:


# Interpretación:

# MedInc: Ingreso medio en el bloque de viviendas.
# Latitude: Latitud del bloque de viviendas.
# Longitude: Longitud del bloque de viviendas.
# Cluster: Etiqueta del clúster asignada por el algoritmo K-medias. En este caso, 
# los clústeres están etiquetados del 0 al 5 (ya que configuraste n_clusters=6).
# MedHouseVal: Valor medio de la vivienda en el bloque de viviendas.
# Análisis:
# ---------
# El algoritmo K-medias ha agrupado los bloques de viviendas en 6 clústeres diferentes, 
# basándose en las características de MedInc, Latitude y Longitude.
# Los primeros tres bloques de viviendas (índices 0, 1 y 2) han sido asignados al 
# clúster 4. Estos bloques tienen ingresos medios relativamente altos y están 
# ubicados en una región geográfica similar.
# Los bloques de viviendas en los índices 3 y 4 han sido asignados al clúster 1. 
# Estos bloques tienen ingresos medios más bajos en comparación con los bloques en 
# el clúster 4 y también están ubicados en una región geográfica diferente.
# La visualización que se genera al correr el script, muestra la distribución de 
# los clusteres a lo largo del mapa, y tambien la distribución de los precios de 
# las casas a lo largo de los clusteres.
# Esto demuestra cómo K-medias puede identificar grupos de bloques de viviendas 
# con características similares, lo cual puede ser útil para diversas aplicaciones, 
# como análisis de mercado, planificación urbana o evaluación de riesgos.
# Próximos Pasos:
# --------------
# Puedes explorar diferentes valores para n_clusters para ver cómo cambia el agrupamiento.
# Puedes visualizar los clústeres en un mapa para obtener una mejor comprensión 
# de su distribución geográfica.
# Puedes analizar la distribución de otras variables dentro de cada clúster para 
# identificar patrones y relaciones.
# Recuerda que para una mejor visualización de los datos, los graficos generados 
# por el script, son muy importantes.                                                    
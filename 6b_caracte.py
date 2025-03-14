import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import os


def analizar_datos_autos(csv_file):
    """
    Analiza un conjunto de datos de autos utilizando técnicas de preprocesamiento, 
    selección de características y visualización.

    Este script realiza las siguientes operaciones:
    -----------------------------------------------
    1. Carga un conjunto de datos desde un archivo CSV especificado.
    2. Realiza un preprocesamiento básico de datos, incluyendo la conversión de 
    variables categóricas a numéricas mediante label encoding.
    3. Calcula las puntuaciones de información mutua entre las características y 
    la variable objetivo (precio).
    4. Visualiza las puntuaciones de información mutua para identificar las 
    características más relevantes.
    5. Realiza visualizaciones de datos para explorar relaciones entre 
    variables específicas.

    Args:
        csv_file (str): La ruta al archivo CSV que contiene los datos de los autos.

    Returns:
        None: Los resultados del análisis se imprimen y visualizan en la consola.

    Raises:
        FileNotFoundError: Si el archivo CSV especificado no se encuentra.
        Exception: Si ocurre algún error durante el procesamiento de los 
        datos o la visualización.

    Ejemplo:
        Para ejecutar el script, asegúrate de tener instalado pandas, 
        numpy, seaborn y scikit-learn.
        Luego, llama a la función con la ruta al archivo CSV:

        >>> analizar_datos_autos("/home/pol/Downloads/autos.csv")
    """
    try:
        # 1. Cargar el conjunto de datos
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Error: El archivo no se encontró en la ruta: {csv_file}")
        
        df = pd.read_csv(csv_file)

        # 2. Preprocesamiento de datos
        X = df.copy()
        y = X.pop("price")

        # Label encoding para variables categóricas
        # - Este proceso convierte las variables categóricas en numéricas asignando un entero a cada categoría
        for colname in X.select_dtypes("object"):
            X[colname], _ = X[colname].factorize()

        # Todas las características discretas ahora deberían tener dtypes enteros
        discrete_features = X.dtypes == int

        # 3. Función para calcular las puntuaciones de información mutua
        def make_mi_scores(X, y, discrete_features):
            """
            Calcula las puntuaciones de información mutua entre las características y 
            la variable objetivo.

            Args:
                X (DataFrame): Conjunto de características.
                y (Series): Variable objetivo.
                discrete_features (Series): Indicadores de características discretas.

            Returns:
                Series: Puntuaciones de información mutua ordenadas en forma descendente.
            """
            mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
            mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
            mi_scores = mi_scores.sort_values(ascending=False)
            return mi_scores

        # Calcular las puntuaciones de información mutua
        mi_scores = make_mi_scores(X, y, discrete_features)

        # Mostrar algunas características con sus puntuaciones de MI
        print(mi_scores[::3])  # Muestra cada tercera característica

        # 4. Función para graficar las puntuaciones de información mutua
        def plot_mi_scores(scores):
            """
            Grafica las puntuaciones de información mutua en un gráfico de 
            barras horizontales.

            Args:
                scores (Series): Puntuaciones de información mutua.
            """
            scores = scores.sort_values(ascending=True)
            width = np.arange(len(scores))
            ticks = list(scores.index)
            plt.barh(width, scores)
            plt.yticks(width, ticks)
            plt.title("Mutual Information Scores")

        # Graficar las puntuaciones de información mutua
        plt.figure(dpi=100, figsize=(8, 5))
        plot_mi_scores(mi_scores)
        plt.show()

        # 5. Visualización de datos: Relación entre curb_weight y price
        # - Utiliza seaborn para crear un gráfico de dispersión que muestra la 
        # relación entre el peso del vehículo y su precio
        sns.relplot(x="curb_weight", y="price", data=df)
        plt.show()

        # Visualización de datos: Relación entre horsepower y price, diferenciado por 
        # fuel_type
        # - Utiliza seaborn para crear un gráfico de regresión lineal que muestra 
        # la relación entre la potencia del motor y el precio, diferenciado por 
        # tipo de combustible
        sns.lmplot(x="horsepower", y="price", hue="fuel_type", data=df)
        plt.show()

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Ocurrió un error: {e}")

# Ejemplo de uso:
csv_file = "/home/pol/Downloads/autos.csv"
analizar_datos_autos(csv_file)

# Salidas: 
# curb_weight          1.556242
# highway_mpg          0.944594
# length               0.617436
# fuel_system          0.484140
# stroke               0.389168
# drive_wheels         0.332973
# compression_ratio    0.133702
# fuel_type            0.047298
# Name: MI Scores, dtype: float64

# Análisis de gráfico: por Perplexity: (características_1.png)
# -----------------------------------
# Respuesta:
# -----------
# La imagen muestra un gráfico de barras horizontales que representa 
# las puntuaciones de información mutua (MI Scores) entre las 
# características del conjunto de datos y la variable objetivo, que 
# en este caso es el precio de los autos. La información mutua mide la 
# dependencia estadística entre dos variables, indicando qué tan 
# informativa es una característica para predecir la variable objetivo.

# Análisis del Gráfico
# Características Más Relevantes:
# -------------------------------
# curb_weight (peso del vehículo) tiene la puntuación más alta, 
# lo que indica que es la característica más informativa para predecir el precio.

# horsepower (potencia del motor) y engine_size (tamaño del motor) 
# también tienen puntuaciones altas, lo que sugiere que son factores 
# clave en la determinación del precio de un auto.

# Características Moderadamente Informativas:
# -------------------------------------------
# Variables como highway_mpg (rendimiento en carretera), city_mpg 
# (rendimiento en ciudad), width (ancho del vehículo), y length (longitud) 
# tienen puntuaciones intermedias. Esto indica que estas características 
# tienen cierta influencia en el precio, pero no son tan determinantes 
# como las principales.

# Características Menos Informativas:
# -----------------------------------
# Al final de la lista se encuentran características como fuel_type 
# (tipo de combustible), engine_location (ubicación del motor), 
# y num_of_doors (número de puertas). Estas tienen puntuaciones bajas, 
# lo que sugiere que tienen poca o ninguna relevancia para predecir el precio.

# Interpretación General:
# -----------------------
# Las características relacionadas con el rendimiento, tamaño y peso del 
# vehículo tienden a ser más relevantes para determinar su precio. Esto es 
# coherente con el hecho de que estos factores suelen influir directamente 
# en los costos de fabricación y percepción de calidad.

# Uso Práctico:
# -------------
# Este análisis puede ser utilizado para reducir la dimensionalidad del 
# conjunto de datos, seleccionando solo las características más relevantes 
# (curb_weight, horsepower, engine_size, etc.) para entrenar modelos 
# predictivos, lo cual podría mejorar el rendimiento y eficiencia del modelo.

# Conclusión:
# -----------
# El gráfico proporciona una visión clara sobre cuáles características 
# deben priorizarse al construir un modelo predictivo para estimar el 
# precio de los autos. Las características con puntuaciones más altas 
# tienen una mayor capacidad explicativa y deberían ser consideradas como 
# esenciales en el análisis.

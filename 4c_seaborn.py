import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analizar_datos_reproducciones(csv_file):
    """
    Analiza un conjunto de datos de reproducciones diarias de canciones.

    Este script realiza las siguientes operaciones:
    1. Carga un conjunto de datos desde un archivo CSV especificado.
    2. Realiza un gráfico de líneas para mostrar la evolución de las reproducciones de cada canción a lo largo del tiempo.

    Args:
        csv_file (str): La ruta al archivo CSV que contiene los datos de reproducciones.

    Returns:
        None: Los resultados del análisis se visualizan en la consola.

    Raises:
        FileNotFoundError: Si el archivo CSV especificado no se encuentra.
        Exception: Si ocurre algún error durante el procesamiento de los datos o la visualización.

    Ejemplo:
        Para ejecutar el script, asegúrate de tener instalado pandas y seaborn.
        Luego, llama a la función con la ruta al archivo CSV:

        >>> analizar_datos_reproducciones("/home/usuario/Downloads/reproducciones.csv")
    """
    try:
        # 1. Carga del conjunto de datos
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Error: El archivo no se encontró en la ruta: {csv_file}")
        
        reproducciones_data = pd.read_csv(csv_file)

        # 2. Inspección de los datos (opcional)
        print("Primeras filas del DataFrame:")
        print(reproducciones_data.head().to_markdown(index=False, numalign="left", stralign="left"))

        # 3. Gráfico de líneas: Evolución de las reproducciones de cada canción
        plt.figure(figsize=(12, 6))
        for columna in reproducciones_data.columns[1:]:  # Omite la columna 'Date'
            plt.plot(reproducciones_data['Date'], reproducciones_data[columna], label=columna)
        plt.title('Evolución de las Reproducciones de Cada Canción')
        plt.xlabel('Fecha')
        plt.ylabel('Número de Reproducciones')
        plt.legend()
        plt.show()

    except FileNotFoundError as e:
        print(f"Error al cargar el archivo: {e}")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

# Ejemplo de uso:
csv_file = "/home/pol/Downloads/spotify.csv"  # Asegúrate de que el archivo tenga la extensión correcta (.csv)
analizar_datos_reproducciones(csv_file)

# Primeras filas del DataFrame:
# | Date       | Shape of You   | Despacito   | Something Just Like This   | HUMBLE.   | Unforgettable   |
# |:-----------|:---------------|:------------|:---------------------------|:----------|:----------------|
# | 2017-01-06 | 12287078       | nan         | nan                        | nan       | nan             |
# | 2017-01-07 | 13190270       | nan         | nan                        | nan       | nan             |
# | 2017-01-08 | 13099919       | nan         | nan                        | nan       | nan             |
# | 2017-01-09 | 14506351       | nan         | nan                        | nan       | nan             |
# | 2017-01-10 | 14275628       | nan         | nan                        | nan       | nan             |
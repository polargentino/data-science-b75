import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import os  # Importa la librería os para manejar rutas

def analizar_resistencia_concreto(csv_file):
    """
    Analiza la resistencia a la compresión del concreto utilizando Random Forest Regression.

    Este script realiza las siguientes operaciones:
    1. Carga un conjunto de datos desde un archivo CSV especificado.
    2. Realiza una limpieza básica de datos (omitiendo valores faltantes).
    3. Entrena un modelo base de Random Forest Regression.
    4. Realiza ingeniería de características creando nuevas características basadas en proporciones.
    5. Entrena un modelo mejorado con las nuevas características.
    6. Evalúa ambos modelos utilizando validación cruzada y muestra las puntuaciones MAE (Error Absoluto Medio).

    Args:
        csv_file (str): La ruta al archivo CSV que contiene los datos del concreto.

    Returns:
        None: Los resultados del análisis se imprimen en la consola.

    Raises:
        FileNotFoundError: Si el archivo CSV especificado no se encuentra.
        Exception: Si ocurre algún error durante el procesamiento de los datos o el entrenamiento del modelo.

    Ejemplo:
        Para ejecutar el script, asegúrate de tener instalado pandas y scikit-learn.
        Luego, llama a la función con la ruta al archivo CSV:

        >>> analizar_resistencia_concreto("/home/pol/Downloads/concrete.csv")
    """
    try:
        # 1. Verifica si el archivo existe
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Error: El archivo no se encontró en la ruta: {csv_file}")

        # 2. Carga los datos
        df = pd.read_csv(csv_file)

        # 3. Muestra las primeras filas (opcional, pero útil)
        print("Primeras 5 filas del DataFrame:")
        print(df.head().to_markdown(index=False, numalign="left", stralign="left"))

        # 4. Línea base
        X = df.copy()
        y = X.pop("CompressiveStrength")

        # Train and score baseline model
        baseline = RandomForestRegressor(criterion="absolute_error", random_state=0)
        baseline_score = cross_val_score(
            baseline, X, y, cv=5, scoring="neg_mean_absolute_error")
        baseline_score = -1 * baseline_score.mean()
        print(f"\nMAE Baseline Score: {baseline_score:.4}")

        # 5. Ingeniería de características
        X = df.copy()
        y = X.pop("CompressiveStrength")

        # Create synthetic features
        X["FCRatio"] = X["FineAggregate"] / X["CoarseAggregate"]
        X["AggCmtRatio"] = (X["CoarseAggregate"] + X["FineAggregate"]) / X["Cement"]
        X["WtrCmtRatio"] = X["Water"] / X["Cement"]

        # 6. Modelo con características de proporción
        # Train and score model on dataset with additional ratio features
        model = RandomForestRegressor(criterion="absolute_error", random_state=0)
        score = cross_val_score(
            model, X, y, cv=5, scoring="neg_mean_absolute_error")
        score = -1 * score.mean()
        print(f"\nMAE Score with Ratio Features: {score:.4}")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Ocurrió un error: {e}")

# Ejemplo de uso:
csv_file = "/home/pol/Downloads/concrete.csv"
analizar_resistencia_concreto(csv_file)


# Salidas: 
# Primeras 5 filas del DataFrame:
# | Cement   | BlastFurnaceSlag   | FlyAsh   | Water   | Superplasticizer   | CoarseAggregate   | FineAggregate   | Age   | CompressiveStrength   |
# |:---------|:-------------------|:---------|:--------|:-------------------|:------------------|:----------------|:------|:----------------------|
# | 540      | 0                  | 0        | 162     | 2.5                | 1040              | 676             | 28    | 79.99                 |
# | 540      | 0                  | 0        | 162     | 2.5                | 1055              | 676             | 28    | 61.89                 |
# | 332.5    | 142.5              | 0        | 228     | 0                  | 932               | 594             | 270   | 40.27                 |
# | 332.5    | 142.5              | 0        | 228     | 0                  | 932               | 594             | 365   | 41.05                 |
# | 198.6    | 132.4              | 0        | 192     | 0                  | 978.4             | 825.5           | 360   | 44.3                  |

# MAE Baseline Score: 8.232
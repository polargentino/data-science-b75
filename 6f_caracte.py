"""
Explicación Detallada:

Importaciones: Se importan las bibliotecas necesarias.

Configuración de Estilo: Se configura el estilo de los gráficos con seaborn-whitegrid.

plot_variance: Función para visualizar la varianza explicada por los componentes principales.

make_mi_scores: Función para calcular las puntuaciones de información mutua.

Carga de Datos: Se carga el archivo CSV autos.csv.

Selección de Características: Se seleccionan las características relevantes.

Estandarización: Se estandarizan las características numéricas.

PCA: Se aplica PCA para obtener los componentes principales.

DataFrame de Componentes: Se crea un DataFrame con los componentes principales.

Visualización de Componentes: Se muestran los primeros registros del DataFrame de componentes.

DataFrame de Cargas: Se crea un DataFrame con las cargas de PCA.

Visualización de Varianza: Se visualiza la varianza explicada por los componentes.

Puntuaciones MI: Se calculan las puntuaciones de información mutua.

Ordenamiento por PC3: Se ordena el DataFrame original por PC3.

Nueva Característica: Se crea la característica sports_or_wagon.

Visualización de Relación: Se visualiza la relación entre la nueva característica 
y el precio.
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# from IPython.display import display
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA

# Configuración de estilo para los gráficos
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

def plot_variance(pca, width=8, dpi=100):
    """
    Visualiza la varianza explicada por los componentes principales.

    Args:
        pca (PCA): Objeto PCA entrenado.
        width (int): Ancho de la figura.
        dpi (int): Puntos por pulgada de la figura.

    Returns:
        axs (tuple): Tupla de objetos de ejes de matplotlib.
    """
    # Crea una figura con dos subgráficos
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)

    # Varianza explicada
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )

    # Varianza acumulada
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )

    # Configura la figura
    fig.set(figwidth=width, dpi=dpi)
    return axs

def make_mi_scores(X, y, discrete_features):
    """
    Calcula las puntuaciones de información mutua entre las características y la variable objetivo.

    Args:
        X (DataFrame): DataFrame de características.
        y (Series): Serie de la variable objetivo.
        discrete_features (bool or array-like): Indica si las características son discretas.

    Returns:
        mi_scores (Series): Serie de puntuaciones de información mutua, ordenada de forma descendente.
    """
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

# 1. Carga de datos
df = pd.read_csv("/home/pol/Downloads/autos.csv")

# 2. Selección de características
features = ["highway_mpg", "engine_size", "horsepower", "curb_weight"]
X = df.copy()
y = X.pop('price')
X = X.loc[:, features]

# 3. Estandarización de características
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

# 4. Aplicación de PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 5. Creación de DataFrame de componentes principales
component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=component_names)

# 6. Visualización de los primeros registros de X_pca
print("Primeros registros de X_pca:")
print(X_pca.head())

# 7. Creación de DataFrame de cargas (loadings)
loadings = pd.DataFrame(
    pca.components_.T,  # Transponer la matriz de cargas
    columns=component_names,  # Nombres de las columnas son los componentes principales
    index=X.columns,  # Nombres de las filas son las características originales
)
print("\nCargas (loadings):")
print(loadings)

# 8. Visualización de la varianza explicada
print("\nVisualización de la varianza explicada:")
plot_variance(pca)
plt.show()

# 9. Cálculo de puntuaciones de información mutua
mi_scores = make_mi_scores(X_pca, y, discrete_features=False)
print("\nPuntuaciones de información mutua:")
print(mi_scores)

# 10. Ordenar DataFrame por PC3 y mostrar columnas seleccionadas
idx = X_pca["PC3"].sort_values(ascending=False).index
cols = ["make", "body_style", "horsepower", "curb_weight"]
print("\nDataFrame ordenado por PC3:")
print(df.loc[idx, cols])

# 11. Creación de nueva característica "sports_or_wagon"
df["sports_or_wagon"] = X.curb_weight / X.horsepower

# 12. Visualización de la relación entre "sports_or_wagon" y "price"
print("\nVisualización de la relación entre 'sports_or_wagon' y 'price':")
sns.regplot(x="sports_or_wagon", y='price', data=df, order=2)
plt.show()

# Analisis por Gemini 2.0 Flash:
# ¡Perfecto! Vamos a analizar los resultados que has obtenido paso a paso:

# 1. Primeros Registros de X_pca:

#    PC1       PC2       PC3       PC4
# 0  0.382486 -0.400222  0.124122  0.169539
# 1  0.382486 -0.400222  0.124122  0.169539
# 2  1.550890 -0.107175  0.598361 -0.256081
# 3 -0.408859 -0.425947  0.243335  0.013920
# 4  1.132749 -0.814565 -0.202885  0.224138
# Esta tabla muestra los primeros cinco registros de los datos transformados a los componentes principales (PC1, PC2, PC3, PC4).
# Cada columna representa un componente principal, que es una combinación lineal de las características originales.
# Los valores representan la posición de cada registro en el espacio de los componentes principales.
# 2. Cargas (Loadings):

#              PC1       PC2       PC3       PC4
# highway_mpg -0.492347  0.770892  0.070142 -0.397996
# engine_size   0.503859  0.626709  0.019960  0.594107
# horsepower    0.500448  0.013788  0.731093 -0.463534
# curb_weight   0.503262  0.113008 -0.678369 -0.523232
# Esta tabla muestra las cargas (loadings) de cada característica original en cada componente principal.
# Las cargas indican cuánto contribuye cada característica a la variación capturada por cada componente.
# Interpretación:
# PC1: Contraste entre vehículos con alto consumo de combustible (highway_mpg negativo) y vehículos con motores grandes, alta potencia y peso elevado (engine_size, horsepower, curb_weight positivos). Representa el "Eje de Lujo/Economía".
# PC2: Principalmente relacionado con highway_mpg y engine_size.
# PC3: Contraste entre horsepower (positivo) y curb_weight (negativo). Representa el contraste entre deportivos y familiares.
# PC4: Contraste entre engine_size (positivo) y highway_mpg, horsepower, curb_weight (negativos).
# 3. Visualización de la Varianza Explicada:

# El gráfico (que no se muestra aquí, pero se generó) muestra la proporción de varianza explicada por cada componente principal y la varianza acumulada.
# Resultado Esperado: PC1 explica la mayor parte de la varianza, seguido por PC2, PC3 y PC4.
# 4. Puntuaciones de Información Mutua (MI):

# PC1    1.014976
# PC2    0.378853
# PC3    0.307622
# PC4    0.203591
# Name: MI Scores, dtype: float64
# Estas puntuaciones muestran la relación entre cada componente principal y la variable objetivo "price".
# Interpretación:
# PC1 tiene la puntuación MI más alta, lo que indica que es el componente más predictivo del precio.
# Los componentes restantes también tienen una relación significativa con el precio, aunque en menor medida.
# 5. DataFrame Ordenado por PC3:

#              make    body_style  horsepower  curb_weight
# 118       porsche       hardtop         207         2756
# 117       porsche       hardtop         207         2756
# 119       porsche  convertible         207         2800
# 45        jaguar         sedan         262         3950
# 96        nissan     hatchback         200         3139
# ...           ...           ...         ...          ...
# 59   mercedes-benz       wagon         123         3750
# 61   mercedes-benz        sedan         123         3770
# 101        peugot       wagon          95         3430
# 105        peugot       wagon          95         3485
# 143        toyota       wagon          62         3110
# Esta tabla muestra los registros del DataFrame original ordenados por el componente PC3.
# Interpretación:
# Los registros con valores PC3 altos (parte superior de la tabla) corresponden a vehículos con alta potencia y bajo peso (deportivos).
# Los registros con valores PC3 bajos (parte inferior de la tabla) corresponden a vehículos con baja potencia y alto peso (familiares).
# 6. Visualización de la Relación entre 'sports_or_wagon' y 'price':

# El gráfico (que no se muestra aquí, pero se generó) muestra la relación entre la nueva característica "sports_or_wagon" (relación entre curb_weight y horsepower) y el precio.
# Resultado Esperado: Se espera una relación no lineal, posiblemente cuadrática, que refleje cómo la relación entre peso y potencia afecta el precio.
# Conclusiones Generales:

# PCA ha permitido identificar patrones de variación significativos en el conjunto de datos de automóviles.
# PC1, que representa el "Eje de Lujo/Economía", es el componente más importante para predecir el precio.
# PC3 revela el contraste entre deportivos y familiares.
# La creación de la nueva característica "sports_or_wagon" y su visualización ayudan a entender mejor la relación entre las características originales y el precio.

# Análisis del gráfico características_9.png:
# -------------------------------------------
# Claro, vamos a analizar el gráfico que has proporcionado:

# Título: No se proporciona un título en la imagen, pero podemos inferir que se trata de la relación entre la característica "sports_or_wagon" y el precio ("price").

# Ejes:

# Eje X (Horizontal): Representa la característica "sports_or_wagon". Esta característica, como se mencionó anteriormente, es la relación entre el peso en vacío ("curb_weight") y la potencia ("horsepower") de los automóviles. Los valores en el eje X varían aproximadamente de 15 a 50.
# Eje Y (Vertical): Representa el precio ("price") de los automóviles, medido en alguna unidad monetaria (asumimos dólares). Los valores en el eje Y varían aproximadamente de 5,000 a 45,000.
# Puntos de Datos:

# Los puntos dispersos en el gráfico representan los datos individuales de los automóviles.
# La posición de cada punto en el gráfico indica el valor de "sports_or_wagon" y el precio correspondiente para ese automóvil.
# Línea de Regresión y Área Sombreada:

# La línea azul que atraviesa los puntos es una línea de regresión polinómica de segundo grado (cuadrática). Esta línea representa la tendencia general de la relación entre "sports_or_wagon" y el precio.
# El área sombreada alrededor de la línea de regresión representa el intervalo de confianza. Indica la incertidumbre en la estimación de la línea de regresión.
# Análisis Detallado:

# Relación No Lineal:

# El gráfico muestra una clara relación no lineal entre "sports_or_wagon" y el precio.
# La línea de regresión cuadrática se ajusta bien a los datos, lo que sugiere que una relación cuadrática describe mejor la relación que una relación lineal.
# Tendencia:

# Para valores bajos de "sports_or_wagon" (aproximadamente 15-25), el precio tiende a disminuir a medida que aumenta "sports_or_wagon". Esto significa que los automóviles con una relación baja entre peso y potencia (más deportivos) tienden a tener precios más altos.
# Para valores medios de "sports_or_wagon" (aproximadamente 25-35), el precio alcanza su punto más bajo. Esto sugiere que hay un rango de valores para esta relación donde los automóviles son más económicos.
# Para valores altos de "sports_or_wagon" (aproximadamente 35-50), el precio tiende a aumentar a medida que aumenta "sports_or_wagon". Esto significa que los automóviles con una relación alta entre peso y potencia (más familiares) también pueden tener precios más altos.
# Dispersión de Datos:

# Los puntos de datos muestran una dispersión considerable alrededor de la línea de regresión. Esto indica que hay otros factores, además de "sports_or_wagon", que influyen en el precio de los automóviles.
# La dispersión es mayor en los extremos del rango de "sports_or_wagon", lo que sugiere que la relación es menos precisa en esos rangos.
# Interpretación de "sports_or_wagon":

# Como se mencionó anteriormente, "sports_or_wagon" es la relación entre el peso en vacío y la potencia.
# Valores bajos de "sports_or_wagon" indican automóviles con alta potencia en relación con su peso (deportivos).
# Valores altos de "sports_or_wagon" indican automóviles con bajo potencia en relación con su peso (familiares).
# Conclusiones:

# El gráfico muestra una relación cuadrática entre la relación peso/potencia ("sports_or_wagon") y el precio de los automóviles.
#Los automóviles deportivos y familiares tienden a tener precios más altos, mientras que los automóviles con una relación peso/potencia intermedia tienden a ser más económicos.
# La dispersión de los datos sugiere que otros factores influyen en el precio de los automóviles.



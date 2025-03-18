import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from category_encoders import MEstimateEncoder

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
warnings.filterwarnings('ignore')

def load_and_prepare_data(filepath):
    """
    Carga y prepara el conjunto de datos MovieLens1M.

    Args:
        filepath (str): Ruta al archivo CSV.

    Returns:
        DataFrame: DataFrame cargado y preparado.
    """
    df = pd.read_csv(filepath)
    df = df.astype(np.uint8, errors='ignore')  # Reducir el uso de memoria
    print("Número de Zipcodes únicos: {}".format(df["Zipcode"].nunique()))
    return df

def split_data(df):
    """
    Divide el DataFrame en conjuntos de entrenamiento y codificación.

    Args:
        df (DataFrame): DataFrame original.

    Returns:
        tuple: Tupla con X_encode, y_encode, X_pretrain, y_train.
    """
    X = df.copy()
    y = X.pop('Rating')
    X_encode = X.sample(frac=0.25)
    y_encode = y[X_encode.index]
    X_pretrain = X.drop(X_encode.index)
    y_train = y[X_pretrain.index]
    return X_encode, y_encode, X_pretrain, y_train

def encode_zipcode(X_encode, y_encode, X_pretrain, m=5.0):
    """
    Codifica la característica 'Zipcode' usando MEstimateEncoder.

    Args:
        X_encode (DataFrame): Conjunto de datos para codificación.
        y_encode (Series): Objetivo para codificación.
        X_pretrain (DataFrame): Conjunto de datos para entrenamiento.
        m (float): Factor de suavizado.

    Returns:
        DataFrame: DataFrame con la característica 'Zipcode' codificada.
    """
    encoder = MEstimateEncoder(cols=["Zipcode"], m=m)
    encoder.fit(X_encode, y_encode)
    X_train = encoder.transform(X_pretrain)
    return X_train

def visualize_encoding(y, X_train):
    """
    Visualiza la distribución del objetivo y la característica 'Zipcode' codificada.

    Args:
        y (Series): Serie del objetivo.
        X_train (DataFrame): DataFrame con la característica 'Zipcode' codificada.
    """
    plt.figure(dpi=90)
    ax = sns.distplot(y, kde=False, norm_hist=True)
    ax = sns.kdeplot(X_train.Zipcode, color='r', ax=ax)
    ax.set_xlabel("Rating")
    ax.legend(labels=['Zipcode', 'Rating'])
    plt.show()

# 1. Carga y preparación de datos
df = load_and_prepare_data("/home/pol/Downloads/movielens1m.csv")

# 2. División de datos
X_encode, y_encode, X_pretrain, y_train = split_data(df)
y = df['Rating'] # Aseguramos que 'y' se defina correctamente

# 3. Codificación de 'Zipcode'
X_train = encode_zipcode(X_encode, y_encode, X_pretrain)

# 4. Visualización de la codificación
visualize_encoding(y, X_train)

# Salidas: 
# Número de Zipcodes únicos: 3439

# Análisis por Gemini 2.0 Flash: (características_10.png):
# 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from category_encoders import MEstimateEncoder

# Configuración de visualización profesional
sns.set_theme(style="whitegrid", palette="pastel")
plt.rc("figure", autolayout=True, figsize=(10, 6))
plt.rc("axes", titlesize=14, titleweight="bold", labelsize=12)
warnings.filterwarnings('ignore')

def load_and_prepare_data(filepath):
    """
    Carga y optimiza el dataset MovieLens1M reduciendo el uso de memoria.
    
    Técnica clave:
    - Downcasting de tipos numéricos para eficiencia de memoria
    
    Args:
        filepath (str): Ruta al archivo CSV
    
    Returns:
        pd.DataFrame: Dataset optimizado
    """
    df = pd.read_csv(filepath)
    
    # Optimización segura de tipos numéricos
    for col in df.select_dtypes(include='integer'):
        df[col] = pd.to_numeric(df[col], downcast='unsigned')
    
    print(f"Análisis inicial: {df['Zipcode'].nunique()} códigos postales únicos")
    return df

def split_data(df, random_state=None):
    """
    Divide estratégicamente los datos para evitar data leakage.
    
    Estrategia:
    - 25% para calcular la codificación (conjunto de codificación)
    - 75% para entrenamiento real (conjunto de entrenamiento)
    
    Args:
        df (pd.DataFrame): Datos completos
        random_state (int): Semilla para reproducibilidad
    
    Returns:
        tuple: (X_encode, y_encode, X_train, y_train)
    """
    X = df.copy()
    y = X.pop('Rating')
    
    # División reproducible
    encode_idx = X.sample(frac=0.25, random_state=random_state).index
    X_encode = X.loc[encode_idx]
    y_encode = y.loc[encode_idx]
    X_train = X.drop(encode_idx)
    y_train = y.drop(encode_idx)
    
    return X_encode, y_encode, X_train, y_train

def encode_zipcode(X_encode, y_encode, X_train, m=5.0):
    """
    Aplica codificación de objetivo con suavizado M-estimado.
    
    Qué hace:
    - Reemplaza cada zipcode por el promedio de ratings suavizado
    - 'm' controla el balance entre media global y local (mayor m = más suavizado)
    
    Args:
        X_encode (pd.DataFrame): Datos para calcular la codificación
        y_encode (pd.Series): Target correspondiente
        X_train (pd.DataFrame): Datos a transformar
        m (float): Factor de suavizado
    
    Returns:
        pd.DataFrame: Datos de entrenamiento con zipcode codificado
    """
    encoder = MEstimateEncoder(cols=["Zipcode"], m=m)
    encoder.fit(X_encode, y_encode)
    return encoder.transform(X_train)

def visualize_encoding(y_true, X_encoded, title):
    """
    Visualiza relación entre la codificación y el target real.
    
    Interpretación ideal:
    - Distribución similar entre línea roja (codificación) y barras (ratings reales)
    - Gran discrepancia sugiere necesidad de ajustar el parámetro 'm'
    """
    plt.figure(dpi=90)
    
    # Histograma de ratings reales
    sns.histplot(y_true, stat='density', bins=30, alpha=0.5, label='Ratings Reales')
    
    # Densidad de la codificación
    sns.kdeplot(X_encoded['Zipcode'], color='r', linewidth=2, label='Codificación Zipcode')
    
    plt.title(f"Distribución comparativa\n{title}", pad=20)
    plt.xlabel("Valor (Escala Rating)")
    plt.legend()
    plt.show()

# Pipeline ejecutable ======================================================
if __name__ == "__main__":
    # 1. Carga inteligente
    df = load_and_prepare_data("/home/pol/Downloads/movielens1m.csv")
    
    # 2. División reproducible
    X_encode, y_encode, X_train, y_train = split_data(df, random_state=42)
    
    # 3. Codificación profesional
    X_encoded = encode_zipcode(X_encode, y_encode, X_train, m=5.0)
    
    # 4. Visualización ejecutiva
    visualize_encoding(y_train, X_encoded, "Codificación de Códigos Postales")

# Explicación por Perplexity: 
# 📚 Explicación para Novatos
# ¿Qué hace este código?
# Codificación de Zipcodes: Convierte códigos postales (ej: '10025') en valores numéricos que representan el rating promedio de esa zona, pero de forma segura para evitar sobreajuste.

# Truco clave: Usa dos subconjuntos de datos separados:

# Uno para calcular la codificación

# Otro para entrenar el modelo final
# (Así evitamos que el modelo "haga trampa" viendo los datos de prueba)

# Visualización: Muestra cómo se relacionan los ratings reales con los valores codificados:

# Barras: Frecuencia de ratings reales (1-5)

# Línea roja: Distribución de los valores codificados

# Meta: Que ambas formas sean similares pero suavizadas

# 💡 Consejos de Uso
# Ajusta el parámetro m:

# Valores bajos (ej: 1.0) = Confía más en los datos locales

# Valores altos (ej: 10.0) = Suaviza más hacia la media global

# Monitoriza la visualización:

# Si la línea roja es muy "nerviosa", aumenta m

# Si pierde relación con los ratings, disminuye m

# Prueba diferentes splits:

# Cambia el random_state para verificar estabilidad de resultados

# ¿Necesitas alguna aclaración adicional? 😊



# Análisis del gráfico (características_10.png):
# ----------------------------------------------
# La imagen muestra una comparación entre dos distribuciones relacionadas 
# con los datos de ratings de películas y la codificación de los códigos 
# postales (Zipcodes):

# Barras azules (Rating): Representan la frecuencia de los ratings reales 
# que los usuarios dieron a las películas. Por ejemplo, se observa que muchos 
# usuarios calificaron con 4.0, mientras que pocos usaron 1.0 o 2.0.

# Línea roja (Zipcode): Representa la distribución de los valores codificados 
# para los códigos postales (Zipcodes). Estos valores son el resultado del 
# promedio suavizado de los ratings en cada código postal.

# ¿Qué significa?
# La línea roja intenta aproximarse a la tendencia general de las barras azules, 
# pero con un nivel de suavizado. Esto indica que la codificación está capturando 
# información útil sobre cómo los códigos postales están relacionados con los ratings.

# Si la línea roja sigue un patrón razonablemente similar al de las barras, 
# significa que la codificación está funcionando bien y representa correctamente 
# la relación entre Zipcodes y Ratings.

# Para novatos:
# Barras: Muestran cómo calificaron las personas.

# Línea roja: Muestra cómo el modelo interpreta esos datos según los códigos postales.

# Meta: La línea roja debe parecerse a las barras, pero sin ser demasiado 
# exacta (esto evita que el modelo "memorice" en lugar de aprender patrones generales).



# Sí, se puede afirmar que servicios de streaming como Netflix monitorean 
# información relacionada con el código postal (Zipcode), aunque no 
# directamente en la forma visualizada en tu gráfico. Este monitoreo ocurre 
# principalmente para personalizar la experiencia del usuario y cumplir con 
# regulaciones de licencias regionales.

# ¿Cómo monitorea Netflix el código postal?
# Uso del IP y ubicación geográfica:

# Netflix utiliza la dirección IP de los usuarios para determinar su ubicación 
# aproximada, que incluye el código postal, ciudad, estado o provincia.

# Esto les permite aplicar restricciones de contenido basadas en licencias 
# regionales y ofrecer recomendaciones personalizadas según las preferencias 
# de usuarios en esa área.

# Datos demográficos:

# Durante la creación de cuentas, los usuarios proporcionan información como 
# su código postal. Esta información puede ser utilizada para segmentar audiencias 
# y personalizar contenido o publicidad.

# Análisis de patrones por región:

# Netflix ha sido conocido por analizar patrones de consumo por códigos postales, 
# como lo demuestra un estudio que visualizó las preferencias de películas por 
# áreas geográficas en EE.UU.. Este tipo de análisis ayuda a identificar tendencias 
# regionales y ajustar estrategias de contenido.

# ¿Qué relación tiene esto con tu gráfico?
# El gráfico que generaste muestra cómo los códigos postales (Zipcodes) están 
# relacionados con los ratings promedio de los usuarios mediante una técnica 
# de codificación. Aunque Netflix probablemente no utilice este tipo de codificación 
# directamente, sí analiza datos geográficos y demográficos para entender mejor 
# las preferencias de sus usuarios. Por ejemplo:

# Si en un área específica hay una alta demanda de cierto género o película, 
# Netflix podría priorizar ese contenido para usuarios similares en esa región.

# En resumen, Netflix monitorea códigos postales como parte de su estrategia 
# para personalizar contenido y cumplir con regulaciones, pero no necesariamente 
# utiliza técnicas como la codificación que se muestra en tu análisis.

# El tipo de código que proporcionaste, basado en la codificación de objetivo 
# (target encoding), es ampliamente utilizado en modelos de machine learning y 
# análisis predictivo en diversas industrias. A continuación, te explico quiénes 
# podrían usar este enfoque y para qué:

# ¿Quién usaría este tipo de código?
# Empresas de Marketing y Publicidad

# Caso de uso: Predecir la probabilidad de que un cliente compre un producto.

# Cómo lo usan: Transforman variables categóricas como ubicación (Zipcode), 
# género o preferencias en valores numéricos basados en el comportamiento 
# promedio de compra. Por ejemplo, podrían analizar si ciertos códigos postales 
# tienen tasas más altas de conversión.

# Plataformas de Streaming (como Netflix o Spotify)

# Caso de uso: Personalizar recomendaciones.

# Cómo lo usan: Analizan patrones de comportamiento por región (basados en Zipcode) 
# para identificar qué contenido es más popular en ciertas áreas. Esto les ayuda a 
# priorizar qué series o películas recomendar.

# Instituciones Financieras

# Caso de uso: Evaluar riesgos crediticios.

# Cómo lo usan: Codifican variables como el propósito del préstamo o ubicación 
# del solicitante (por ejemplo, Zipcode) para predecir la probabilidad de impago.

# Retail y E-commerce

# Caso de uso: Optimizar inventarios y campañas.

# Cómo lo usan: Utilizan la codificación para identificar patrones regionales 
# en las compras. Por ejemplo, ciertos productos podrían ser más populares en 
# códigos postales específicos.

# Telecomunicaciones

# Caso de uso: Predicción del abandono de clientes (churn).

# Cómo lo usan: Codifican variables como el tipo de contrato o ubicación para 
# entender mejor qué características están asociadas con clientes que cancelan 
# sus servicios.

# ¿Por qué usarían este enfoque?
# Manejo de datos categóricos complejos: La codificación por objetivo es 
# especialmente útil cuando se trabaja con variables categóricas de alta cardinalidad, 
# como los códigos postales, que tienen miles de valores únicos.

# Captura patrones relevantes: Al reemplazar categorías con estadísticas basadas en el 
# objetivo (como el promedio), se introducen patrones significativos que ayudan al 
# modelo a hacer predicciones más precisas.

# Eficiencia computacional: A diferencia del one-hot encoding, no aumenta la 
# dimensionalidad del conjunto de datos, lo que lo hace más eficiente.

# Ejemplo práctico: Netflix y Zipcodes
# Imagina que Netflix quiere predecir qué tan probable es que un usuario califique 
# una película con 5 estrellas dependiendo del código postal:

# Dividen los datos históricos en dos subconjuntos:

# Uno para calcular el promedio de ratings por Zipcode.

# Otro para entrenar el modelo con esos valores codificados.

# Usan la codificación para transformar cada Zipcode en un valor numérico basado 
# en el promedio histórico.

# Entrenan el modelo para hacer predicciones personalizadas según los patrones observados.

# En este caso, los Zipcodes podrían revelar tendencias como:

# En ciertas áreas urbanas, las películas románticas reciben mejores calificaciones.

# En áreas rurales, los documentales tienen mayor aceptación.

# En resumen, este tipo de código es una herramienta poderosa para transformar 
# datos categóricos complejos (como Zipcodes) en información útil para modelos 
# predictivos, ayudando a empresas a tomar decisiones más informadas y personalizadas.


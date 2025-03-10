import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

# Cargar los datos
data = pd.read_csv('/home/pol/Downloads/melb_data.csv')

# Seleccionar un subconjunto de predictores
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Seleccionar la variable objetivo
y = data.Price

# import pandas as pd: Importa la biblioteca pandas para la manipulación de datos.
# from sklearn.ensemble import RandomForestRegressor: Importa el 
# modelo de bosque aleatorio para regresión.
# from sklearn.pipeline import Pipeline: Importa la clase Pipeline 
# para crear una secuencia de transformaciones y un modelo.
# from sklearn.impute import SimpleImputer: Importa la clase 
# SimpleImputer para manejar valores faltantes.
# from sklearn.model_selection import cross_val_score: Importa 
# la función cross_val_score para realizar la validación cruzada.
# data = pd.read_csv('/home/pol/Downloads/melb_data.csv'): Carga 
# el conjunto de datos desde un archivo CSV.
# cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']: 
# Define las columnas que se utilizarán como predictores.
# X = data[cols_to_use]: Selecciona las columnas predictoras y las asigna a la variable X.
# y = data.Price: Selecciona la variable objetivo (precio) y la asigna a la variable y.

# Definir la pipeline
my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', RandomForestRegressor(n_estimators=50,
                                                              random_state=0))
                             ])
# my_pipeline = Pipeline(steps=[...]): Crea una pipeline que incluye dos pasos:
# ('preprocessor', SimpleImputer()): Un paso de preprocesamiento 
# que imputa los valores faltantes con la media.
# ('model', RandomForestRegressor(n_estimators=50, random_state=0)): 
# Un paso de modelado que utiliza un modelo de bosque aleatorio con 50 árboles.


# Obtener las puntuaciones de validación cruzada
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')
# scores = -1 * cross_val_score(my_pipeline, X, y, cv=5, 
# scoring='neg_mean_absolute_error'): Realiza la validación 
# cruzada utilizando la pipeline definida anteriormente:
# my_pipeline: La pipeline que se utilizará para el modelado.
# X: Las variables predictoras.
# y: La variable objetivo.
# cv=5: El número de pliegues (folds) para la validación cruzada.
# scoring='neg_mean_absolute_error': La métrica de evaluación utilizada 
# (error absoluto medio negativo). Se multiplica por -1 porque scikit-learn 
# utiliza valores negativos para las métricas de error.

# Imprimir las puntuaciones MAE
print("MAE scores:\n", scores)
#Imprime las puntuaciones MAE obtenidas en cada pliegue de la validación cruzada.

# Imprimir la puntuación MAE promedio
print("Average MAE score (across experiments):")
print(scores.mean())
# print("Average MAE score (across experiments):"): Imprime el texto "Average MAE score (across experiments):".
# print(scores.mean()): Calcula e imprime la puntuación MAE promedio de todos los pliegues.

# MAE scores:
#  [301628.7893587  303164.4782723  287298.331666   236061.84754543
#  260383.45111427]
# Average MAE score (across experiments):
# 277707.3795913405

# """
# Puntuaciones MAE:

# [301628.7893587 303164.4782723 287298.331666 236061.84754543 260383.45111427]

# Esta lista muestra las puntuaciones del Error Absoluto Medio 
# (MAE, por sus siglas en inglés) para cada uno de los 5 pliegues 
# (experimentos) en tu validación cruzada. Cada puntuación representa 
# el error absoluto promedio entre los valores predichos y reales para 
# el conjunto de validación en ese pliegue en particular.

# Puntuación MAE Promedio:

# 277707.3795913405

# Este es el promedio de las 5 puntuaciones MAE. Te da una medida general
# de qué tan bien se está desempeñando tu modelo con datos no vistos, 
# teniendo en cuenta las variaciones entre los diferentes pliegues.

# Interpretación:

# Rendimiento del Modelo: La puntuación MAE promedio de 277,707 sugiere que, 
# en promedio, las predicciones de tu modelo se desvían en aproximadamente 
# $277,707 de los precios reales de las viviendas. Esto te da una idea general 
# de la precisión del modelo.

# Variación Entre Pliegues: Las puntuaciones MAE individuales para cada 
# pliegue muestran cierta variación. Esto es de esperar, ya que los 
# diferentes pliegues representan diferentes subconjuntos de los datos, 
# y el modelo podría funcionar ligeramente mejor o peor en ciertos subconjuntos.

# Beneficio de la Validación Cruzada: Al usar la validación cruzada, 
# obtienes una estimación más sólida del rendimiento de tu modelo en 
# comparación con el uso de un solo conjunto de validación.
# La puntuación MAE promedio entre los pliegues tiene menos probabilidades 
# de verse influenciada por la elección específica de un solo conjunto de validación.

# Próximos Pasos:

# Comparar Modelos: Puedes usar la puntuación MAE promedio para comparar 
# diferentes modelos o diferentes configuraciones de hiperparámetros para 
# el mismo modelo. Las puntuaciones MAE más bajas generalmente indican un 
# mejor rendimiento.

# Analizar la Variación: Si hay una variación significativa en las 
# puntuaciones MAE entre los pliegues, podría indicar que tu modelo 
# es sensible a los datos específicos con los que se entrena. Podrías 
# investigar más a fondo para comprender por qué está sucediendo esto.

# Afinar el Modelo: Según los resultados de la validación cruzada, 
# puedes afinar tu modelo ajustando los hiperparámetros, probando 
# diferentes pasos de preprocesamiento o explorando otros tipos de 
# modelos para mejorar su rendimiento.

# Recuerda, la validación cruzada es una herramienta valiosa para 
# evaluar el rendimiento del modelo y tomar decisiones informadas 
# sobre la selección y el ajuste del modelo.
# """

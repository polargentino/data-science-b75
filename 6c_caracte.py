import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

def demostracion_ingenieria_caracteristicas():
    """
    Demuestra varias técnicas de ingeniería de características 
    utilizando diferentes conjuntos de datos.

    Perfecto, ahora vemos que tu script se está ejecutando correctamente y 
    produciendo los resultados esperados. Los ejemplos que has proporcionado 
    muestran las transformaciones de ingeniería de características que estás 
    aplicando a tus conjuntos de datos.

Aquí hay un resumen de lo que vemos en la salida:

1. Transformaciones Matemáticas (Autos):
----------------------------------------
Se calcula la relacion_carrera dividiendo stroke entre bore.
Se calcula el desplazamiento usando una fórmula que involucra bore, stroke y el 
número de cilindros.

2. Características de Conteo (Accidentes y Concreto):
-----------------------------------------------------
Se cuenta el número de características de carretera presentes en cada fila del 
conjunto de datos de accidentes (CaracteristicasCarretera).
Se cuenta el número de componentes presentes en cada fila del conjunto de datos de 
concreto (Componentes).

3. Manipulación de Cadenas (Cliente):
-------------------------------------
Se divide la columna 'Policy' en 'Tipo' y 'Nivel'.

4. Combinación de Características (Autos):
-----------------------------------------
Se crea la característica marca_y_estilo combinando las columnas 'make' y 'body_style'.

5. Transformación de Grupo (Cliente):
-------------------------------------
Se calcula el IngresoPromedio para cada estado.
Se calcula la FrecuenciaEstado para cada estado.
Se calcula el ReclamoPromedio para cada tipo de cobertura, dividiendo el conjunto 
de datos en entrenamiento y validación.

Puntos clave:
-------------
Tu script está funcionando correctamente y realizando las transformaciones de ingeniería 
de características según lo previsto.
Los ejemplos de salida muestran los resultados de cada transformación.
Estás utilizando seaborn para visualizar las distribuciones de datos, lo cual es 
una buena práctica.
Estás manejando las rutas de los archivos correctamente utilizando os.path.join y 
os.path.expanduser.
Estas usando correctamente el método str.split() para separar cadenas de texto en columnas.
Se maneja correctamente el merge de DataFrames para unir información de dataframes divididos.

Posibles mejoras:(tarea)
------------------------
Considera agregar más visualizaciones para comprender mejor los datos transformados.
Puedes explorar otras técnicas de ingeniería de características, como la creación 
de características polinómicas o la codificación de variables categóricas.
Si los conjuntos de datos son grandes, podrías optimizar el script para mejorar 
el rendimiento.
Considera la posibilidad de guardar los dataframes resultantes en archivos csv nuevos 
para su posterior uso.


El código que genera el gráfico y la transformación logarítmica de la velocidad 
del viento en el conjunto de datos de accidentes tiene varios propósitos y busca 
demostrar lo siguiente:

1. Manejo de datos asimétricos:
-------------------------------
El objetivo principal es ilustrar cómo se pueden transformar datos que no siguen 
una distribución normal (asimétricos) para hacerlos más adecuados para el análisis 
y el modelado.
La velocidad del viento, como se muestra en el gráfico de la izquierda, tiende a 
tener una distribución sesgada, con muchos valores bajos y algunos valores altos. 
La transformación logarítmica ayuda a "normalizar" esta distribución.

2. Reducción del impacto de valores atípicos:
---------------------------------------------
Los valores atípicos (outliers) pueden tener un impacto desproporcionado en los 
análisis estadísticos y los modelos predictivos. La transformación logarítmica 
reduce la influencia de estos valores extremos.

3. Mejora del análisis y el modelado:
--------------------------------------
Los datos transformados son más fáciles de analizar mediante técnicas estadísticas que 
asumen una distribución normal.
Los modelos de aprendizaje automático, especialmente aquellos que asumen normalidad 
en los datos, pueden funcionar mejor con datos transformados.

4. Visualización del efecto de la transformación:
-------------------------------------------------
El gráfico permite visualizar claramente cómo la transformación logarítmica cambia 
la forma de la distribución de los datos. Esto ayuda a comprender el efecto de 
la transformación y a evaluar su utilidad.

5. Demostración de técnicas de ingeniería de características:
-------------------------------------------------------------
Este código es un ejemplo de una técnica común de ingeniería de características, 
que implica transformar variables para hacerlas más útiles para el análisis y el modelado.
En resumen, el código busca demostrar cómo la transformación logarítmica puede 
mejorar la calidad de los datos y facilitar el análisis y el modelado. No pretende 
establecer una relación causal directa entre el viento y los accidentes, sino más 
bien mostrar cómo se pueden manejar y transformar los datos para hacerlos más informativos.
    """

    # Obtener la ruta del directorio de descargas
    ruta_descargas = os.path.expanduser("~/Downloads")

    # Cargar conjuntos de datos
    accidentes = pd.read_csv(os.path.join(ruta_descargas, "accidents.csv"))
    autos = pd.read_csv(os.path.join(ruta_descargas, "autos.csv"))
    concreto = pd.read_csv(os.path.join(ruta_descargas, "concrete.csv"))
    cliente = pd.read_csv(os.path.join(ruta_descargas, "customer.csv"))

    # 1. Transformaciones Matemáticas (Conjunto de datos de Autos)
    # Crear la característica 'relacion_carrera'
    autos["relacion_carrera"] = autos.stroke / autos.bore
    print("\nEjemplo de Relación de Carrera:")
    print(autos[["stroke", "bore", "relacion_carrera"]].head())

    # Crear la característica 'desplazamiento'
    autos["desplazamiento"] = (
        np.pi * ((0.5 * autos.bore) ** 2) * autos.stroke * autos.num_of_cylinders
    )

    # 2. Transformación Logarítmica (Conjunto de datos de Accidentes)
    # Aplicar la transformación log1p a 'VelocidadViento'
    accidentes["LogVelocidadViento"] = accidentes.WindSpeed.apply(np.log1p)

    # Graficar las distribuciones originales y transformadas de 'VelocidadViento'
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    sns.kdeplot(accidentes.WindSpeed, fill=True, ax=axs[0])
    sns.kdeplot(accidentes.LogVelocidadViento, fill=True, ax=axs[1])
    plt.show()

    # 3. Características de Conteo (Conjunto de datos de Accidentes)
    # Crear la característica de conteo 'CaracteristicasCarretera'
    caracteristicas_carretera = [
        "Amenity", "Bump", "Crossing", "GiveWay", "Junction", "NoExit", "Railway",
        "Roundabout", "Station", "Stop", "TrafficCalming", "TrafficSignal"
    ]
    accidentes["CaracteristicasCarretera"] = accidentes[caracteristicas_carretera].sum(axis=1)
    print("\nEjemplo de Conteo de Características de Carretera:")
    print(accidentes[caracteristicas_carretera + ["CaracteristicasCarretera"]].head(10))

    # 4. Características de Conteo con Booleanos (Conjunto de datos de Concreto)
    # Crear la característica de conteo 'Componentes'
    componentes = [
        "Cement", "BlastFurnaceSlag", "FlyAsh", "Water",
        "Superplasticizer", "CoarseAggregate", "FineAggregate"
    ]
    concreto["Componentes"] = concreto[componentes].gt(0).sum(axis=1)
    print("\nEjemplo de Conteo de Componentes de Concreto:")
    print(concreto[componentes + ["Componentes"]].head(10))

    # 5. Manipulación de Cadenas (Conjunto de datos de Cliente)
    # Dividir 'Policy' en 'Tipo' y 'Nivel'
    cliente[["Tipo", "Nivel"]] = (
        cliente["Policy"].str.split(" ", expand=True)
    )
    print("\nEjemplo de División de Póliza:")
    print(cliente[["Policy", "Tipo", "Nivel"]].head(10))

    # 6. Combinación de Características (Conjunto de datos de Autos)
    # Crear la característica 'marca_y_estilo'
    autos["marca_y_estilo"] = autos["make"] + "_" + autos["body_style"]
    print("\nEjemplo de Combinación de Marca y Estilo:")
    print(autos[["make", "body_style", "marca_y_estilo"]].head())

    # 7. Transformación de Grupo (Conjunto de datos de Cliente)
    # Crear la característica 'IngresoPromedio'
    cliente["IngresoPromedio"] = (
        cliente.groupby("State")["Income"].transform("mean")
    )
    print("\nEjemplo de Ingreso Promedio por Estado:")
    print(cliente[["State", "Income", "IngresoPromedio"]].head(10))

    # 8. Codificación de Frecuencia (Conjunto de datos de Cliente)
    # Crear la característica 'FrecuenciaEstado'
    cliente["FrecuenciaEstado"] = (
        cliente.groupby("State")["State"].transform("count") / cliente.State.count()
    )
    print("\nEjemplo de Frecuencia de Estado:")
    print(cliente[["State", "FrecuenciaEstado"]].head(10))

    # 9. Transformación de Grupo con División Entrenamiento/Validación (Conjunto de datos de Cliente)
    # Crear divisiones de entrenamiento y validación
    df_entrenamiento = cliente.sample(frac=0.5)
    df_validacion = cliente.drop(df_entrenamiento.index)

    # Crear 'ReclamoPromedio' en el conjunto de entrenamiento
    df_entrenamiento["ReclamoPromedio"] = df_entrenamiento.groupby("Coverage")["ClaimAmount"].transform("mean")

    # Combinar 'ReclamoPromedio' en el conjunto de validación
    df_validacion = df_validacion.merge(
        df_entrenamiento[["Coverage", "ReclamoPromedio"]].drop_duplicates(),
        on="Coverage",
        how="left",
    )
    print("\nEjemplo de Reclamo Promedio por Cobertura (Entrenamiento/Validación):")
    print(df_validacion[["Coverage", "ReclamoPromedio"]].head(10))

if __name__ == "__main__":
    demostracion_ingenieria_caracteristicas()

# Salidas: 
# Ejemplo de Relación de Carrera:
#    stroke  bore  relacion_carrera
# 0    2.68  3.47          0.772334
# 1    2.68  3.47          0.772334
# 2    3.47  2.68          1.294776
# 3    3.40  3.19          1.065831
# 4    3.40  3.19          1.065831

# Ejemplo de Conteo de Características de Carretera:
#    Amenity   Bump  Crossing  ...  TrafficCalming  TrafficSignal  CaracteristicasCarretera
# 0    False  False     False  ...           False          False                         0
# 1    False  False     False  ...           False          False                         0
# 2    False  False     False  ...           False          False                         0
# 3    False  False     False  ...           False          False                         0
# 4    False  False     False  ...           False          False                         0
# 5    False  False     False  ...           False          False                         1
# 6    False  False     False  ...           False          False                         0
# 7    False  False      True  ...           False           True                         2
# 8    False  False      True  ...           False           True                         2
# 9    False  False     False  ...           False          False                         0

# [10 rows x 13 columns]

# Ejemplo de Conteo de Componentes de Concreto:
#    Cement  BlastFurnaceSlag  FlyAsh  ...  CoarseAggregate  FineAggregate  Componentes
# 0   540.0               0.0     0.0  ...           1040.0          676.0            5
# 1   540.0               0.0     0.0  ...           1055.0          676.0            5
# 2   332.5             142.5     0.0  ...            932.0          594.0            5
# 3   332.5             142.5     0.0  ...            932.0          594.0            5
# 4   198.6             132.4     0.0  ...            978.4          825.5            5
# 5   266.0             114.0     0.0  ...            932.0          670.0            5
# 6   380.0              95.0     0.0  ...            932.0          594.0            5
# 7   380.0              95.0     0.0  ...            932.0          594.0            5
# 8   266.0             114.0     0.0  ...            932.0          670.0            5
# 9   475.0               0.0     0.0  ...            932.0          594.0            4

# [10 rows x 8 columns]

# Ejemplo de División de Póliza:
#          Policy       Tipo Nivel
# 0  Corporate L3  Corporate    L3
# 1   Personal L3   Personal    L3
# 2   Personal L3   Personal    L3
# 3  Corporate L2  Corporate    L2
# 4   Personal L1   Personal    L1
# 5   Personal L3   Personal    L3
# 6  Corporate L3  Corporate    L3
# 7  Corporate L3  Corporate    L3
# 8  Corporate L3  Corporate    L3
# 9    Special L2    Special    L2

# Ejemplo de Combinación de Marca y Estilo:
#           make   body_style           marca_y_estilo
# 0  alfa-romero  convertible  alfa-romero_convertible
# 1  alfa-romero  convertible  alfa-romero_convertible
# 2  alfa-romero    hatchback    alfa-romero_hatchback
# 3         audi        sedan               audi_sedan
# 4         audi        sedan               audi_sedan

# Ejemplo de Ingreso Promedio por Estado:
#         State  Income  IngresoPromedio
# 0  Washington   56274     38122.733083
# 1     Arizona       0     37405.402231
# 2      Nevada   48767     38369.605442
# 3  California       0     37558.946667
# 4  Washington   43836     38122.733083
# 5      Oregon   62902     37557.283353
# 6      Oregon   55350     37557.283353
# 7     Arizona       0     37405.402231
# 8      Oregon   14072     37557.283353
# 9      Oregon   28812     37557.283353

# Ejemplo de Frecuencia de Estado:
#         State  FrecuenciaEstado
# 0  Washington          0.087366
# 1     Arizona          0.186446
# 2      Nevada          0.096562
# 3  California          0.344865
# 4  Washington          0.087366
# 5      Oregon          0.284760
# 6      Oregon          0.284760
# 7     Arizona          0.186446
# 8      Oregon          0.284760
# 9      Oregon          0.284760

# Ejemplo de Reclamo Promedio por Cobertura (Entrenamiento/Validación):
#    Coverage  ReclamoPromedio
# 0     Basic       376.549222
# 1  Extended       481.117097
# 2     Basic       376.549222
# 3     Basic       376.549222
# 4  Extended       481.117097
# 5     Basic       376.549222
# 6     Basic       376.549222
# 7     Basic       376.549222
# 8   Premium       658.690379
# 9     Basic       376.549222

# Análisis por: https://gemini.google.com/(2.0 Flash):
#-----------------------------------------------------
# Análisis Detallado:

# 1. Contexto General:

# "Este análisis se centra en demostrar el poder de la ingeniería de 
# características para extraer información valiosa de nuestros datos. 
# A través de la manipulación y transformación de múltiples conjuntos 
# de datos CSV, hemos creado nuevas características que revelan patrones y relaciones ocultas. Este proceso no solo mejora la comprensión de nuestros datos, sino que también sienta las bases para modelos predictivos más precisos."

# 2. Transformaciones Matemáticas (autos.csv):

# "En el conjunto de datos autos.csv, aplicamos transformaciones matemáticas 
# para crear dos nuevas características clave: relacion_carrera y desplazamiento. 
# La relacion_carrera, calculada como stroke / bore, nos proporciona información 
# sobre la configuración del motor, lo que puede influir en el rendimiento del 
# vehículo. El desplazamiento, calculado a partir de las dimensiones del cilindro 
# y el número de cilindros, es un indicador directo de la capacidad del motor. 
# Estas transformaciones convierten datos crudos en características significativas 
# que capturan aspectos fundamentales del diseño del motor."

# 3. Transformación Logarítmica y Visualización (accidents.csv):

# "El conjunto de datos accidents.csv presentaba una distribución asimétrica 
# en la característica WindSpeed. Para abordar esto, aplicamos una transformación 
# logarítmica (log1p) y creamos la característica LogVelocidadViento. 
# La visualización muestra claramente cómo la transformación logarítmica 
# ha normalizado la distribución, lo que facilita el análisis y mejora el 
# rendimiento de los modelos predictivos. Esta transformación es crucial 
# para manejar datos asimétricos y garantizar que nuestros modelos no se 
# vean sesgados por valores atípicos."

# 4. Características de Conteo (accidents.csv y concrete.csv):

# "Creamos características de conteo en los conjuntos de datos accidents.csv 
# y concrete.csv. En accidents.csv, contamos el número de características de 
# la carretera presentes en cada accidente, como cruces y señales de tráfico, 
# creando la característica CaracteristicasCarretera. En concrete.csv, contamos 
# el número de componentes presentes en cada mezcla de concreto, creando la 
# característica Componentes. Estas características de conteo resumen información 
# compleja en valores numéricos simples, lo que facilita la identificación de 
# patrones y relaciones."

# 5. Manipulación de Cadenas (customer.csv):

# "En el conjunto de datos customer.csv, dividimos la columna Policy en Tipo y Nivel. 
# Esta manipulación de cadenas nos permite analizar las pólizas de seguro con mayor 
# detalle y descubrir patrones específicos asociados con diferentes tipos y niveles 
# de pólizas. Por ejemplo, podríamos encontrar que ciertos tipos de pólizas están 
# relacionados con mayores montos de reclamaciones."

# 6. Combinación de Características (autos.csv):

# "Combinamos las columnas make y body_style en el conjunto de datos autos.csv para 
# crear la característica marca_y_estilo. Esta combinación nos permite analizar la 
# relación entre la marca y el estilo del vehículo, lo que puede ser útil para predecir 
# el precio o el rendimiento del vehículo. Por ejemplo, podríamos encontrar que ciertos 
# estilos de carrocería son más populares entre ciertas marcas."

# 7. Transformación de Grupo (customer.csv):

# "Aplicamos transformaciones de grupo al conjunto de datos customer.csv para 
# calcular el IngresoPromedio por estado y la FrecuenciaEstado. Estas transformaciones 
# nos permiten analizar cómo varían los ingresos y la frecuencia de los clientes entre 
# diferentes estados. Además, dividimos el conjunto de datos en entrenamiento y 
# validación para calcular el ReclamoPromedio por cobertura, lo que nos permite 
# evaluar el rendimiento de nuestros modelos en datos no vistos."

# 8. Codificación de Frecuencia (customer.csv):

# "Calculamos la FrecuenciaEstado para cada estado en el conjunto de datos customer.csv. 
# Esta característica nos permite analizar la distribución de clientes entre diferentes 
# estados y puede ser útil para identificar patrones geográficos en los datos."

# Conclusión Impactante:

# "Este análisis demuestra el poder de la ingeniería de características para 
# transformar datos crudos en información valiosa. A través de una combinación de 
# transformaciones matemáticas, manipulación de cadenas, características de conteo y 
# transformaciones de grupo, hemos creado nuevas características que revelan patrones 
# y relaciones ocultas en nuestros conjuntos de datos. Este proceso no solo mejora la 
# comprensión de nuestros datos, sino que también sienta las bases para modelos 
# predictivos más precisos y robustos. Con estas técnicas, podemos tomar decisiones 
# más informadas y obtener una ventaja competitiva en el mercado."

# Consejos para los Novatos:

# Comprender los datos: Antes de aplicar cualquier transformación, es crucial 
# comprender los datos y las relaciones entre las características.
# Experimentar con diferentes técnicas: No tengas miedo de experimentar con 
# diferentes técnicas de ingeniería de características para encontrar las que 
# mejor se adapten a tus datos.
# Visualizar los resultados: La visualización es una herramienta poderosa para 
# comprender el impacto de las transformaciones y detectar patrones en los datos.
# Documentar el proceso: Documenta cada paso del proceso de ingeniería de 
# características para que otros puedan entender y reproducir tu trabajo.
# ¡Espero que este análisis impresione a tu jefe y sea útil para los novatos!


# ¿Qué dice el gráfico? : El gráfico muestra la distribución de la velocidad 
# del viento (WindSpeed) antes  y después de aplicar una transformación 
# logarítmica (LogVelocidadViento). Aquí hay un análisis detallado:

# Gráfico de la izquierda (WindSpeed):
# ------------------------------------
# Asimetría: La distribución de WindSpeed está fuertemente sesgada hacia la derecha. 
# Esto significa que hay muchos valores bajos y relativamente pocos valores altos.
# Valores atípicos: La cola larga hacia la derecha sugiere la presencia de valores 
# atípicos (outliers), es decir, velocidades de viento extremadamente altas en 
# comparación con la mayoría de los datos.
# Dificultad para el análisis: La asimetría y los valores atípicos pueden 
# dificultar el análisis estadístico y el modelado predictivo, ya que pueden 
# sesgar los resultados.

# Gráfico de la derecha (LogVelocidadViento):
# ------------------------------------------
# Transformación logarítmica: Se ha aplicado una transformación logarítmica 
# (log1p, que es log(1 + x)) a la variable WindSpeed.
# Normalización: La distribución de LogVelocidadViento es mucho más simétrica y 
# cercana a una distribución normal. Esto facilita el análisis estadístico y 
# mejora el rendimiento de los modelos predictivos.
# Reducción del impacto de valores atípicos: La transformación logarítmica reduce 
# el impacto de los valores atípicos, ya que comprime los valores altos y expande 
# los valores bajos.

# En resumen:
# ----------
# El gráfico de la izquierda muestra una distribución asimétrica de la velocidad 
# del viento, con valores atípicos presentes.
# El gráfico de la derecha muestra la distribución transformada logarítmicamente, 
# que es mucho más simétrica y normal, lo que facilita el análisis y el modelado.
# Implicaciones:

# La transformación logarítmica es una técnica útil para manejar datos asimétricos y 
# reducir el impacto de los valores atípicos.
# La normalización de los datos puede mejorar el rendimiento de los modelos 
# predictivos y hacer que los resultados sean más confiables.





# ¿Qué relación tiene el gráfico con el código???
# -----------------------------------------------

# Claro, la relación entre el gráfico y el código es directa y fundamental. 
# El gráfico es la representación visual de una transformación específica que se 
# realiza en el código. Aquí te explico la conexión paso a paso:

# 1. Carga de Datos y Transformación:
# -----------------------------------
# El código comienza cargando el archivo accidents.csv en un DataFrame de pandas 
# llamado accidentes.
# Luego, se aplica una transformación logarítmica a la columna WindSpeed utilizando 
# la función np.log1p (logaritmo natural de 1 + x). Esta transformación se realiza 
# con la siguiente línea de código:
# Python:
# -------
# accidentes["LogVelocidadViento"] = accidentes.WindSpeed.apply(np.log1p)
# El resultado de esta transformación se almacena en una nueva columna llamada 
# LogVelocidadViento dentro del mismo DataFrame.

# 2. Creación del Gráfico:
# -------------------------
# El código utiliza las bibliotecas matplotlib y seaborn para crear el gráfico.
# Se crea una figura con dos subgráficos (axes) utilizando plt.subplots.
# Se utiliza sns.kdeplot para generar gráficos de densidad de kernel (KDE) 
# que muestran la distribución de WindSpeed y LogVelocidadViento.
# La primera llamada a sns.kdeplot grafica la distribución de la columna original WindSpeed:
# Python:
# -------
# sns.kdeplot(accidentes.WindSpeed, fill=True, ax=axs[0])
# La segunda llamada a sns.kdeplot grafica la distribución de la columna transformada LogVelocidadViento:
# Python:
# -------
# sns.kdeplot(accidentes.LogVelocidadViento, fill=True, ax=axs[1])
# Finalmente, plt.show() muestra el gráfico en la pantalla.

# 3. Relación Directa:
# -------------------
# El gráfico de la izquierda muestra la distribución de los datos antes de la 
# transformación logarítmica, tal como están en la columna WindSpeed del DataFrame.
# El gráfico de la derecha muestra la distribución de los datos después de la 
# transformación logarítmica, tal como están en la columna LogVelocidadViento del DataFrame.
# En resumen, el código realiza una transformación matemática en los datos, y 
# el gráfico visualiza el efecto de esa transformación. El gráfico permite 
# comprender cómo la transformación logarítmica cambia la forma de la distribución
# de la velocidad del viento, haciéndola más simétrica y reduciendo el impacto de 
# los valores atípicos.
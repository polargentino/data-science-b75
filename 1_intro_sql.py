# Importar la biblioteca necesaria para interactuar con Google BigQuery
from google.cloud import bigquery

# 1. Crear un objeto "Client"
# Este objeto actúa como un punto de entrada para todas las interacciones con BigQuery.
# Es como abrir una sesión con el servicio BigQuery.
client = bigquery.Client()

# 2. Construir una referencia al dataset "hacker_news"
# En BigQuery, los datasets son contenedores de tablas.
# "hacker_news" es un dataset público que contiene datos de la plataforma Hacker News.
# "bigquery-public-data" es el proyecto que contiene este dataset.
# Esta línea crea una referencia al dataset, pero aún no lo descarga.
dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")

# 3. Descargar el dataset usando la referencia
# Esta línea realiza una llamada a la API de BigQuery para obtener el dataset real.
# Ahora tenemos el objeto 'dataset' que representa el dataset "hacker_news".
dataset = client.get_dataset(dataset_ref)

# 4. Listar todas las tablas en el dataset "hacker_news"
# Un dataset puede contener múltiples tablas, como hojas en una hoja de cálculo.
# Esta línea obtiene una lista de todas las tablas en el dataset.
tables = list(client.list_tables(dataset))

# 5. Imprimir los nombres de todas las tablas en el dataset
# Este bucle itera sobre la lista de tablas e imprime el nombre de cada tabla.
# Los nombres de las tablas se almacenan en el atributo 'table_id'.
for table in tables:
    print(table.table_id)

# 6. Construir una referencia a la tabla "full"
# "full" es una de las tablas en el dataset "hacker_news".
# Esta línea crea una referencia a la tabla, pero aún no la descarga.
table_ref = dataset_ref.table("full")

# 7. Descargar la tabla usando la referencia
# Esta línea realiza una llamada a la API de BigQuery para obtener la tabla real.
# Ahora tenemos el objeto 'table' que representa la tabla "full".
table = client.get_table(table_ref)

# 8. Previsualizar las primeras cinco filas de la tabla "full"
# 'client.list_rows()' obtiene las filas de la tabla.
# 'max_results=5' limita el número de filas a 5.
# '.to_dataframe()' convierte las filas a un DataFrame de pandas para facilitar la visualización.
full_table_preview = client.list_rows(table, max_results=5).to_dataframe()

# 9. Imprimir las primeras cinco filas de la tabla "full"
# Imprimimos el DataFrame para ver las primeras filas de la tabla.
print("\nPrimeras cinco filas de la tabla 'full':")
print(full_table_preview)

# 10. Previsualizar las primeras cinco entradas de la columna "by" de la tabla "full"
# 'table.schema[:1]' selecciona la primera columna del esquema de la tabla, que es la columna "by".
# 'selected_fields' especifica las columnas que queremos recuperar.
by_column_preview = client.list_rows(table, selected_fields=table.schema[:1], max_results=5).to_dataframe()

# 11. Imprimir las primeras cinco entradas de la columna "by"
# Imprimimos el DataFrame para ver las primeras entradas de la columna "by".
print("\nPrimeras cinco entradas de la columna 'by' de la tabla 'full':")
print(by_column_preview)
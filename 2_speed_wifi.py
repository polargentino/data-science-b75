# 1. Importar la biblioteca
import speedtest
# Propósito: Importa el módulo speedtest que permite interactuar 
# con servidores de prueba de velocidad (como Speedtest.net).

# Detalle técnico: Esta biblioteca maneja automáticamente:
# Búsqueda de servidores cercanos
# Protocolos HTTP para medir velocidad
# Cálculo de latencia (ping)

# 2. Crear una instancia de Speedtest
s = speedtest.Speedtest()
# Qué hace: Inicializa un objeto de la clase Speedtest.
# Funcionalidad interna:
# Configura parámetros iniciales (como timeout de conexión).
# Prepara sockets (enchufes) para mediciones de red.

# 3. Obtener lista de servidores
s.get_servers()
# Propósito: Descarga una lista actualizada de servidores de prueba disponibles.
# Detalle:
# Usa la API pública de Speedtest.net.
# Devuelve servidores ordenados por proximidad geográfica (basado en tu IP).

# 4. Seleccionar el mejor servidor
s.get_best_server()
# Lógica de selección:
# Prueba la latencia con varios servidores cercanos.
# Elige el que tenga menor ping.
# Establece conexión con ese servidor para las pruebas.

# 5. Medir velocidad de descarga
download_speed = s.download() / 1e6  # Convertir a Mbps
#Método:
# Descarga un archivo binario de tamaño variable.
# Mide el tiempo que tarda en completarse.
# Calcula: (tamaño_datos * 8) / tiempo → bits/segundo.
# Conversión: Dividir entre 1,000,000 (1e6) convierte bits/segundo a Mbps.

# 6. Medir velocidad de subida
upload_speed = s.upload() / 1e6  # Convertir a Mbps
# Similar a la descarga pero con archivos de prueba que se envían al servidor.
# Precisión: Usa múltiples hilos para saturar la conexión y obtener valores realistas.

# 7. Obtener ping
ping = s.results.ping
# Definición: Tiempo de ida y vuelta (en milisegundos) entre tu dispositivo y el servidor.
# Cálculo: Se mide durante la fase de selección de servidores con paquetes ICMP.

print(f"Velocidad de descarga: {download_speed:.2f} Mbps")
print(f"Velocidad de subida: {upload_speed:.2f} Mbps")
print(f"Ping: {ping:.2f} ms")
# Formato:
# f-string para interpolación de variables.
# :.2f redondea a 2 decimales.
# Unidades claras (Mbps y ms) para mejor interpretación.
# Salidas: 
# Velocidad de descarga: 9.32 Mbps
# Velocidad de subida: 1.33 Mbps
# Ping: 36.55 ms
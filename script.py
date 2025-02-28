"""
Este script verifica la instalación de librerías para web scraping y machine learning,
realizando pruebas básicas de cada una.
"""

# import sys

# Librerías para Web Scraping
try:
    import requests  # Para hacer peticiones HTTP
    print(f"[\u2713] requests v{requests.__version__}")
    response = requests.get(
        "https://httpbin.org/get", timeout=10)  # Realiza una solicitud GET de prueba
    print(f"  > Status code: {response.status_code}")
except ImportError:
    print("[X] requests no está instalado")

try:
    from bs4 import BeautifulSoup  # Para parsear HTML
    print("[\u2713] beautifulsoup4")
    soup = BeautifulSoup(
        "<html><body><h1>Hola Mundo</h1></body></html>", "html.parser")  # HTML de prueba
    print(f"  > Título encontrado: {soup.h1.text}")
except ImportError:
    print("[X] beautifulsoup4 no está instalado")

try:
    import scrapy  # Framework de Web Scraping
    print(f"[\u2713] scrapy v{scrapy.__version__}")
except ImportError:
    print("[X] scrapy no está instalado")

try:
    from selenium import webdriver  # Para automatizar navegadores
    print(f"[\u2713] selenium v{webdriver.__version__}")
except ImportError:
    print("[X] selenium no está instalado")

try:
    from playwright.sync_api import sync_playwright  # Otra opción para automatización web
    print("[\u2713] playwright")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)  # Inicia un navegador sin interfaz
        page = browser.new_page()
        page.goto("https://google.com")  # Carga Google
        print(f"  > Playwright cargó {page.title()}")
        browser.close()
except ImportError:
    print("[X] playwright no está instalado")

# Librerías para Análisis de Datos
try:
    import pandas as pd  # Manipulación de datos
    print(f"[\u2713] pandas v{pd.__version__}")
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})  # Crea un DataFrame de prueba
    print(f"  > DataFrame:\n{df}")
except ImportError:
    print("[X] pandas no está instalado")

try:
    import numpy as np  # Cálculo numérico
    print(f"[\u2713] numpy v{np.__version__}")
    arr = np.array([1, 2, 3])  # Crea un array de prueba
    print(f"  > Array: {arr}")
except ImportError:
    print("[X] numpy no está instalado")

try:
    import matplotlib  # Gráficos
    import matplotlib.pyplot as plt  # Módulo para trazar gráficos
    print(f"[\u2713] matplotlib v{matplotlib.__version__}")
    plt.plot([1, 2, 3], [4, 5, 6])  # Crea un gráfico de prueba
    plt.title("Prueba Matplotlib")
    plt.savefig("grafico.png")  # Guarda el gráfico en un archivo
    print("  > Gráfico guardado como 'grafico.png'")
except ImportError:
    print("[X] matplotlib no está instalado")

try:
    import sklearn  # Machine Learning
    print(f"[\u2713] scikit-learn v{sklearn.__version__}")
except ImportError:
    print("[X] scikit-learn no está instalado")

print("\n\u2705 Prueba finalizada")
# Salida:
# [✓] requests v2.32.3
#   > Status code: 200
# [✓] beautifulsoup4
#   > Título encontrado: Hola Mundo
# [✓] scrapy v2.12.0
# [✓] selenium v4.29.0
# [✓] playwright
#   > Playwright cargó Google
# [✓] pandas v2.2.3
#   > DataFrame:
#    A  B
# 0  1  4
# 1  2  5
# 2  3  6
# [✓] numpy v2.2.3
#   > Array: [1 2 3]
# [✓] matplotlib v3.10.0
#   > Gráfico guardado como 'grafico.png'
# [✓] scikit-learn v1.6.1

# ✅ Prueba finalizada
                       
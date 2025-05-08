import numpy as np
import matplotlib.pyplot as plt

# Generar una secuencia de valores de x de -1 a 1
x = np.arange(-1, 1, 0.0001)

# Implementación de la fórmula
y1 = np.sqrt(1 - x**2)
y2 = -np.sqrt(1 - x**2)

# Graficar el gráfico con las dos partes del círculo
plt.plot(x, y1, 'r')
plt.plot(x, y2, 'r' )

# Agregar el título del gráfico y las etiquetas de los ejes x e y
plt.title("Círculo")
plt.xlabel("Eje x")
plt.ylabel("Eje y")

# Mostrar el gráfico
plt.show()
import pygame
import random

# Inicialización de Pygame
pygame.init()

# Dimensiones de la ventana
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Movimiento con Pygame")

# Colores
white = (255, 255, 255)
red = (255, 0, 0)

# Posición y velocidad inicial del objeto
x = width // 2
y = height // 2
speed_x = 5
speed_y = 5

# Bucle principal del juego
running = True
while running:
    # Manejo de eventos
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                speed_x = -5
            elif event.key == pygame.K_RIGHT:
                speed_x = 5
            elif event.key == pygame.K_UP:
                speed_y = -5
            elif event.key == pygame.K_DOWN:
                speed_y = 5

    # Actualización de la posición del objeto
    x += speed_x
    y += speed_y

    # Mantener el objeto dentro de la ventana
    if x < 0 or x > width:
        speed_x = -speed_x
        x += speed_x # Evitar que quede pegado en el borde.
    if y < 0 or y > height:
        speed_y = -speed_y
        y += speed_y # Evitar que quede pegado en el borde.

    # Limpiar la pantalla
    screen.fill(white)

    # Dibujar un círculo
    pygame.draw.circle(screen, red, (x, y), 30)

    # Actualizar la pantalla
    pygame.display.flip()

    # Controlar la velocidad del juego
    pygame.time.Clock().tick(60)

# Salir de Pygame
pygame.quit()


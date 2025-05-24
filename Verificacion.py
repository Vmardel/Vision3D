import cv2 #opencv
import numpy as np #funciones matematicas
import os #trata de directorios

def verificar_rectificacion(img_izq, img_der, guardar_en="output/verificacion_rectificacion.png"):
    img_left = cv2.imread(img_izq)
    img_right = cv2.imread(img_der)

    # Redimensionar si no tienen la misma altura y combinarlas
    h = min(img_left.shape[0], img_right.shape[0])
    img_left = img_left[:h]
    img_right = img_right[:h]
    img_comb = np.hstack((img_left, img_right))

    # Dibujar líneas horizontales
    spacing = 40
    for y in range(0, h, spacing):
        cv2.line(img_comb, (0, y), (img_comb.shape[1], y), (0, 255, 0), 1)

    # Mostrar (si tienes GUI) y guardar
    os.makedirs(os.path.dirname(guardar_en), exist_ok=True)
    cv2.imwrite(guardar_en, img_comb)

'''
Dibuja líneas horizontales verdes para la verificación visual de que las imágenes están alineadas horizontalmente (rectificadas):
'''

if __name__ == "__main__":
    verificar_rectificacion("output/rect_hartley_izq.png", "output/rect_hartley_der.png")

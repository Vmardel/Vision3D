import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def cargar_imagen(path):
    """Carga una imagen en formato RGB y la devuelve como array NumPy."""
    img = Image.open(path).convert("RGB")
    return np.array(img)

def dibujar_lineas_horizontales(imagen, num_lineas=10):
    """
    Muestra la imagen con líneas horizontales equidistantes.
    Las líneas sirven para verificar visualmente que las correspondencias (bordes, 
    etc.) estén alineadas horizontalmente tras la rectificación.
    """
    h, w = imagen.shape[:2]
    plt.imshow(imagen)
    for i in range(1, num_lineas):
        y = int(i * h / num_lineas)
        plt.axhline(y=y, color='lime', linestyle='--', linewidth=1)
    plt.title("Imagen con Líneas Horizontales")
    plt.axis('off')

def superponer_imagenes(im_izq, im_der, alpha=0.5):
    """
    Superpone dos imágenes con transparencia y muestra el resultado.
    Si la rectificación es correcta, las estructuras importantes se verán solapadas verticalmente.
    """
    # Asegurarse de que ambas imágenes tengan el mismo tamaño
    im_izq = im_izq.astype(np.float32) / 255.0
    im_der = im_der.astype(np.float32) / 255.0
    superpuesta = alpha * im_izq + (1 - alpha) * im_der
    plt.imshow(superpuesta)
    plt.title("Superposición de Imágenes Rectificadas")
    plt.axis('off')

if __name__ == "__main__":
    # Carga de las imágenes rectificadas generadas previamente
    img_rect_izq = cargar_imagen("output/rect_calibrada_izq.png")
    img_rect_der = cargar_imagen("output/rect_calibrada_der.png")

    # 1. Visualización individual con líneas horizontales
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    dibujar_lineas_horizontales(img_rect_izq)
    
    plt.subplot(1, 2, 2)
    dibujar_lineas_horizontales(img_rect_der)
    
    plt.suptitle("Verificación Visual: Líneas Horizontales en Cada Imagen")
    plt.show()

    # 2. Superposición de las dos imágenes
    plt.figure(figsize=(6, 6))
    superponer_imagenes(img_rect_izq, img_rect_der, alpha=0.5)
    plt.show()

    print("Revise las ventanas gráficas: \n- En la primera verificación, las líneas horizontales deben coincidir en ambos lados (lo que indica que las epipolares son realmente horizontales).\n- En la superposición, las estructuras comunes deben solaparse verticalmente.")

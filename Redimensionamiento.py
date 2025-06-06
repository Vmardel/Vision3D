import os
import cv2
#hay que reescalarlo (mismo ratio)

# Carpeta de entrada
carpeta_entrada = "/mnt/c/Users/vicen/Documents/GitHub/Vision3D/Imagenes"
# Carpeta de salida
carpeta_salida = os.path.join(carpeta_entrada, "redimensionadas")

# Crear carpeta de salida si no existe
os.makedirs(carpeta_salida, exist_ok=True)

# Nombres específicos de las imágenes que queremos redimensionar
imagenes_deseadas = ["left.png", "left1.png", "right.png", "right1.png"]

# Dimensiones deseadas
nuevo_ancho = 450
nueva_altura = 375

# Procesar solo las imágenes seleccionadas
for nombre_archivo in imagenes_deseadas:
    ruta_entrada = os.path.join(carpeta_entrada, nombre_archivo)
    ruta_salida = os.path.join(carpeta_salida, nombre_archivo)

    # Leer imagen (con canal alfa si lo tiene)
    imagen = cv2.imread(ruta_entrada, cv2.IMREAD_UNCHANGED)

    if imagen is not None:
        # Redimensionar
        imagen_redimensionada = cv2.resize(imagen, (nuevo_ancho, nueva_altura))
        # Guardar como PNG
        cv2.imwrite(ruta_salida, imagen_redimensionada)
        print(f"Guardado: {ruta_salida}")
    else:
        print(f"No se pudo leer: {ruta_entrada}")

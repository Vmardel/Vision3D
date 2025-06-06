import cv2 #opencv
import numpy as np #funciones matematicas
import os #trata de directorios

def obtener_SIFT(img_izq, img_der, guardar_en="output"):
    # Cargar imagenes en escala de grises
    Il = cv2.imread(img_izq, cv2.IMREAD_GRAYSCALE)
    Ir = cv2.imread(img_der, cv2.IMREAD_GRAYSCALE)

    if Il is None or Ir is None:
        print("Error al cargar imagenes.")
        return

    # Inicializar SIFT
    sift = cv2.SIFT_create()

    # Detectar keypoints y descriptores
    kp1, des1 = sift.detectAndCompute(Il, None)
    kp2, des2 = sift.detectAndCompute(Ir, None)

    # Inicializar FLANN matcher
    params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
    search_time = dict(checks=50)
    flann = cv2.FlannBasedMatcher(params, search_time)

    matches = flann.knnMatch(des1, des2, k=2)

    # Ratio test de Lowe
    good_matches = []
    for m, n in matches:
        if m.distance < 0.51 * n.distance:
            good_matches.append(m)

    print(f" {len(good_matches)} correspondencias validas encontradas.")

    # Extraer coordenadas de puntos
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    os.makedirs(guardar_en, exist_ok=True)

    # Guardar puntos
    np.save(os.path.join(guardar_en, "pts1.npy"), pts1)
    np.save(os.path.join(guardar_en, "pts2.npy"), pts2)

    # Dibujar y guardar matches
    img_matches = cv2.drawMatches(Il, kp1, Ir, kp2, good_matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(os.path.join(guardar_en, "sift_matches.png"), img_matches)

    print(f"Puntos guardados en '{guardar_en}/pts1.npy' y 'pts2.npy'")
    print(f"Imagen de correspondencias guardada en '{guardar_en}/sift_matches.png'")

    '''
    Esta funcion utiliza SIFT para detectar puntos clave en dos imagenes y encontrar correspondencias entre ellos. 
    Primero, carga ambas imagenes en escala de grises para que la deteccion de caracteristicas sea mas efectiva. Luego, aplica SIFT para extraer los keypoints y 
    descriptores de cada imagen.

    Para emparejar estos descriptores, se usa el algoritmo FLANN, que busca similitudes entre los descriptores de ambas imagenes. Se seleccionan los dos mejores matches
    para cada punto y se aplica el Ratio Test de Lowe para filtrar los emparejamientos mas confiables.

    Despues de obtener las mejores correspondencias, se extraen las coordenadas de los puntos clave y se guardan en archivos .npy para ser utilizados en 
    calculos posteriores. Tambien se genera una imagen visualizando las correspondencias, lo que ayuda a verificar la calidad del emparejamiento.
    '''

if __name__ == "__main__":
    img_izq = "/mnt/c/Users/vicen/Documents/GitHub/Vision3D/Imagenes/redimensionadas/left1.png"
    img_der = "/mnt/c/Users/vicen/Documents/GitHub/Vision3D/Imagenes/redimensionadas/right1.png"

    obtener_SIFT(img_izq, img_der)


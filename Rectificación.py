import numpy as np #funciones matematicas
import cv2 #opencv
import os #trata de directorios

def descomponer_matriz_esencial(E):
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0: U *= -1
    if np.linalg.det(Vt) < 0: Vt *= -1

    W = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]
    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]

'''
Toma la matriz esencial E, que contiene la geometria entre dos camaras y la descompone para obtener 4 posibles combinaciones de rotacion y traslacion
(Estas representan cómo está colocada la segunda cámara respecto a la primera).
'''

# Triangular un punto 3D desde dos cámaras
def triangula_punto(P1, P2, x1, x2):
    A = np.array([
        x1[0] * P1[2] - P1[0],
        x1[1] * P1[2] - P1[1],
        x2[0] * P2[2] - P2[0],
        x2[1] * P2[2] - P2[1]
    ])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]

'''
Recibe un punto (x1) de la imagen 1 y otro (x2) de la imagen 2. Luego usa las matrices de proyeccion de "ambas cámaras" (la misma pero angulos distintos) para 
reconstruir ese punto en 3D y devolver su posicion en el espacio.
'''

# Comprobar si el punto está delante de la cámara
def punto_delante(P, X):
    X_h = np.append(X, 1)
    x_cam = P @ X_h
    return x_cam[2] > 0

'''
Verifica si el punto 3D que se ha triangulado está delante de la camara (y no detras), si está delante, significa que esa combinacion de R (rotacion), t (traslacion) 
es válida fisicamente.
'''

# Seleccionar (R, t) consistente con E y con el punto delante
def resolver_pose(E, y1, y2, K):
    poses = descomponer_matriz_esencial(E)
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

    for R, t in poses:
        P2 = K @ np.hstack((R, t.reshape(3, 1)))
        X = triangula_punto(P1, P2, y1, y2)
        if punto_delante(P1, X) and punto_delante(P2, X):
            return R, t
    raise ValueError("No se encontró ninguna configuración válida con el punto delante de ambas cámaras.")

'''
Toma los puntos normalizados (y1,y2) y la matriz esencial para probar las 4 combinaciones posibles de R y t. Luego triangula el punto en 3D con cada combinación y
devuelve la única que deja el punto delante de ambas cámaras.
'''

# Construir las homografías Hl y Hr
def construir_homografias(K, R, t):
    z = t / np.linalg.norm(t)
    x = np.cross([0, 1, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_rect = np.vstack((x, y, z)).T

    H_l = K @ R_rect @ np.linalg.inv(K)
    H_r = K @ R_rect @ R.T @ np.linalg.inv(K)
    return H_l, H_r

'''
Construye dos homografías (matrices de transformación) para rotar cada imagen con la idea de alinear los angulos virtualmente, como si ambos estuvieran mirando 
al frente para hacer que los puntos correspondientes estén en la misma altura en ambas imágenes
'''

# Aplicar homografías a imágenes
def aplicar_rectificacion(img_izq, img_der, Hl, Hr, guardar_en="output"):
    img_left = cv2.imread(img_izq)
    img_right = cv2.imread(img_der)

    h, w = img_left.shape[:2]

    img_left_rect = cv2.warpPerspective(img_left, Hl, (w, h))
    img_right_rect = cv2.warpPerspective(img_right, Hr, (w, h))

    os.makedirs(guardar_en, exist_ok=True)
    cv2.imwrite(os.path.join(guardar_en, "rect_izq.png"), img_left_rect)
    cv2.imwrite(os.path.join(guardar_en, "rect_der.png"), img_right_rect)
    print(f"\nRectificación completada. Imágenes guardadas en '{guardar_en}/rect_izq.png' y 'rect_der.png'.")

'''
Carga las dos imágenes originales (img_izq, img_der) y aplica las homografías calculadas con cv2.warpPerspective(), luego guarda las imágenes 
rectificadas, donde las líneas epipolares ya son horizontales.
'''

if __name__ == "__main__":
    # Cargar entradas previas
    E = np.load("output/E.npy")
    K = np.load("output/K_P.npy")
    pts1 = np.load("output/pts1.npy")
    pts2 = np.load("output/pts2.npy")
    inliers = np.load("output/inliers.npy")

    # Elegir una correspondencia válida
    y1 = np.linalg.inv(K) @ np.append(pts1[inliers[0]], 1)
    y2 = np.linalg.inv(K) @ np.append(pts2[inliers[0]], 1)

    # Calcular R, t consistentes con E y el punto delante
    R, t = resolver_pose(E, y1, y2, K)

    # Calcular homografías de rectificación
    Hl, Hr = construir_homografias(K, R, t)

    # Mostrar resultados
    print("\nRotación R:\n", R)
    print("\nTraslación t:\n", t)
    print("\nHomografía izquierda Hl:\n", Hl)
    print("\nHomografía derecha Hr:\n", Hr)

    # Guardar resultados
    np.save("output/R_rectified.npy", R)
    np.save("output/t_rectified.npy", t)
    np.save("output/Hl.npy", Hl)
    np.save("output/Hr.npy", Hr)

    # Aplicar homografías a imágenes reales
    img_izq = "/mnt/c/Users/vicen/Documents/GitHub/Vision3D/Imagenes/Image15.png"
    img_der = "/mnt/c/Users/vicen/Documents/GitHub/Vision3D/Imagenes/Image16.png"
    aplicar_rectificacion(img_izq, img_der, Hl, Hr)


































'''

#otra manera de hacer el mismo codigo, esta es una prueba, funciona diferente y no nos gusta el resultado

import numpy as np
import cv2
import os

def descomponer_matriz_esencial(E):
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0: U *= -1
    if np.linalg.det(Vt) < 0: Vt *= -1

    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]
    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]

def triangula_punto(P1, P2, x1, x2):
    A = np.array([
        x1[0] * P1[2] - P1[0],
        x1[1] * P1[2] - P1[1],
        x2[0] * P2[2] - P2[0],
        x2[1] * P2[2] - P2[1]
    ])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]

def punto_esta_delante(P, X):
    X_h = np.append(X, 1)
    x_cam = P @ X_h
    return x_cam[2] > 0

def resolver_pose(E, y1, y2, K):
    poses = descomponer_matriz_esencial(E)
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

    for R, t in poses:
        P2 = K @ np.hstack((R, t.reshape(3, 1)))
        X = triangula_punto(P1, P2, y1, y2)
        if punto_esta_delante(P1, X) and punto_esta_delante(P2, X):
            return R, t
    raise ValueError("No se encontró ninguna configuración válida con el punto delante de ambas cámaras.")

def construir_homografias_rectificacion(K, R, t):
    z = t / np.linalg.norm(t)
    x = np.cross([0, 1, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_rect = np.vstack((x, y, z)).T

    H_l = K @ R_rect @ np.linalg.inv(K)
    H_r = K @ R_rect @ R.T @ np.linalg.inv(K)
    return H_l, H_r

def aplicar_rectificacion_mejorado(im_izq_path, im_der_path, Hl, Hr, guardar_en="output"):
    img_l = cv2.imread(im_izq_path)
    img_r = cv2.imread(im_der_path)
    h, w = img_l.shape[:2]

    img_l_rect = cv2.warpPerspective(img_l, Hl, (w, h))
    img_r_rect = cv2.warpPerspective(img_r, Hr, (w, h))

    def recortar_imagen(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        x, y, w, h = cv2.boundingRect(coords)
        return img[y:y+h, x:x+w]

    img_l_rec = recortar_imagen(img_l_rect)
    img_r_rec = recortar_imagen(img_r_rect)

    os.makedirs(guardar_en, exist_ok=True)
    cv2.imwrite(os.path.join(guardar_en, "rect_izq.png"), img_l_rec)
    cv2.imwrite(os.path.join(guardar_en, "rect_der.png"), img_r_rec)

    print("Imágenes rectificadas y recortadas guardadas:")
    print(f" - {guardar_en}/rect_izq.png")
    print(f" - {guardar_en}/rect_der.png")

# ----------- MAIN ------------
if __name__ == "__main__":
    # Cargar datos
    E = np.load("output/E.npy")
    K = np.load("output/K_P.npy")
    pts1 = np.load("output/pts1.npy")
    pts2 = np.load("output/pts2.npy")
    inliers = np.load("output/inliers.npy")

    # Seleccionar un punto válido
    y1 = np.linalg.inv(K) @ np.append(pts1[inliers[0]], 1)
    y2 = np.linalg.inv(K) @ np.append(pts2[inliers[0]], 1)

    # Calcular (R, t) correctos
    R, t = resolver_pose(E, y1, y2, K)

    # Homografías
    Hl, Hr = construir_homografias_rectificacion(K, R, t)

    # Mostrar
    print("\nRotación R:\n", R)
    print("\nTraslación t:\n", t)
    print("\nHomografía izquierda Hl:\n", Hl)
    print("\nHomografía derecha Hr:\n", Hr)

    # Guardar matrices
    np.save("output/R_rectified.npy", R)
    np.save("output/t_rectified.npy", t)
    np.save("output/Hl.npy", Hl)
    np.save("output/Hr.npy", Hr)

    # Aplicar y recortar imágenes rectificadas
    aplicar_rectificacion_mejorado("Imagenes/Image15.png", "Imagenes/Image16.png", Hl, Hr)
'''

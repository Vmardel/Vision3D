import numpy as np #funciones matematicas
import os #trata de archivos
from PIL import Image, ImageOps #procesamiento de imagenes
from skimage.transform import ProjectiveTransform, warp #transformaicon de imagenes


def cargar_imagen(path):
    img = Image.open(path).convert("RGB")
    img = ImageOps.exif_transpose(img)  # Corrige orientación según EXIF (ya no hace nada relevante por el cambio de imagenes)
    return np.array(img)


def guardar_imagen(path, imagen):
    Image.fromarray((imagen * 255).astype(np.uint8)).save(path)


def aplicar_homografia(imagen, H, shape_out):
    H_inv = np.linalg.inv(H)
    transform = ProjectiveTransform(H_inv)
    return warp(imagen, transform, output_shape=shape_out)

'''
Esta funcion se encarga de aplicar las homografias calculadas a las imagenes para rectificarlas
'''

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

'''
Estafuncion se encarga de cargar la matriz E calculada en Fase4 y codificar la geometria de las camara con cuatro configurariones posibes de matrizes de Rotacion
y traslacion.
'''

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


def punto_delante(P, X):
    X_h = np.append(X, 1)
    return (P @ X_h)[2] > 0

'''
Estas dos funciones tratan con los puntos 2D y 3D de las imagenes, la primero calcula el punto 3D que corresponde a los dos puntos 2D de las dos imagenes en las 
dos fotos mientras que la sengunda se encarga de confirmar que este punto calculado se encuentre delande de la camara y no fuera de esta.
'''


def encontrar_mejor_pose(E, pts1, pts2, inliers, K, n_muestras=20):
    muestras = inliers[:min(n_muestras, len(inliers))]
    soluciones = descomponer_matriz_esencial(E)
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    mejor_R, mejor_t, max_validos = None, None, 0

    for R, t in soluciones:
        P2 = K @ np.hstack((R, t.reshape(3, 1)))
        num_validos = 0
        for i in muestras:
            y1 = np.linalg.inv(K) @ np.append(pts1[i], 1)
            y2 = np.linalg.inv(K) @ np.append(pts2[i], 1)
            X = triangula_punto(P1, P2, y1, y2)
            if punto_delante(P1, X) and punto_delante(P2, X):
                num_validos += 1

        if num_validos > max_validos:
            mejor_R, mejor_t, max_validos = R, t, num_validos

    if mejor_R is None:
        raise ValueError("No se encontró ninguna configuración válida con múltiples puntos.")
    
    print(f"\nConfiguración seleccionada con {max_validos}/{len(muestras)} puntos válidos.")
    return mejor_R, mejor_t

'''
Esta funcion se encargar de analizar las cuatro posibles combinaciones de matrices de la funcion descomponer_matriz_esencial para encontrar la mejor combinacion
de estas que hagan que hallan mas puntos visibles de manera correcta mediante una triangulacion.
'''

def construir_homografias(K, R, t):
    z = t / np.linalg.norm(t)
    x = np.cross([0, 1, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_rect = np.vstack((x, y, z)).T
    Hl = K @ R_rect @ np.linalg.inv(K)
    Hr = K @ R_rect @ R.T @ np.linalg.inv(K)
    return Hl, Hr

'''
Aqui simplemente creamos las homografias que usaremos en el codigo.
'''

def rectificacion_hartley(img_izq, img_der, F, pts1, pts2, out_dir="data"):
    from skimage.transform import estimate_transform

    img1 = cargar_imagen(img_izq)
    img2 = cargar_imagen(img_der)
    h, w = img1.shape[:2]

    tform1 = estimate_transform('projective', pts1, pts2)
    tform2 = estimate_transform('projective', pts2, pts1)

    rect1 = warp(img1, tform1.inverse, output_shape=(h, w))
    rect2 = warp(img2, tform2.inverse, output_shape=(h, w))

    os.makedirs(out_dir, exist_ok=True)
    guardar_imagen(os.path.join(out_dir, "rect_hartley_izq.png"), rect1)
    guardar_imagen(os.path.join(out_dir, "rect_hartley_der.png"), rect2)
    print("\nRectificación completada.")

'''
En esta ultima funcion realizamos la rectificacion de las imagenes utilizando la matriz F (fundamental) y las correspondencias calculadas anteriormente.
'''


if __name__ == "__main__":
    E = np.load("/mnt/c/Users/vicen/Documents/GitHub/Vision3D/Etapa4/data/E.npy")
    K = np.load("/mnt/c/Users/vicen/Documents/GitHub/Vision3D/Etapa2/data/K_intrinseca.npy")
    pts1 = np.load("/mnt/c/Users/vicen/Documents/GitHub/Vision3D/Etapa3/data/pts1.npy")
    pts2 = np.load("/mnt/c/Users/vicen/Documents/GitHub/Vision3D/Etapa3/data/pts2.npy")
    inliers = np.load("/mnt/c/Users/vicen/Documents/GitHub/Vision3D/Etapa3/data/inliers.npy")

    img_izq = "/mnt/c/Users/vicen/Documents/GitHub/Vision3D/Etapa3/Imagenes/left1.png"
    img_der = "/mnt/c/Users/vicen/Documents/GitHub/Vision3D/Etapa3/Imagenes/right1.png"

    R, t = encontrar_mejor_pose(E, pts1, pts2, inliers, K)
    Hl, Hr = construir_homografias(K, R, t)

    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)

    np.save("data/R_rectified.npy", R)
    np.save("data/t_rectified.npy", t)
    np.save("data/Hl.npy", Hl)
    np.save("data/Hr.npy", Hr)

    F = np.load("/mnt/c/Users/vicen/Documents/GitHub/Vision3D/Etapa3/data/F.npy")
    rectificacion_hartley(img_izq, img_der, F, pts1[inliers], pts2[inliers])
import numpy as np
import os
from PIL import Image, ImageOps
from skimage.transform import ProjectiveTransform, warp


def cargar_imagen(path):
    img = Image.open(path).convert("RGB")
    img = ImageOps.exif_transpose(img)  # Corrige orientación según EXIF
    return np.array(img)


def guardar_imagen(path, imagen):
    Image.fromarray((imagen * 255).astype(np.uint8)).save(path)


def aplicar_homografia(imagen, H, shape_out):
    H_inv = np.linalg.inv(H)
    transform = ProjectiveTransform(H_inv)
    return warp(imagen, transform, output_shape=shape_out)


def epipolo(F):
    # Epipolo derecho: vector nulo de F.T
    _, _, Vt = np.linalg.svd(F.T)
    e = Vt[-1]
    return e / e[2]


def construir_rectificacion_no_calibrada(F, pts1, pts2, img_shape, p=None):
    h, w = img_shape[:2]
    e = epipolo(F)
    
    if p is None:
        p = np.array([w / 2, h / 2, 1.0])  # centro de la imagen

    # Traslación para que el punto p se mueva al origen
    T = np.array([
        [1, 0, -p[0]],
        [0, 1, -p[1]],
        [0, 0,    1]
    ])

    # Rotación para que el epipolo apunte en la dirección x
    e = T @ e
    norm = np.sqrt(e[0]**2 + e[1]**2)
    alpha = e[0] / norm
    beta = e[1] / norm
    R = np.array([
        [ alpha,  beta, 0],
        [-beta,  alpha, 0],
        [    0,     0, 1]
    ])

    # Proyección a infinito
    G = np.eye(3)
    G[2, 0] = -1.0 / e[0]

    H2 = np.linalg.inv(T) @ np.linalg.inv(R) @ G @ R @ T

    # Para H1 buscamos una H1 tal que H1*x1 esté alineado con H2*x2
    # Aquí usamos un método simplificado basado en transferencia lineal
    pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
    pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])
    H1 = np.eye(3)

    # Ajustamos H1 para minimizar la diferencia en Y
    pts1_new = (H1 @ pts1_h.T).T
    pts2_new = (H2 @ pts2_h.T).T
    pts1_new /= pts1_new[:, 2:]
    pts2_new /= pts2_new[:, 2:]

    dy = np.mean(pts2_new[:, 1] - pts1_new[:, 1])
    H1[1, 2] = dy

    return H1, H2


def rectificacion_hartley_algoritmo(F, pts1, pts2, img_izq_path, img_der_path, out_dir="output"):
    img1 = cargar_imagen(img_izq_path)
    img2 = cargar_imagen(img_der_path)
    H1, H2 = construir_rectificacion_no_calibrada(F, pts1, pts2, img1.shape)

    rect1 = aplicar_homografia(img1, H1, img1.shape[:2])
    rect2 = aplicar_homografia(img2, H2, img2.shape[:2])

    os.makedirs(out_dir, exist_ok=True)
    guardar_imagen(os.path.join(out_dir, "rect_alg20_izq.png"), rect1)
    guardar_imagen(os.path.join(out_dir, "rect_alg20_der.png"), rect2)
    print("\nRectificación NO CALIBRADA (Algoritmo 20.1) completada.")


if __name__ == "__main__":
    F = np.load("output/F.npy")
    pts1 = np.load("output/pts1.npy")
    pts2 = np.load("output/pts2.npy")
    inliers = np.load("output/inliers.npy")

    img_izq = "Imagenes/image15.png"
    img_der = "Imagenes/image16.png"

    rectificacion_hartley_algoritmo(F, pts1[inliers], pts2[inliers], img_izq, img_der)
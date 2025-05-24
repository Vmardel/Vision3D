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


def punto_delante(P, X):
    X_h = np.append(X, 1)
    return (P @ X_h)[2] > 0


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


def construir_homografias(K, R, t):
    z = t / np.linalg.norm(t)
    x = np.cross([0, 1, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_rect = np.vstack((x, y, z)).T
    Hl = K @ R_rect @ np.linalg.inv(K)
    Hr = K @ R_rect @ R.T @ np.linalg.inv(K)
    return Hl, Hr


def rectificacion(img_izq_path, img_der_path, Hl, Hr, out_dir="output"):
    img1 = cargar_imagen(img_izq_path)
    img2 = cargar_imagen(img_der_path)
    h, w = img1.shape[:2]
    rect1 = aplicar_homografia(img1, Hl, (h, w))
    rect2 = aplicar_homografia(img2, Hr, (h, w))
    os.makedirs(out_dir, exist_ok=True)
    guardar_imagen(os.path.join(out_dir, "rect_calibrada_izq.png"), rect1)
    guardar_imagen(os.path.join(out_dir, "rect_calibrada_der.png"), rect2)
    print("\nRectificación CALIBRADA completada.")


def rectificacion_hartley(img_izq, img_der, F, pts1, pts2, out_dir="output"):
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
    print("\nRectificación NO CALIBRADA (Hartley) completada.")


if __name__ == "__main__":
    E = np.load("output/E.npy")
    K = np.load("output/K_P.npy")
    pts1 = np.load("output/pts1.npy")
    pts2 = np.load("output/pts2.npy")
    inliers = np.load("output/inliers.npy")

    img_izq = "Imagenes/image15.png"
    img_der = "Imagenes/image16.png"

    print("\n--- Rectificación Calibrada ---")
    R, t = encontrar_mejor_pose(E, pts1, pts2, inliers, K)
    Hl, Hr = construir_homografias(K, R, t)

    np.save("output/R_rectified.npy", R)
    np.save("output/t_rectified.npy", t)
    np.save("output/Hl.npy", Hl)
    np.save("output/Hr.npy", Hr)

    rectificacion(img_izq, img_der, Hl, Hr)

    print("\n--- Rectificación No Calibrada (Hartley) ---")
    F = np.load("output/F.npy")
    rectificacion_hartley(img_izq, img_der, F, pts1[inliers], pts2[inliers])

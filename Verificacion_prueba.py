import cv2
import numpy as np
import os

def verificar_rectificacion(img_izq, img_der, pts1, pts2, H1, H2, guardar_en="output/verificacion_rectificacion.png"):
    # Cargar imágenes
    img_left = cv2.imread(img_izq)
    img_right = cv2.imread(img_der)

    if img_left is None or img_right is None:
        print("Error: No se pudieron cargar las imágenes.")
        return

    # Aplicar homografías a los puntos
    pts1_h = cv2.perspectiveTransform(pts1.reshape(-1, 1, 2).astype(np.float32), H1)
    pts2_h = cv2.perspectiveTransform(pts2.reshape(-1, 1, 2).astype(np.float32), H2)

    # Redimensionar si alturas no coinciden
    h = min(img_left.shape[0], img_right.shape[0])
    img_left = img_left[:h]
    img_right = img_right[:h]
    img_comb = np.hstack((img_left, img_right))

    # Dibujar líneas horizontales verdes como referencia
    spacing = 40
    for y in range(0, h, spacing):
        cv2.line(img_comb, (0, y), (img_comb.shape[1], y), (0, 255, 0), 1)

    # Dibujar puntos y líneas de correspondencia
    offset_x = img_left.shape[1]
    for (p1, p2) in zip(pts1_h, pts2_h):
        x1, y1 = int(p1[0][0]), int(p1[0][1])
        x2, y2 = int(p2[0][0]) + offset_x, int(p2[0][1])

        cv2.circle(img_comb, (x1, y1), 4, (0, 0, 255), -1)
        cv2.circle(img_comb, (x2, y2), 4, (0, 0, 255), -1)
        cv2.line(img_comb, (x1, y1), (x2, y2), (255, 0, 0), 1)

    # Guardar imagen
    os.makedirs(os.path.dirname(guardar_en), exist_ok=True)
    cv2.imwrite(guardar_en, img_comb)
    print(f"Verificación guardada en: {guardar_en}")


if __name__ == "__main__":
    # Cargar imágenes y datos necesarios
    img_izq = "output/rect_alg20_izq.png"
    img_der = "output/rect_alg20_der.png"

    pts1 = np.load("output/pts1.npy")
    pts2 = np.load("output/pts2.npy")
    inliers = np.load("output/inliers.npy")

    Hl = np.load("output/Hl.npy")
    Hr = np.load("output/Hr.npy")

    verificar_rectificacion(
        img_izq, img_der,
        pts1[inliers], pts2[inliers],
        Hl, Hr
    )

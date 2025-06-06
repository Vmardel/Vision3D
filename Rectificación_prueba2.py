import numpy as np
import os
from PIL import Image, ImageOps
from skimage.transform import ProjectiveTransform, warp, estimate_transform


def cargar_imagen(path):
    img = Image.open(path).convert("RGB")
    img = ImageOps.exif_transpose(img)  # Corrige orientación EXIF
    return np.array(img) / 255.0  # Normalizado para skimage


def guardar_imagen(path, imagen):
    imagen = (imagen * 255).astype(np.uint8)
    Image.fromarray(imagen).save(path)


def verificar_rectificacion_visual(img1, img2, path_salida, spacing=40):
    h = min(img1.shape[0], img2.shape[0])
    img1 = (img1 * 255).astype(np.uint8)[:h]
    img2 = (img2 * 255).astype(np.uint8)[:h]
    img_comb = np.hstack((img1, img2))

    for y in range(0, h, spacing):
        img_comb[y:y+1, :, :] = [0, 255, 0]  # Línea verde

    os.makedirs(os.path.dirname(path_salida), exist_ok=True)
    Image.fromarray(img_comb).save(path_salida)
    print(f"Verificación visual guardada en: {path_salida}")


def rectificacion_hartley_simetrica(img_izq_path, img_der_path, pts1, pts2, out_dir="output", debug=True):
    print("Cargando imágenes...")
    img1 = cargar_imagen(img_izq_path)
    img2 = cargar_imagen(img_der_path)

    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    shape_out = (h, w)

    print("Estimando homografía simétrica para ambos...")
    tform = estimate_transform('projective', pts1, pts2)

    print("Aplicando homografías simétricas...")
    rect1 = warp(img1, tform.inverse, output_shape=shape_out)
    rect2 = warp(img2, tform, output_shape=shape_out)

    os.makedirs(out_dir, exist_ok=True)
    path1 = os.path.join(out_dir, "rect_prueba_izq.png")
    path2 = os.path.join(out_dir, "rect_prueba_der.png")
    guardar_imagen(path1, rect1)
    guardar_imagen(path2, rect2)
    print("Rectificación Hartley simétrica completada.")
    print(f"Guardadas:\n - {path1}\n - {path2}")

    if debug:
        verificar_rectificacion_visual(rect1, rect2, os.path.join(out_dir, "verificacion_prueba.png"))


if __name__ == "__main__":
    print("Iniciando rectificación Hartley SIMÉTRICA...")

    # Archivos requeridos
    img_izq = "Imagenes/redimensionadas/left1.png"
    img_der = "Imagenes/redimensionadas/right1.png"
    pts1 = np.load("output/pts1.npy")
    pts2 = np.load("output/pts2.npy")
    inliers = np.load("output/inliers.npy")

    rectificacion_hartley_simetrica(
        img_izq_path=img_izq,
        img_der_path=img_der,
        pts1=pts1[inliers],
        pts2=pts2[inliers],
        out_dir="output",
        debug=True
    )

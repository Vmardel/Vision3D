import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import open3d as o3d
import os
import cv2 as cv


def cargar_imagenes(ruta_izq, ruta_der):
    img_izq = Image.open(ruta_izq)
    img_der = Image.open(ruta_der)
    img_izq = ImageOps.exif_transpose(img_izq).convert("RGB")
    img_der = ImageOps.exif_transpose(img_der).convert("RGB")
    return img_izq, img_der


def resize_imagenes(img_izq, img_der, escala=1.0):
    dim_nueva = (int(img_izq.width * escala), int(img_izq.height * escala))
    img_izq = img_izq.resize(dim_nueva, Image.BILINEAR)
    img_der = img_der.resize(dim_nueva, Image.BILINEAR)
    return img_izq, img_der


def convertir_a_gris(imagen):
    return np.array(imagen.convert("L"))


def ecualizar_histograma(imagen_gris):
    return cv.equalizeHist(imagen_gris.astype(np.uint8))


def extraer_ventana(imagen, fila, col, radio, desplazamiento=0):
    return imagen[fila - radio:fila + radio + 1,
                  col - radio - desplazamiento: col + radio + 1 - desplazamiento]


def varianza_ventana(ventana):
    return np.var(ventana)


def interpolacion_parabolica(mejor_disp, errores):
    e = np.array(errores, dtype=np.float64)
    if 0 < mejor_disp < len(errores) - 1:
        e_antes = e[mejor_disp - 1]
        e_actual = e[mejor_disp]
        e_despues = e[mejor_disp + 1]
        denom = e_antes + e_despues - 2 * e_actual
        if denom != 0:
            ajuste = (e_antes - e_despues) / (2 * denom)
            if -1 <= ajuste <= 1:
                return ajuste
    return 0.0


def calcular_disparidad(imagen_izq, imagen_der, tamaño_ventana, disparidad_max, usar_subpixeles):
    alto, ancho = imagen_izq.shape
    radio = tamaño_ventana // 2
    escala_disp = 255 / disparidad_max
    mapa_disparidad = np.zeros_like(imagen_izq, dtype=np.float32)

    for fila in range(radio, alto - radio):
        for col in range(disparidad_max + radio, ancho - radio):
            ventana_izq = extraer_ventana(imagen_izq, fila, col, radio)
            if varianza_ventana(ventana_izq) < 15:
                continue
            ventana_izq = ventana_izq - ventana_izq.mean()

            errores = []
            mejor_error = np.inf
            mejor_disp = 0

            for d in range(disparidad_max):
                ventana_der = extraer_ventana(imagen_der, fila, col, radio, d)
                if ventana_der.shape != ventana_izq.shape:
                    errores.append(np.inf)
                    continue
                ventana_der = ventana_der - ventana_der.mean()
                error = np.sum((ventana_izq - ventana_der) ** 2)
                errores.append(error)

                if error < mejor_error:
                    mejor_error = error
                    mejor_disp = d

            if usar_subpixeles:
                mejor_disp += interpolacion_parabolica(mejor_disp, errores)

            mapa_disparidad[fila, col] = mejor_disp * escala_disp

    return mapa_disparidad


def guardar_mapa_disparidad(mapa, ruta_salida):
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    normalizado = 255 * (mapa - np.min(mapa)) / (np.max(mapa) - np.min(mapa))
    imagen_disp = Image.fromarray(normalizado.astype(np.uint8))
    imagen_disp.save(ruta_salida)
    print(f"Mapa de disparidad guardado en: {ruta_salida}")


def generar_nube_puntos(mapa_disp, imagen_rgb, ruta_nube):
    Q = np.array([[1, 0, 0, -mapa_disp.shape[1] / 2],
                  [0, -1, 0, mapa_disp.shape[0] / 2],
                  [0, 0, 0, -0.8 * mapa_disp.shape[1]],
                  [0, 0, 1 / 0.05, 0]])

    puntos_3d = cv.reprojectImageTo3D(mapa_disp, Q)
    mascara = mapa_disp > 0

    puntos = puntos_3d[mascara]
    colores = imagen_rgb[mascara].astype(np.float32) / 255.0

    nube = o3d.geometry.PointCloud()
    nube.points = o3d.utility.Vector3dVector(puntos)
    nube.colors = o3d.utility.Vector3dVector(colores)

    o3d.io.write_point_cloud(ruta_nube, nube)
    print(f"Nube de puntos exportada en: {ruta_nube}")
    o3d.visualization.draw_geometries([nube])


def main():
    ruta_izq = "output/rect_hartley_izq.png"
    ruta_der = "output/rect_hartley_der.png"

    img_izq, img_der = cargar_imagenes(ruta_izq, ruta_der)
    img_izq, img_der = resize_imagenes(img_izq, img_der, escala=1.0)

    gris_izq = convertir_a_gris(img_izq)
    gris_der = convertir_a_gris(img_der)

    gris_izq = ecualizar_histograma(gris_izq)
    gris_der = ecualizar_histograma(gris_der)

    mapa = calcular_disparidad(gris_izq, gris_der,
                              tamaño_ventana=11,
                              disparidad_max=128,
                              usar_subpixeles=True)

    guardar_mapa_disparidad(mapa, "output/gt_estimado2.png")

    plt.imshow(mapa, cmap="plasma")
    plt.colorbar()
    plt.title("Disparidad BM")
    plt.show()

    generar_nube_puntos(mapa, np.array(img_izq), "output/nube.ply")


if __name__ == "__main__":
    main()

import numpy as np
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt
import cv2 as cv
import os
from PIL import Image, ImageOps

# ---------------------------
# Block Matching Class (BM)
# ---------------------------

class BM:
    def __init__(self, kernel_size, max_disparity, subpixel_interpolation):
        self.kernel_size = kernel_size
        self.max_disparity = max_disparity
        self.kernel_half = kernel_size // 2
        self.offset_adjust = 255 / max_disparity
        self.subpixel_interpolation = subpixel_interpolation

        '''
        param ini
        '''

    def _get_window(self, y, x, img, offset=0):
        y_start = y - self.kernel_half
        y_end = y + self.kernel_half + 1
        x_start = x - self.kernel_half - offset
        x_end = x + self.kernel_half + 1 - offset
        return img[y_start:y_end, x_start:x_end]



    
    '''
    centramos la imagen para ejecutar algoritomo
    '''

    def _compute_subpixel_offset(self, best_offset, errors):
        errors = np.array(errors, dtype=np.float64)
        if (
            0 < best_offset < self.max_disparity - 1 and
            np.isfinite(errors[best_offset]) and
            np.isfinite(errors[best_offset - 1]) and
            np.isfinite(errors[best_offset + 1])
        ):
            denom = errors[best_offset - 1] + errors[best_offset + 1] - 2 * errors[best_offset]
            if denom != 0:
                numerator = errors[best_offset - 1] - errors[best_offset + 1]
                subpixel = numerator / (2 * denom)
                if -1.0 <= subpixel <= 1.0:
                    return subpixel
        return 0.0
    
    '''
    reajustamos los pixeles para los errores
    '''

    def compute(self, left, right):
        h, w = left.shape
        disp_map = np.zeros_like(left, dtype=np.float32)

        for y in range(self.kernel_half, h - self.kernel_half):
            for x in range(self.max_disparity + self.kernel_half, w - self.kernel_half):
                best_offset = 0
                min_error = float("inf")
                errors = []

                W_left = self._get_window(y, x, left)

            # Filtro por textura (varianza mínima)
                if np.var(W_left) < 15:
                    continue

                W_left_mean = W_left - np.mean(W_left)

                for offset in range(self.max_disparity):
                    W_right = self._get_window(y, x, right, offset)

                    if W_left.shape != W_right.shape:
                        errors.append(np.inf)
                        continue

                    W_right_mean = W_right - np.mean(W_right)

                # ZSSD: Zero-mean Sum of Squared Differences
                    error = np.sum((W_left_mean - W_right_mean) ** 2)
                    errors.append(error)

                    if error < min_error:
                        min_error = error
                        best_offset = offset

                if self.subpixel_interpolation:
                    best_offset += self._compute_subpixel_offset(best_offset, errors)

                disp_map[y, x] = best_offset * self.offset_adjust

        return disp_map

    
    '''
    nucleo algoritom bm y crea mapa disparidad
    '''

# ---------------------------
# Utilidades
# ---------------------------

def cargar_imagenes(ruta_izq, ruta_der):
    left_img = Image.open(ruta_izq)
    right_img = Image.open(ruta_der)

    # Corregir orientación según EXIF
    left_img = ImageOps.exif_transpose(left_img).convert("RGB")
    right_img = ImageOps.exif_transpose(right_img).convert("RGB")

    return left_img, right_img

def guardar_disparidad(disparidad, nombre_archivo):
    os.makedirs("output", exist_ok=True)
    disp_norm = 255 * (disparidad - disparidad.min()) / (disparidad.max() - disparidad.min())
    disp_img = Image.fromarray(disp_norm.astype(np.uint8))
    disp_img.save(nombre_archivo)
    print(f"Disparidad guardada como {nombre_archivo}")

#modifcar q con los valores de k pag 128 libro fusilleo
def crear_nube_puntos(disparidad, imagen_color, ruta_guardado):
    Q = np.array([[1, 0, 0, -disparidad.shape[1] / 2],
                  [0, -1, 0, disparidad.shape[0] / 2],
                  [0, 0, 0, -0.8 * disparidad.shape[1]],
                  [0, 0, 1 / 0.05, 0]])

    puntos_3d = cv.reprojectImageTo3D(disparidad, Q)
    mask = disparidad > 0

    puntos = puntos_3d[mask]
    colores = imagen_color[mask].astype(np.float64) / 255.0

    nube = o3d.geometry.PointCloud()
    nube.points = o3d.utility.Vector3dVector(puntos)
    nube.colors = o3d.utility.Vector3dVector(colores)

    # Guardar la nube en .ply
    o3d.io.write_point_cloud(ruta_guardado, nube)
    print(f"Nube de puntos guardada en {ruta_guardado}")

    # Mostrar la nube
    o3d.visualization.draw_geometries([nube])

# ---------------------------
# Main
# ---------------------------

#quiere ver la correspondencia minima en una grafica (correlacion max)

def main():
    ruta_left = "output/rect_hartley_izq.png"
    ruta_right = "output/rect_hartley_der.png"

    left_img, right_img = cargar_imagenes(ruta_left, ruta_right)

    # Reducción de tamaño para pruebas (mitad de ancho y alto)
    scale_factor = 1 #0.25
    new_size = (int(left_img.width * scale_factor), int(left_img.height * scale_factor))
    left_img = left_img.resize(new_size, Image.BILINEAR)
    right_img = right_img.resize(new_size, Image.BILINEAR)

    left_gray = np.array(left_img.convert("L"))
    right_gray = np.array(right_img.convert("L"))

    left_gray = cv.equalizeHist(left_gray.astype(np.uint8))
    right_gray = cv.equalizeHist(right_gray.astype(np.uint8))


    bm = BM(kernel_size=11, max_disparity=128, subpixel_interpolation=True)
    disparidad = bm.compute(left_gray, right_gray)

    guardar_disparidad(disparidad, "output/gt_estimado.png")

    plt.imshow(disparidad, cmap="plasma")
    plt.title("Mapa de Disparidad Estimado")
    plt.colorbar()
    plt.show()

    crear_nube_puntos(disparidad, np.array(left_img), "output/nube.ply")


if __name__ == "__main__":
    main()

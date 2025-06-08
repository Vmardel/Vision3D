import numpy as np #funciones matematicas
from PIL import Image #procesamiento de imagenes
import open3d as o3d #trabajo con datos 3D
from scipy.ndimage import gaussian_filter, generic_filter #procesamiento de imagenes

def obtener_ventana(imagen, fila, col, tam_ventana, desplaz=0):
    radio = tam_ventana // 2
    fila_inicio = fila - radio
    fila_fin = fila + radio
    col_inicio = col - radio - desplaz + 1
    col_fin = col + radio - desplaz + 1
    return imagen[fila_inicio:fila_fin, col_inicio:col_fin]

def ajuste_subpixel(pos_mejor, lista_errores, max_disp):
    errores = np.array(lista_errores, dtype=np.float64)
    if (0 < pos_mejor < max_disp - 1 and
        np.all(np.isfinite(errores[pos_mejor-1:pos_mejor+2]))
    ):
        denom = errores[pos_mejor-1] + errores[pos_mejor+1] - 2 * errores[pos_mejor]
        if denom != 0:
            num = errores[pos_mejor-1] - errores[pos_mejor+1]
            corr = num / (2 * denom)
            if -1 <= corr <= 1:
                return corr
    return 0.0

'''
Esta funcion mejora la precisión de la disparidad usando interpolación parabólica buscando un mejor valor "entre píxeles" cuando la mejor 
coincidencia no está exactamente en un píxel entero
'''

def calcular_mapa_disparidad(img_izq, img_der, tam_ventana, max_disp, usar_subpixel=True):
    alto, ancho = img_izq.shape
    mapa_disp = np.zeros_like(img_izq, dtype=np.float32)
    radio = tam_ventana // 2
    escala = 255  / max_disp

    for y in range(radio, alto - radio):
        for x in range(max_disp, ancho - radio):
            lista_errores = []
            mejor_disp = 0
            error_min = float('inf')

            for d in range(max_disp):
                ventana_izq = obtener_ventana(img_izq, y, x, tam_ventana)
                ventana_der = obtener_ventana(img_der, y, x, tam_ventana, d)

                if ventana_izq.shape != ventana_der.shape:
                    lista_errores.append(np.inf)
                    continue

                ssd = np.sum((ventana_izq - ventana_der) ** 2)
                lista_errores.append(ssd)

                if ssd < error_min:
                    error_min = ssd
                    mejor_disp = d

            if usar_subpixel:
                mejor_disp += ajuste_subpixel(mejor_disp, lista_errores, max_disp)

            mapa_disp[y, x] = mejor_disp * escala

    return mapa_disp

def filtro_bilateral_gris(imagen, sigma_esp=3, sigma_int=0.1):
    def filtro_vecino(parche):
        centro = parche[len(parche)//2]
        pes_espacial = np.exp(-0.5 * ((np.arange(len(parche)) - len(parche)//2)**2) / (sigma_esp**2))
        pes_intensidad = np.exp(-0.5 * ((parche - centro)**2) / (sigma_int**2))
        pesos = pes_espacial * pes_intensidad
        return np.sum(parche * pesos) / np.sum(pesos)

    return generic_filter(imagen, filtro_vecino, size=3, mode='reflect')

'''
Esta funcion aplica un filtro bilateral personalizado en imágenes escala de grises:
 -Suaviza la imagen sin perder bordes importantes.
 -Tiene en cuenta la cercanía entre píxeles y la diferencia de intensidad.
'''

def afilar_imagen(img_norm, sigma=0.3, factor=0.5, limite_min=0.2):
    desenf = gaussian_filter(img_norm, sigma=sigma)
    img_afilada = np.clip(img_norm + factor * (img_norm - desenf), limite_min, 1)
    return img_afilada

'''
Aqui aumenta el contraste entre detalles finos (bordes) de la imagen.
'''

def guardar_imagen_disparidad(mapa_disp, ruta="data/disparidad_plana.png"):
    disp_filtrada = filtro_bilateral_gris(mapa_disp, sigma_esp=2, sigma_int=0.1)
    disp_norm = (disp_filtrada - disp_filtrada.min()) / (disp_filtrada.max() - disp_filtrada.min() + 1e-6)
    disp_afilada = afilar_imagen(disp_norm)
    img_disp = Image.fromarray(np.uint8(disp_afilada * 255), mode='L')
    img_disp.save(ruta)
    print(f"Disparidad guardada en: {ruta}")

def generar_nube_puntos(mapa_disp, colores, foco=1.0, base=0.05, ruta_ply="data/nube_plana.ply"):
    h, w = mapa_disp.shape
    mascara = mapa_disp > 0
    indices = np.indices((h, w), dtype=np.float32)
    xs = indices[1][mascara]
    ys = indices[0][mascara]
    disp_vals = mapa_disp[mascara]

    z = (foco * base) / disp_vals
    x = (xs - w / 2) * z / foco
    y = (ys - h / 2) * z / foco

    puntos = np.stack((x, y, z), axis=1)
    colores_norm = colores[mascara].astype(np.float32) / 255.0

    nube = o3d.geometry.PointCloud()
    nube.points = o3d.utility.Vector3dVector(puntos)
    nube.colors = o3d.utility.Vector3dVector(colores_norm)

    o3d.visualization.draw_geometries([nube])

    o3d.io.write_point_cloud(ruta_ply, nube)
    print(f"Nube de puntos guardada en: {ruta_ply}")

def cargar_imagenes(ruta_izq, ruta_der):
    img_izq_rgb = Image.open(ruta_izq).convert("RGB")
    img_der_rgb = Image.open(ruta_der).convert("RGB")
    img_izq_gray = np.array(img_izq_rgb.convert("L"))
    img_der_gray = np.array(img_der_rgb.convert("L"))
    return img_izq_rgb, img_der_rgb, img_izq_gray, img_der_gray

def main():
    ruta_izq = "/mnt/c/Users/vicen/Documents/GitHub/Vision3D/Etapa5_6/data/rect_hartley_izq.png"
    ruta_der = "/mnt/c/Users/vicen/Documents/GitHub/Vision3D/Etapa5_6/data/rect_hartley_der.png"

    izq_rgb, der_rgb, izq_gray, der_gray = cargar_imagenes(ruta_izq, ruta_der)

    disparidad = calcular_mapa_disparidad(izq_gray, der_gray, tam_ventana=9, max_disp=64, usar_subpixel=True)
    guardar_imagen_disparidad(disparidad)
    generar_nube_puntos(disparidad, np.array(izq_rgb))

if __name__ == "__main__":
    main()

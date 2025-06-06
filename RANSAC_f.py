import numpy as np #funciones matematicas
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import random

def normalizar_puntos(pts):
    centroide = np.mean(pts, axis=0)
    pts_center = pts - centroide
    dist = np.mean(np.linalg.norm(pts_center, axis=1))
    scale = np.sqrt(2) / dist
    T = np.array([
        [scale, 0, -scale * centroide[0]],
        [0, scale, -scale * centroide[1]],
        [0,     0,                  1]
    ])
    pts_homog = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_norm = (T @ pts_homog.T).T[:, :2]
    return pts_norm, T

'''
Esta funcion se encarga de normalizar los puntos para mejorar la precision en calculos geometricos. 

Primero, calcula el centroide de los puntos y luego ajusta cada punto restandole el centroide para centrarlo en el origen. Despues, encuentra la distancia promedio 
de los puntos al centro y define un factor de escala que permite controlar la dispersion. Luego, construye una matriz de transformacion que aplica este ajuste 
a los puntos en coordenadas homogeneas. Al final, devuelve los puntos ya normalizados junto con la matriz de transformacion, que sera util para revertir la 
normalizacion cuando sea necesario.
'''

def construir_A(pts1, pts2):
    A = []
    for p1, p2 in zip(pts1, pts2):
        x1, y1 = p1
        x2, y2 = p2
        A.append([
            x2 * x1, x2 * y1, x2,
            y2 * x1, y2 * y1, y2,
            x1, y1, 1
        ])
    return np.array(A)

'''
Aqui construimos la matriz A que nos servira para calcular la matriz fundamental. Para ello, tomamos pares de puntos correspondientes de dos imagenes y 
organizamos sus coordenadas en una estructura algebraica especifica. Cada fila de la matriz A representa una ecuacion que describe la relacion entre un punto 
en la primera imagen y su correspondiente en la segunda. 
'''

def estimar_F_8p(pts1, pts2): #algoritmo de los 8 puntos
    pts1_norm, T1 = normalizar_puntos(pts1)
    pts2_norm, T2 = normalizar_puntos(pts2)

    A = construir_A(pts1_norm, pts2_norm)
    _, _, Vt = np.linalg.svd(A)
    F_norm = Vt[-1].reshape(3, 3)

    # Hacer que F tenga rango 2
    U, S, Vt = np.linalg.svd(F_norm)
    S[2] = 0
    F_rango2 = U @ np.diag(S) @ Vt

    # Desnormalizar
    F = T2.T @ F_rango2 @ T1
    return F / F[2,2]

'''
En esta funcion calculamos la matriz fundamental F con el metodo de los 8 puntos. Primero, normalizamos los puntos para mejorar la estabilidad numerica en el calculo. 

Luego, construimos la matriz A con los puntos normalizados y aplicamos descomposicion en valores singulares (SVD) para obtener la mejor aproximacion de F. Sin embargo, 
matematicamente F debe tener rango 2, asi que ajustamos sus valores singulares para que cumpla con esta condicion. Por ultimo, desnormalizamos F utilizando las 
matrices de transformacion obtenidas en la etapa de normalizacion, asegurando que el resultado sea coherente con las coordenadas originales.
'''

def calcular_error_sampson(F, pts1, pts2):
    pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
    pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])

    Fx1 = F @ pts1_h.T
    Ftx2 = F.T @ pts2_h.T
    numerador = np.square(np.sum(pts2_h * (F @ pts1_h.T).T, axis=1))
    denominador = Fx1[0]**2 + Fx1[1]**2 + Ftx2[0]**2 + Ftx2[1]**2
    return numerador / denominador

'''
Esta funcion calcula el error de Sampson, primero, convierte los puntos a coordenadas homogeneas para que los calculos sean correctos. Luego, usa F para obtener 
las lineas epipolares en cada imagen y calcula el error comparando los valores obtenidos con los puntos reales. 

El resultado es un conjunto de errores para cada correspondencia, lo que nos permitira identificar que puntos cumplen con el modelo y cuales pueden ser descartados.

(Resolver error geometrico, con la linea epipolar / distacia de punto a linea)
'''

def ransac_fundamental(pts1, pts2, threshold=1.0, max_iter=2000):
    best_F = None
    best_inliers = []
    n = pts1.shape[0]

    for _ in range(max_iter):
        idx = np.random.choice(n, 8, replace=False)
        F_candidate = estimar_F_8p(pts1[idx], pts2[idx])
        error = calcular_error_sampson(F_candidate, pts1, pts2)
        inliers = np.where(error < threshold)[0]

        if len(inliers) > len(best_inliers):
            best_F = F_candidate
            best_inliers = inliers

    if best_F is None:
        raise ValueError("No se encontro ninguna F valida con RANSAC.")

    # Reestimar F con todos los inliers
    F_final = estimar_F_8p(pts1[best_inliers], pts2[best_inliers])
    return F_final, best_inliers

'''
Para obtener una matriz fundamental mas precisa, usamos RANSAC. Luego, evaluamos que tan bien la matriz obtenida ajusta los puntos usando el error de Sampson y 
contamos cuantos puntos cumplen con el modelo. Guardamos la mejor version de F encontrada y al final recalculamos la matriz utilizando todos los inliers, 
obteniendo una estimacion mas confiable.
'''

'''
#cambiar la manera de dibujarlo, quiere dos pares de imagenes, en una la derecha tendra puntos y la izquierda lineas y luego cambiarlo 
#diagrama epipolar con nuestra imagenes, con menos lineas, quiere seleccionar puntos de la imagen sin las correspondencias mostrar si se puede donde converge el epipolo.
def dibujar_lineas(imagen1_path, imagen2_path, pts1, pts2, F, inliers, num_lineas=10, mostrar_epipolos=False):
    # Corregir orientación EXIF
    img1 = ImageOps.exif_transpose(Image.open(imagen1_path))
    img2 = ImageOps.exif_transpose(Image.open(imagen2_path))

    img1 = np.array(img1)
    img2 = np.array(img2)

    # Redimensionar alturas
    if img1.shape[0] != img2.shape[0]:
        altura_comun = min(img1.shape[0], img2.shape[0])
        img1 = np.array(Image.fromarray(img1).resize((img1.shape[1], altura_comun)))
        img2 = np.array(Image.fromarray(img2).resize((img2.shape[1], altura_comun)))

    img = np.hstack((img1, img2))
    width = img1.shape[1]

    # Submuestreo aleatorio de líneas
    if len(inliers) > num_lineas:
        muestra = random.sample(list(inliers), num_lineas)
    else:
        muestra = inliers

    pts1_in = pts1[muestra]
    pts2_in = pts2[muestra]

    pts1_h = np.hstack([pts1_in, np.ones((len(muestra), 1))])
    pts2_h = np.hstack([pts2_in, np.ones((len(muestra), 1))])

    lines2 = (F @ pts1_h.T).T
    lines1 = (F.T @ pts2_h.T).T

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    # Left image
    axs[0].imshow(img1)
    axs[0].set_title("Left image")
    axs[0].axis("off")
    for (a, b, c), (x, y) in zip(lines1, pts1_in):
        x0, x1 = 0, img1.shape[1]
        y0 = int(-(a * x0 + c) / b) if b != 0 else 0
        y1 = int(-(a * x1 + c) / b) if b != 0 else img1.shape[0]
        axs[0].plot([x0, x1], [y0, y1], 'k-', linewidth=1)
        axs[0].plot(x, y, 'wo', markersize=6, markeredgecolor='black')

    # Right image
    axs[1].imshow(img2)
    axs[1].set_title("Right image")
    axs[1].axis("off")
    for (a, b, c), (x, y) in zip(lines2, pts2_in):
        x0, x1 = 0, img2.shape[1]
        y0 = int(-(a * x0 + c) / b) if b != 0 else 0
        y1 = int(-(a * x1 + c) / b) if b != 0 else img2.shape[0]
        axs[1].plot([x0, x1], [y0, y1], 'k-', linewidth=1)
        axs[1].plot(x, y, 'wo', markersize=6, markeredgecolor='black')

    # Calcular epipolos
    _, _, Vt = np.linalg.svd(F)
    epipolo_d = Vt[-1] / Vt[-1][2]  # En imagen derecha

    _, _, Vt_T = np.linalg.svd(F.T)
    epipolo_i = Vt_T[-1] / Vt_T[-1][2]  # En imagen izquierda

    # Dibujar epipolos en las imágenes
    axs[0].plot(epipolo_i[0], epipolo_i[1], 'rx', markersize=10, label='Epipolo')
    axs[1].plot(epipolo_d[0], epipolo_d[1], 'rx', markersize=10, label='Epipolo')

    # Leyenda (solo una vez, si quieres)
    axs[0].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig("output/lineas_epipolares.png")
    plt.show()
'''

def dibujar_lineas(imagen1_path, imagen2_path, pts1, pts2, F, inliers, num_lineas=10, mostrar_epipolos=True):
    from PIL import ImageOps

    # Cargar y corregir orientación EXIF
    img1 = ImageOps.exif_transpose(Image.open(imagen1_path))
    img2 = ImageOps.exif_transpose(Image.open(imagen2_path))

    # Redimensionar ambas imágenes para que tengan la misma altura (la menor entre las dos)
    altura_comun = min(img1.height, img2.height)
    img1 = img1.resize((int(img1.width * altura_comun / img1.height), altura_comun), Image.Resampling.LANCZOS)
    img2 = img2.resize((int(img2.width * altura_comun / img2.height), altura_comun), Image.Resampling.LANCZOS)

    img1 = np.array(img1)
    img2 = np.array(img2)

    # Submuestreo aleatorio
    if len(inliers) > num_lineas:
        muestra = random.sample(list(inliers), num_lineas)
    else:
        muestra = inliers

    pts1_in = pts1[muestra]
    pts2_in = pts2[muestra]

    pts1_h = np.hstack([pts1_in, np.ones((len(muestra), 1))])
    pts2_h = np.hstack([pts2_in, np.ones((len(muestra), 1))])

    lines2 = (F @ pts1_h.T).T  # Líneas en imagen derecha generadas por pts1
    lines1 = (F.T @ pts2_h.T).T  # Líneas en imagen izquierda generadas por pts2

    # Epipolos
    _, _, Vt = np.linalg.svd(F)
    epipolo_der = Vt[-1] / Vt[-1][2]

    _, _, Vt_T = np.linalg.svd(F.T)
    epipolo_izq = Vt_T[-1] / Vt_T[-1][2]

    def trazar_lineas(ax, img, lineas):
        ax.imshow(img)
        ax.axis("off")
        for a, b, c in lineas:
            x0, x1 = 0, img.shape[1]
            if b != 0:
                y0 = -(a * x0 + c) / b
                y1 = -(a * x1 + c) / b
                ax.plot([x0, x1], [y0, y1], 'b-', linewidth=1)

    def trazar_puntos(ax, img, puntos):
        ax.imshow(img)
        ax.axis("off")
        for x, y in puntos:
            ax.plot(x, y, 'ro', markersize=5, markeredgecolor='black')

    def trazar_epipolo_fuera(ax, epipolo, lado, img_shape):
        y_lim = img_shape[0]
        x_offset = 60  # Más separado del borde que antes

        if lado == 'izq':
            x = -x_offset
            y = np.clip(epipolo[1], 0, y_lim)
        elif lado == 'der':
            x = img_shape[1] + x_offset
            y = np.clip(epipolo[1], 0, y_lim)
        else:
            x, y = epipolo[0], epipolo[1]

        ax.plot(x, y, 'rx', markersize=12)

    #FIGURA 1
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    trazar_puntos(ax1, img1, pts1_in)
    ax1.set_title("Imagen Izquierda (Puntos)")
    trazar_lineas(ax2, img2, lines2)
    ax2.set_title("Imagen Derecha (Líneas)")
    if mostrar_epipolos:
        trazar_epipolo_fuera(ax1, epipolo_izq, 'izq', img1.shape)

    plt.subplots_adjust(wspace=0.05)  
    plt.tight_layout()
    plt.savefig("output/epipolar_fig1.png")
    plt.close(fig1)

    #FIGURA 2
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 7))
    trazar_lineas(ax3, img1, lines1)
    ax3.set_title("Imagen Izquierda (Líneas)")
    trazar_puntos(ax4, img2, pts2_in)
    ax4.set_title("Imagen Derecha (Puntos)")
    if mostrar_epipolos:
        trazar_epipolo_fuera(ax4, epipolo_der, 'der', img2.shape)

    plt.subplots_adjust(wspace=0.05)
    plt.tight_layout()
    plt.savefig("output/epipolar_fig2.png")
    plt.close(fig2)


if __name__ == "__main__":
    # Cargar los puntos emparejados
    pts1 = np.load("output/pts1.npy")
    pts2 = np.load("output/pts2.npy")

    # Ejecutar RANSAC para obtener F
    F, inliers = ransac_fundamental(pts1, pts2, threshold=1.0)

    print("Matriz fundamental F:\n", F)
    print(f"\nSe encontraron {len(inliers)} correspondencias validas.")

    # Guardar F e inliers para la siguiente fase
    np.save("output/F.npy", F)
    np.save("output/inliers.npy", inliers)

    # Visualizar líneas epipolares sin OpenCV
    '''
    dibujar_lineas("output/sift_matches.png", pts1, pts2, F, inliers)
    '''
    '''
    dibujar_lineas("Imagenes/image15.png", "Imagenes/image16.png", pts1, pts2, F, inliers)
    '''
    dibujar_lineas("Imagenes/redimensionadas/left1.png", "Imagenes/redimensionadas/right1.png", pts1, pts2, F, inliers, num_lineas=10)
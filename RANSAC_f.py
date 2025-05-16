import numpy as np #funciones matematicas

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

'''
Dibujar lineas epipolares
'''
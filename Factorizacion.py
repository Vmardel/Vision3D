import numpy as np #funciones matematicas

def descomposicion_rq(M):

    # Invertimos columnas y filas para aplicar QR
    M_inv = np.flipud(np.fliplr(M))
    Q, R = np.linalg.qr(M_inv.T)
    R = np.fliplr(np.flipud(R.T))
    Q = np.fliplr(np.flipud(Q.T))

    # diagonal de K sea positiva si o si
    T = np.diag(np.sign(np.diag(R)))
    K = R @ T
    R = T @ Q

    # Normalizar K para que K[2,2] = 1 (si no es cero)
    if K[2, 2] != 0:
        K = K / K[2, 2]

    return K, R

def factorizar_P(P):

    print("Matriz de proyeccion P:\n", P)
    print("Forma:", P.shape)

    M = P[:, :3] #3 primeras columnas de la matriz proyeccion P
    K, R = descomposicion_rq(M)

    print("Matriz K antes de resolver t:\n", K)
    print("Determinante de K:", np.linalg.det(K))

    # vector de traslacion t
    try:
        t = np.linalg.solve(K, P[:, 3])
    except np.linalg.LinAlgError:
        print("La matriz K es singular o casi singular.")
        t = np.zeros(3)

    return K, R, t

'''
Este codigo descompone la matriz de proyeccion p sacada del codigo Calibra_Cam.py.

Para ello deberemos realizar la descomposicion RQ (igual que la QR pero invertida) para obtener la matriz K y la matriz R con la funcion 
descomposicion_rq, que trabaja con M (siendo esta las 3 primeras columnas de la matriz de proyeccion P)

Esta funcion se encarga de lo siguiente:
- Se transforma invierte filas y columnas de M y luego le aplica la descomposicion.
- Se reajustan los resultados para que se alineen con la estructura RQ (QR invertida).
- Se asegura que la diagonal de K sea positiva y normaliza la matriz K para que K[2,2] (fila y columna 3) sea siempre 1 para evitar problemas de interpretacion futuros.

Una vez formulada la descomposicion podemos acabar la factorizacion de P, para ello:
- Extrae la matriz de transformacion M de la matriz P (solo primeras 3 columnas).
- LLamamos a la funcion de descomposicion para obtener K y R y evalua el determinante de K.
- Calculamos el vector de traslacion t resolviendo un sistema lineal y si K es singular (no invertible), asigna un vector nulo.

Por utlimo simplemente guardamos la informacion de la matriz intrinseca K, la matriz de rotacion R y la de traslacion t en archivos .npy para su posterior uso.
'''

if __name__ == "__main__":
    # Cargar matriz de proyeccion
    P = np.load("output/P.npy")

    K, R, t = factorizar_P(P)

    print("\nResultados de la factorizacion:")
    print("\nMatriz intrinseca K:\n", K)
    print("\nMatriz de rotaciin R:\n", R)
    print("\nVector de traslaciin t:\n", t)
    np.save("output/K_P.npy", K)
    np.save("output/R_P.npy", R)
    np.save("output/t_P.npy", t)

'''
reazliar un test de los inputs y outputs (testeo unitario / unit tester)
'''
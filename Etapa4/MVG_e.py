import numpy as np #funciones matematicas
import os  # para crear carpetas

def calcular_matriz_esencial(F, K_l, K_r=None):
    '''
    Calcula la matriz esencial E a partir de la matriz fundamental F
    y las matrices de calibracion Kl y Kr.
    Si Kr no se proporciona, se asume que Kl = Kr.
    '''
    if K_r is None:
        K_r = K_l
    E = K_r.T @ F @ K_l
    return E / np.linalg.norm(E)

'''
Esta funcion calcula la matriz esencial E a partir de la matriz fundamental F y las matrices de calibracion de la camara (Kl y Kr).

Primero, verifica si la matriz de calibracion de la segunda camara \(K_r\) fue proporcionada; si no, asume que es la misma que la primera. Luego, multiplica las 
matrices de calibracion con F para obtener E. Finalmente, devuelve la matriz normalizada, lo que ayuda a estabilizar los calculos posteriores.
'''

if __name__ == "__main__":
    # Cargar matriz fundamental F y calibracion K (misma camara)
    F = np.load("/mnt/c/Users/vicen/Documents/GitHub/Vision3D/Etapa3/data/F.npy")
    K = np.load("/mnt/c/Users/vicen/Documents/GitHub/Vision3D/Etapa2/data/K_intrinseca.npy")  

    # matriz esencial
    E = calcular_matriz_esencial(F, K)

    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)

    # Mostrar y guardar resultado
    print(" Matriz esencial E:\n", E)
    np.save("data/E.npy", E)
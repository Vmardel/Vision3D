import cv2 #opencv
import numpy as np #funciones matematicas
import glob #trata de ficheros
import os #trata de directorios

def calibrar_camara(imagenes, tamano_tablero, tamano_casilla):
    objp = np.zeros((tamano_tablero[0]*tamano_tablero[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:tamano_tablero[0], 0:tamano_tablero[1]].T.reshape(-1, 2)
    objp *= tamano_casilla

    objpoints = []
    imgpoints = []

    images = glob.glob(os.path.join(imagenes, '*.png'))
    if len(images) == 0:
        print("No se encontraron imagenes.")
        return

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, tamano_tablero, None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(img, tamano_tablero, corners, ret)

            # imagen con esquinas
            out_dir = "output"
            os.makedirs(out_dir, exist_ok=True)
            out_name = os.path.basename(fname).replace('.png', '_corners.png')
            cv2.imwrite(os.path.join(out_dir, out_name), img)

    # Calibrar camara (rvecs y tvecs documentacion oficial de OpenCV)
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Obtener matriz de proyeccion P
    R, _ = cv2.Rodrigues(rvecs[0])
    t = tvecs[0]
    Rt = np.hstack((R, t))
    P = K @ Rt

    # Guardar matriz P
    np.save("output/P.npy", P)
    print("Calibracion completada. Matriz P guardada en output/P.npy")

'''
Esta funcion calcula la matriz de calibracion de la camara a partir de imagenes de un tablero con patron de tablero de ajedrez.

Primero inicializamos los puntos 3D creando objp, que contiene la disposicion de las esquinas del tablero.
Luego se multiplica por tamano_casilla, que representa el tamaño de cada casilla del tablero en milimetros (en nuestro caso al ser un tablero de 8 x 6 casillas,
deberemos utilziar un tamaño de 7 x 5 debido a que trabajaremos con las intersecciones internas del tablero)

Luego buscamos todas las imagenes .png dentro de la carpeta especificada y si no hay la ejecucion se detiene.

Para lograr la deteccion de esquinas trabajamos con las imagenes de manera individual de la siguiente forma:
- Se convierte a escala de grises.
- Aplicamos cv2.findChessboardCorners para detectar las esquinas, si se consigue agregamos la informaicon de los puntos reales a objpoints y los puntos 2D a imgpoints.
- Se dibujan las esquinas sobre la imagen y se guarda una copia de la imagen con las esquinas detectadas en la carpeta output.

Con lo anterior llamamos a cv2.calibrateCamera, que calcula los parametros de la camara utilizando las esquinas detectadas lo que devuelve:

- K: Matriz intrinseca.
- rvecs y tvecs: Vectores de rotacion y traslacion.

Por ultimo obtenemos la matriz de proyeccion (P) con las matrices de rotacion y traslacion obtenidas de los vectores anteriores, luego, guardamos la informacion
de la matriz P en un archivo .npy para su posterior uso.
'''

if __name__ == "__main__":
    imagenes = "/mnt/c/Users/vicen/Documents/GitHub/Vision3D/Imagenes"
    tamano_tablero = (7, 5)      # Esquinas internas (tablero de 8x6 casillas)
    tamano_casilla = 30         # milimetros

    calibrar_camara(imagenes, tamano_tablero, tamano_casilla)

import numpy as np
from skimage.transform import ProjectiveTransform, warp
from PIL import Image
import os

def cargar_imagen(path):
    img = Image.open(path).convert("RGB")
    return np.array(img) / 255.0

def guardar_imagen(path, imagen):
    img = Image.fromarray((imagen * 255).astype(np.uint8))
    img.save(path)

def estimar_epipolo(F):
    """
    Calcula el epipolo como el null space de F (para la imagen derecha) y de F^T (para la izquierda),
    asegurando que la tercera componente sea positiva.
    """
    U, S, Vt = np.linalg.svd(F)
    e = Vt[-1]
    if e[2] < 0:
        e = -e  # Normalizamos para mantener e[2] positivo
    return e / e[2]

def construir_transformacion(epipolo, y0, flip=False):
    """
    Construye una transformación proyectiva basada en el epipolo y el centro de la imagen.
    El parámetro flip invierte el ángulo para corregir la orientación (útil en la imagen derecha).
    """
    e = epipolo[:2] / epipolo[2]
    y0 = y0[:2] / y0[2]
    # Trasladamos el centro a (0,0)
    t = -y0
    e_trasladado = e + t
    # Calculamos el ángulo de rotación
    theta = -np.arctan2(e_trasladado[1], e_trasladado[0])
    if flip:
        theta = -theta  # Invierte la rotación para evitar que la imagen derecha se vea invertida
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    e_rotado = R @ e_trasladado
    f = np.linalg.norm(e_rotado)
    H = np.eye(3)
    H[:2, :2] = R
    H[:2, 2] = -R @ y0
    return H, f

def ajustar_homografia(H, shape):
    """
    Ajusta la homografía H para que, al aplicar la transformación, la imagen quede completamente
    dentro del marco de salida.
    """
    h, w = shape
    # Definimos las esquinas de la imagen original en coordenadas homogéneas
    esquinas = np.array([
        [0,     0,     1],
        [w - 1, 0,     1],
        [w - 1, h - 1, 1],
        [0,     h - 1, 1]
    ]).T  # forma: (3,4)
    
    # Se aplican las transformaciones a las esquinas
    esquinas_trans = H @ esquinas
    esquinas_trans /= esquinas_trans[2, :]
    
    # Hallamos el mínimo en x e y
    min_x = np.min(esquinas_trans[0, :])
    min_y = np.min(esquinas_trans[1, :])
    
    # Matriz de traslación para desplazar la imagen a coordenadas positivas
    T = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ])
    return T @ H

def construir_rectificacion(F, pts1, pts2, img_shape):
    """
    A partir de la matriz fundamental F y puntos clave, calcula las homografías de rectificación
    para la imagen izquierda y la derecha, realizando además la corrección de escala y centrado.
    """
    h, w = img_shape
    # Utilizamos el centro de la imagen para la transformación
    centro = np.array([w / 2, h / 2, 1])
    y0 = centro
    
    # Calculamos los epipolos (F.T para la izquierda, F para la derecha)
    e1 = estimar_epipolo(F.T)
    e2 = estimar_epipolo(F)
    
    H1_trans, _ = construir_transformacion(e1, y0, flip=False)
    H2_trans, f = construir_transformacion(e2, y0, flip=True)
    
    # Corrección de escala
    A = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [-1/f, 0, 1]])
    H1 = A @ H1_trans
    H2 = A @ H2_trans
    
    # Ajustamos la homografía para centrar la imagen en el marco de salida
    H1 = ajustar_homografia(H1, (h, w))
    H2 = ajustar_homografia(H2, (h, w))
    
    return H1, H2

def aplicar_homografia(img, H, output_shape):
    H_inv = np.linalg.inv(H)
    transform = ProjectiveTransform(H_inv)
    return warp(img, transform, output_shape=output_shape)

def rectificar_y_guardar(im_izq_path, im_der_path, F, pts1, pts2, out_dir="output"):
    os.makedirs(out_dir, exist_ok=True)
    im1 = cargar_imagen(im_izq_path)
    im2 = cargar_imagen(im_der_path)
    shape = im1.shape[:2]

    H1, H2 = construir_rectificacion(F, pts1, pts2, shape)
    img1_rect = aplicar_homografia(im1, H1, shape)
    img2_rect = aplicar_homografia(im2, H2, shape)

    guardar_imagen(os.path.join(out_dir, "rectificada_izq.png"), img1_rect)
    guardar_imagen(os.path.join(out_dir, "rectificada_der.png"), img2_rect)
    print("[✓] Rectificación completada correctamente.")

if __name__ == "__main__":
    # Actualiza estos paths con la localización de tus archivos
    F = np.load("output/F.npy")
    pts1 = np.load("output/pts1.npy")
    pts2 = np.load("output/pts2.npy")
    img_izq = "Imagenes/redimensionadas/left1.png"
    img_der = "Imagenes/redimensionadas/right1.png"

    rectificar_y_guardar(img_izq, img_der, F, pts1, pts2)

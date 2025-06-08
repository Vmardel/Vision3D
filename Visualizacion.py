import numpy as np

# Cargar el archivo .npy
data = np.load("/mnt/c/Users/vicen/Documents/GitHub/Vision3D/Etapa2/data/K_intrinseca.npy")
data2 = np.load("/mnt/c/Users/vicen/Documents/GitHub/Vision3D/Etapa2/data/R_rotacion.npy")
data3 = np.load("/mnt/c/Users/vicen/Documents/GitHub/Vision3D/Etapa2/data/t_traslacion.npy")

# Mostrar el contenido
print(data)
print(data2)
print(data3)


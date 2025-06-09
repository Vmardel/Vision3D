Para ejecutar el pipeline completo hay que seguir estos pasos:
Fase 1:
Cambiar en el main el direccionamiento de los directorios del codigo calibra_cam.py a la carpeta /imagenes dentro del propio directorio de trabajo Fase1
Ejecutar el archivo en una terminal de ubuntu con python Calibra_Cam.py
Fase 2:
Cambiar en el main el direccionamiento de los directorio del codigo Factorizacion.py a la carpeta /data de la Fase 1 en concreto al archivo P.npy dentro del propio directorio de trabajo Fase1
Ejecutar el archivo en una terminal de ubuntu con python Factorizacion.py
Fase 3:
Cambiar en el main el direccionamiento de los directorios del codigo SIFT.py a la carpeta /imagenes dentro del propio directorio de trabajo Fase3
Cambiar en el main el direccionamiento de los directorios del codigo RANSAC.py a la carpeta /data dentro del propio directorio de trabajo Fase3
Ejecutar el archivo en una terminal de ubuntu con python SIFT.py y luego python RANSAC.py
Fase 4:
Cambiar en el main el direccionamiento del directorios del codigo MVG_e.py de la carga de la matriz F a la carpeta /data dentro del propio directorio de trabajo Fase3
Cambiar en el main el direccionamiento del directorios del codigo MVG_e.py de la carga de la matriz K a la carpeta /data dentro del propio directorio de trabajo Fase2
Ejecutar el archivo en una terminal de ubuntu con python MVG_e.py
Fase 5-6:
Cambiar en el main el direccionamiento del directorios del codigo Rectificacion.py de la carga de la matriz E a la carpeta /data dentro del propio directorio de trabajo Fase4
Cambiar en el main el direccionamiento del directorios del codigo Rectificacion.py de la carga de la matriz K a la carpeta /data dentro del propio directorio de trabajo Fase2
Cambiar en el main el direccionamiento del directorios del codigo Rectificacion.py de la carga de los puntos pts1 y pts2 a la carpeta /data dentro del propio directorio de trabajo Fase3
Cambiar en el main el direccionamiento del directorios del codigo Rectificacion.py de la carga de los inliers a la carpeta /data dentro del propio directorio de trabajo Fase3
Cambiar en el main el direccionamiento de los directorios del codigo Rectificacion.py a la carpeta /imagenes dentro del propio directorio de trabajo Fase3
Ejecutar el archivo en una terminal de ubuntu con python Rectificacion.py
Fase 7:
Cambiar en el main el direccionamiento de los directorios del codigo Nube_de_puntos_profunda.py a la carpeta /data dentro del propio directorio de trabajo Fase5-6
Ejecutar el archivo en una terminal de ubuntu con python Nube_de_puntos_profunda.py

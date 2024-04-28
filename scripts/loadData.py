import os
import time
import uuid
import cv2
from sklearn.model_selection import train_test_split
import shutil


# Creacion de carpetas en caso de no estar creadas
ruta_directorio = "data"
if not os.path.exists(ruta_directorio):
    os.makedirs(ruta_directorio)
    print("Directorio creado:", ruta_directorio)
    os.makedirs(os.path.join(ruta_directorio, "images"))
    subdirectorios = ["train", "test", "val"]
    for subdirectorio in subdirectorios:
        os.makedirs(os.path.join(ruta_directorio, subdirectorio))
        os.makedirs(os.path.join(ruta_directorio, subdirectorio, "images"))
        os.makedirs(os.path.join(ruta_directorio, subdirectorio, "labels"))
    print("Carpetas 'images', train, test y val creadas dentro de 'data'")

# Direccion donde se guardan imagenes
IMAGES_PATH =  os.path.join('data','images')
number_images = 100

cap = cv2.VideoCapture(1)
for imgnum in range(number_images):
    print('Collecting image {}'.format(imgnum))
    ret, frame = cap.read()
    imgname = os.path.join(IMAGES_PATH,f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(imgname, frame)
    cv2.imshow('frame', frame)
    time.sleep(0.2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

# Obtener una lista de todas las imágenes
imagenes = [archivo for archivo in os.listdir('data/images') if archivo.lower().endswith(('.png', '.jpg', '.jpeg'))]

train, test_val = train_test_split(imagenes, test_size=0.3, random_state=42)
test, val = train_test_split(test_val, test_size=0.5, random_state=42)

# Función para copiar las imágenes a las nuevas carpetas
def copiar_imagenes(imagenes, ruta_destino):
    for imagen in imagenes:
        ruta_origen_imagen = os.path.join('data/images', imagen)
        ruta_destino_imagen = os.path.join(ruta_destino, imagen)
        shutil.copy(ruta_origen_imagen, ruta_destino_imagen)

# Copiar las imágenes a sus respectivas carpetas
copiar_imagenes(train, 'data/train/images')
copiar_imagenes(test, 'data/test/images')
copiar_imagenes(val, 'data/val/images')

print('REALIZAR EL ETIQUETADO DE IMAGENES CON LABELME!!')
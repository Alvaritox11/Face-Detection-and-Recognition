# Face Detection and Recognition
Este es un proyecto de Deep Learning donde hacemos un Face Blurring desde cero aplicando detección y reconocimiento facial. El proposito del proyecto es poder difuminar la cara en tiempo real a todas las personas dentro del frame o a alguien en particular especificado, buscando respetar la privacidad de la persona en todo momento.

## Dependencias
Utilizar versiones de Python entre 3.6 y 3.9 para el correcto funcionamiento. Recomendamos la 3.8.7.

Descargar las siguientes dependencias:
` %pip install labelme tensorflow opencv-python matplotlib albumentations numpy`

## Usage
Para poder utilizar este proyecto se necesita crear una base de datos manualmente. Utilizar el script `loadData.py`. Este codigo te hara 100 fotos para guardarlas en una carpeta llamada /images. Se tendra que hacer un tratamiento manual con Labelme para indicar los puntos de la BBox donde se situa la cara y clasificar esta misma como Face. Una vez hecho esto, se crearan los archivos .json y se separaran las imagenes en diferentes carpetas (train, test, val) para realizar el modelo de deteccion. El script `augmentData.py` hara un aumento de imagenes a partir de esas mismas gracias a configuraciones aleatorias con albumentations. Finalmente obtendremos una carpeta aug_data que nos servira para el resto de codigo. 

Tener en cuenta que el pipeline del proyecto esta hecho para que funcione correctamente siguiendo los pasos una sola vez. En caso de querer volver a repetir pasos como el aumento de data pueden ocurrir errores inesperados.

## Carpeta Scripts
- `loadData.py`: hace 100 fotos y las guarda en la carpeta 'data/images'. Separara las imagenes en las carpetas 'train/images', 'test/images' y 'val/images'. Abrir cada una de estas carpetas en Labelme para realizar el etiquetado y guardar los archivos .json en la carpeta 'label' de 'train', 'test' y 'val'.
- `augmentData.py`: se encarga de aumentar las imagenes aplicando diferentes configuraciones.

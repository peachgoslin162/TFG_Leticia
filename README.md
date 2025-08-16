# PFG: Segmentación Semántica en Cirugía Laparoscópica con Optimización para Clases Minoritarias
Se ha desarrollado de una red neuronal convolucional para su uso en segmentación semántica de imágenes laparoscópicas de colecistectomía.
El modelo ha sido entrenado a partir de un conjunto de imágenes reales de cirugías (dataset CholecSeg8k), incorporando técnicas de aumento de datos y estrategias específicas para optimizar la segmentación de clases minoritarias, con el fin de mejorar su capacidad de generalización frente a distintos contextos quirúrgicos y condiciones visuales.
El propósito principal de dicho modelo es identificar y delimitar tanto los distintos elementos anatómicos como las herramientas quirúrgicas más comunes presentes en el campo operatorio.

## Pasos para descargar del Dataset desde Kaggle

Para descargar el dataset `CholecSeg8k` usando el script `download_dataset.py`, sigue estos pasos:


### 1. Obtener tus credenciales Kaggle

- Accede a [Kaggle](https://www.kaggle.com/) y entra en tu cuenta.
- Ve a **My Account** > **API** > **Create New API Token**.
- Se descargará un archivo llamado `kaggle.json` que contiene tu `username` y `key`.

### 2. Crear un archivo `.env`

En la raíz del proyecto, crea un archivo llamado `.env` con este contenido:

KAGGLE_USERNAME=tu_usuario_kaggle
KAGGLE_KEY=tu_key_kaggle


Adicionalmente, para correr y guardar los resultados del modelo se deberá indicar:

DATASET_PATH=./data/cholecseg8k
OUTPUT_DIR=directorio_salida_de_datos


Reemplaza `tu_usuario_kaggle` y `tu_key_kaggle` por los valores que aparecen en el archivo `kaggle.json`.

### 3. Descargar el dataset

Ejecuta el siguiente comando para descargar y extraer el dataset:

```bash
python src/download_dataset.py


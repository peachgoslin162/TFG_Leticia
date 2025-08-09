# Descargar del Dataset desde Kaggle

Para descargar el dataset `CholecSeg8k` usando el script `download_dataset.py`, sigue estos pasos:


## 1. Obtener tus credenciales Kaggle

- Accede a [Kaggle](https://www.kaggle.com/) y entra en tu cuenta.
- Ve a **My Account** > **API** > **Create New API Token**.
- Se descargará un archivo llamado `kaggle.json` que contiene tu `username` y `key`.

## 2. Crear un archivo `.env`

En la raíz del proyecto, crea un archivo llamado `.env` con este contenido:

KAGGLE_USERNAME=tu_usuario_kaggle
KAGGLE_KEY=tu_key_kaggle

Reemplaza `tu_usuario_kaggle` y `tu_key_kaggle` por los valores que aparecen en el archivo `kaggle.json`.

## 3. Descargar el dataset

Ejecuta el siguiente comando para descargar y extraer el dataset:

```bash
python src/download_dataset.py

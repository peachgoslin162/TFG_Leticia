import os
import json
import sys
from dotenv import load_dotenv

def setup_kaggle_token():
    """
    Crea el archivo ~/.kaggle/kaggle.json con las credenciales de Kaggle
    extraídas de las variables de entorno.
    """

    load_dotenv()

    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    if not username or not key:
        print("Error: No se encontraron las variables de entorno KAGGLE_USERNAME y KAGGLE_KEY.")
        print("Por favor, crea un archivo .env con estas variables.")
        print("Consulta el README para más detalles.")
        sys.exit(1)

    kaggle_config = {
        "username": username,
        "key": key
    }

    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)

    token_path = os.path.join(kaggle_dir, "kaggle.json")
    with open(token_path, "w") as f:
        json.dump(kaggle_config, f)

    os.chmod(token_path, 0o600)
    print(f"Archivo kaggle.json creado en {token_path}")

def download_cholecseg8k(output_dir="./data/cholecseg8k"):
    """
    Descarga y descomprime el dataset CholecSeg8k desde Kaggle.

    Parámetros:
        output_dir (str): Carpeta donde se guardará el dataset.
    """
    os.makedirs(output_dir, exist_ok=True)

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files('newslab/cholecseg8k', path=output_dir, unzip=True)

    print(f"Dataset descargado y extraído en: {output_dir}")

if __name__ == "__main__":
    setup_kaggle_token()
    download_cholecseg8k()

import os
import torch as t
from dotenv import load_dotenv

load_dotenv()  # carga las variables .env automáticamente

# Hiperparámetros y configuración
LEARNING_RATE = 3e-4
BATCH_SIZE = 4
START_EPOCH = 0
TOTAL_EPOCHS = 50

HOURS_LIMIT = 11
SECONDS_LIMIT = HOURS_LIMIT * 60 * 60

SEED = 42

DESIRED_IMAGE_SIZE = (360, 640)
TARGET_CLASSES = [7, 8, 11, 12]

CROP_SIZE = [320, 320, 3]
STRONG_CROP_SIZE = [192, 192, 3]

NUM_CLASSES = 13

ALPHA = t.tensor([0.25, 1.0, 1.0, 1.15, 1.0, 1.0, 1.25, 3.0, 3.8, 1.5, 1.0, 4.0, 3.5])
GAMMA = 1.5
WEIGHT_DECAY = 1e-4

# Paths leídos de variables de entorno
DATASET_PATH = os.getenv('DATASET_PATH')
OUTPUT_DIR = os.getenv('OUTPUT_DIR')

# CAMBIAR: usar variables de entorno para checkpoint también

CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pth") if OUTPUT_DIR else None
LAST_CHECKPOINT_PATH = os.getenv('LAST_CHECKPOINT_PATH', '/kaggle/input/epoch_50/pytorch/default/1/best_checkpoint_epoch_50.pth')

METRICS_DIR = os.path.join(OUTPUT_DIR, "plots")

# Diccionario para mapear colores RGB a clases
ID_TO_CLASS = {
    50: 0,
    11: 1,
    21: 2,
    13: 3,
    12: 4,
    31: 5,
    23: 6,
    24: 7,
    25: 8,
    32: 9,
    22: 10,
    33: 11,
    5: 12
}

# Diccionario para mapear índices a nombres de clases
CLASS_NAME_MAPPING = {
    0: 'Black Background',
    1: 'Abdominal Wall',
    2: 'Liver',
    3: 'Gastrointestinal Tract',
    4: 'Fat',
    5: 'Grasper',
    6: 'Connective Tissue',
    7: 'Blood',
    8: 'Cystic Duct',
    9: 'L-hook Electrocautery',
    10: 'Gallbladder',
    11: 'Hepatic Vein',
    12: 'Liver Ligament'
}

# Colores pastel para representar clases (normalizados a [0,1])
PASTEL_COLORS = [
    (166, 206, 227),  # Azul pastel
    (178, 223, 138),  # Verde menta
    (251, 154, 153),  # Rosa suave
    (253, 191, 111),  # Naranja claro
    (202, 178, 214),  # Lila
    (255, 255, 153),  # Amarillo pastel
    (230, 185, 184),  # Rosa viejo
    (255, 222, 173),  # Melocotón suave
    (204, 235, 197),  # Verde claro pastel
    (255, 204, 229),  # Rosa algodón
    (196, 222, 255),  # Celeste tenue
    (240, 230, 140),  # Kaki claro
    (219, 219, 141)   # Verde seco pastel
]

NORMALIZED_COLORS = [(r / 255, g / 255, b / 255) for r, g, b in PASTEL_COLORS]


START_FROM_CHECKPOINT = False
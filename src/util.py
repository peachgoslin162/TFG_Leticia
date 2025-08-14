import random
import numpy as np
import torch as t


# FUNCIÓN PARA FIJAR TODAS LAS SEMILLAS

def set_seed(seed=42):
    random.seed(seed)                         # Semilla para operaciones aleatorias de Python
    np.random.seed(seed)                      # Semilla para NumPy
    t.manual_seed(seed)                       # Semilla para PyTorch CPU
    t.cuda.manual_seed_all(seed)              # Semilla para PyTorch en todas las GPUs

    # Estas dos líneas hacen que las operaciones en GPU se comporten igual siempre
    t.backends.cudnn.deterministic = True     # Resultados deterministas
    t.backends.cudnn.benchmark = False        # No optimiza para velocidad (¡pero sí para repetibilidad!)


def load_checkpoint(checkpoint_path, model, optimizer,
                               cosine_scheduler, early_stopper,
                               best_val_loss,device):
    print(f"Cargando checkpoint desde: {checkpoint_path}")
    checkpoint = t.load(checkpoint_path, map_location=device, weights_only=False)

    # Modelo y optimizador
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)  # Muy importante
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    cosine_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print("Usando scheduler: CosineAnnealing")

    # Métricas
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    mean_ious = checkpoint.get('mean_ious', [])
    pixel_accuracies = checkpoint.get('pixel_accuracies', [])

    start_epoch = checkpoint['epoch'] + 1
    print(f"Reanudar desde la época: {start_epoch}")

    if 'early_stopper_state' in checkpoint:
        early_stopper.__dict__.update(checkpoint['early_stopper_state'])
        print("Estado de EarlyStopping restaurado.")
    else:
        print("No se encontró early_stopper_state. Se inicia desde cero.")
        early_stopper.counter = 0
        early_stopper.best_loss = float("inf")
        early_stopper.early_stop = False

    for group in optimizer.param_groups:
        print("LR restaurado:", group['lr'])

    best_val_loss = checkpoint['best_val_loss']

    # Imprimir el valor de best_val_loss
    print(f"Best validation loss: {best_val_loss}")

    return model, optimizer, start_epoch, train_losses, val_losses, mean_ious, pixel_accuracies, early_stopper, best_val_loss


# Función que para el entrenamiento si no mejora un determinado umbral
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode  # "max" para IoU, "min" para loss

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif ((self.mode == "min" and current_score > self.best_score - self.min_delta) or
              (self.mode == "max" and current_score < self.best_score + self.min_delta)):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.counter = 0


def calculate_percentages(cm):
    row_sums = cm.sum(axis=1)  # Sumar filas (total de ejemplos verdaderos por clase)
    col_sums = cm.sum(axis=0)  # Sumar columnas (total de predicciones por clase)

    # Inicializamos los resultados
    correct = np.zeros(len(cm))  # Inicializamos con 0s
    false_positives = np.zeros(len(cm))
    false_negatives = np.zeros(len(cm))

    for i in range(len(cm)):
        if row_sums[i] == 0:  # Si no hay ejemplos verdaderos de esta clase
            correct[i] = np.nan
            false_positives[i] = np.nan
            false_negatives[i] = np.nan
        else:
            correct[i] = cm[i, i] / row_sums[i] * 100  # Predicción correcta
            false_positives[i] = (col_sums[i] - cm[i, i]) / col_sums[i] * 100  # Falsos positivos
            false_negatives[i] = (row_sums[i] - cm[i, i]) / row_sums[i] * 100  # Falsos negativos

    return correct, false_positives, false_negatives


def calculate_confusion_metrics(cm):
    """
    Calcula métricas clásicas de la matriz de confusión:
    - Recall (Sensibilidad)
    - False Positive Rate (FPR)
    - False Negative Rate (FNR)
    - Precisión (Opcional, comentada)

    Args:
        cm (np.array): Matriz de confusión (n_classes x n_classes).

    Returns:
        dict: Diccionario con arrays de porcentajes para cada métrica.
    """
    row_sums = cm.sum(axis=1)  # Total de verdaderos por clase
    col_sums = cm.sum(axis=0)  # Total de predicciones por clase
    total = cm.sum()  # Total de muestras

    metrics = {
        'recall': np.zeros(len(cm)),
        'fpr': np.zeros(len(cm)),
        'fnr': np.zeros(len(cm)),
        # 'precision': np.zeros(len(cm))  # Opcional
    }

    for i in range(len(cm)):
        TP = cm[i, i]
        FP = col_sums[i] - TP
        FN = row_sums[i] - TP
        TN = total - row_sums[i] - FP

        # Recall (Sensibilidad)
        metrics['recall'][i] = (TP / row_sums[i] * 100) if row_sums[i] > 0 else np.nan

        # False Positive Rate (FPR)
        metrics['fpr'][i] = (FP / (FP + TN) * 100) if (FP + TN) > 0 else np.nan

        # False Negative Rate (FNR = 1 - Recall)
        metrics['fnr'][i] = (FN / row_sums[i] * 100) if row_sums[i] > 0 else np.nan

        # Precisión (Opcional)
        # metrics['precision'][i] = (TP / col_sums[i] * 100) if col_sums[i] > 0 else np.nan

    return metrics
import random
import numpy as np
import torch as t

# FUNCIÓN PARA FIJAR TODAS LAS SEMILLAS
def set_seed(seed=42):
    """
    Fija una semilla para todas las librerías relevantes (Python, NumPy, PyTorch)
    para asegurar que los resultados sean reproducibles.

    Parámetros:
        seed (int): valor de la semilla (por defecto 42).
    """
    random.seed(seed)  # Semilla para operaciones aleatorias de Python
    np.random.seed(seed)  # Semilla para NumPy
    t.manual_seed(seed)  # Semilla para PyTorch en CPU
    t.cuda.manual_seed_all(seed)  # Semilla para PyTorch en todas las GPUs

    # Configuración para que PyTorch en GPU sea determinista
    t.backends.cudnn.deterministic = True  # Hace que los algoritmos sean deterministas
    t.backends.cudnn.benchmark = False  # Desactiva la búsqueda automática de la configuración más rápida
    # (prioriza reproducibilidad sobre velocidad)


def load_checkpoint(checkpoint_path, model, optimizer,
                    scheduler, early_stopper, device):
    """
    Carga un checkpoint guardado previamente y restaura:
    - El estado del modelo
    - El estado del optimizador
    - El estado del scheduler
    - El estado del early stopping
    - Las métricas previas de entrenamiento

    Parámetros:
        checkpoint_path (str): ruta del archivo .pth o .pt con el checkpoint.
        model (torch.nn.Module): modelo a restaurar.
        optimizer (torch.optim.Optimizer): optimizador a restaurar.
        scheduler (torch.optim.lr_scheduler): scheduler a restaurar.
        early_stopper (EarlyStopping): objeto de early stopping.
        device (torch.device): CPU o GPU donde cargar el modelo.

    Retorna:
        model, optimizer, start_epoch, train_losses, val_losses,
        mean_ious, pixel_accuracies, early_stopper, best_val_loss
    """
    print(f"Cargando checkpoint desde: {checkpoint_path}")
    checkpoint = t.load(checkpoint_path, map_location=device, weights_only=False)

    # Restaurar modelo y optimizador
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)  # Importante para que esté en el mismo dispositivo que el entrenamiento
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Restaurar scheduler
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print("Usando scheduler: CosineAnnealing")

    # Restaurar métricas guardadas
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    mean_ious = checkpoint.get('mean_ious', [])
    pixel_accuracies = checkpoint.get('pixel_accuracies', [])

    # Restaurar la época desde la que continuar
    start_epoch = checkpoint['epoch'] + 1
    print(f"Reanudar desde la época: {start_epoch}")

    # Restaurar estado del early stopping
    if 'early_stopper_state' in checkpoint:
        early_stopper.__dict__.update(checkpoint['early_stopper_state'])
        print("Estado de EarlyStopping restaurado.")
    else:
        print("No se encontró early_stopper_state. Se inicia desde cero.")
        early_stopper.counter = 0
        early_stopper.best_loss = float("inf")
        early_stopper.early_stop = False

    # Mostrar tasa de aprendizaje restaurada
    for group in optimizer.param_groups:
        print("LR restaurado:", group['lr'])

    best_val_loss = checkpoint['best_val_loss']

    return model, optimizer, start_epoch, train_losses, val_losses, mean_ious, pixel_accuracies, early_stopper, best_val_loss


class EarlyStopping:
    """
    Implementa el mecanismo de early stopping: detiene el entrenamiento
    si no mejora la métrica seleccionada tras un número de épocas (patience).

    Atributos:
        patience (int): número de épocas sin mejora antes de parar.
        min_delta (float): mejora mínima para considerarlo una mejora real.
        mode (str): "max" para métricas que queremos maximizar (ej. IoU),
                    "min" para métricas que queremos minimizar (ej. loss).
    """

    def __init__(self, patience=5, min_delta=1e-4, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode

    def __call__(self, current_score):
        """
        Evalúa si el entrenamiento debe parar según la métrica actual.

        Parámetros:
            current_score (float): valor de la métrica en la época actual.
        """
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
    """
    Calcula porcentajes de aciertos, falsos positivos y falsos negativos
    para cada clase a partir de una matriz de confusión.

    Parámetros:
        cm (ndarray): matriz de confusión.

    Retorna:
        correct (ndarray): % de aciertos por clase.
        false_positives (ndarray): % de falsos positivos por clase.
        false_negatives (ndarray): % de falsos negativos por clase.
    """
    row_sums = cm.sum(axis=1)  # Total de ejemplos reales por clase
    col_sums = cm.sum(axis=0)  # Total de predicciones por clase

    # Inicializar arrays de resultados
    correct = np.zeros(len(cm))
    false_positives = np.zeros(len(cm))
    false_negatives = np.zeros(len(cm))

    for i in range(len(cm)):
        if row_sums[i] == 0:  # Si no hay ejemplos reales de esa clase
            correct[i] = np.nan
            false_positives[i] = np.nan
            false_negatives[i] = np.nan
        else:
            correct[i] = cm[i, i] / row_sums[i] * 100
            false_positives[i] = (col_sums[i] - cm[i, i]) / col_sums[i] * 100
            false_negatives[i] = (row_sums[i] - cm[i, i]) / row_sums[i] * 100

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
import config
import visualization
import torch as t
from torch import nn
import torch.nn.functional as fnn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import math


def train_loop(train_dataloader, model, loss_fn, optimizer, device):
    """
    Ejecuta una época de entrenamiento sobre los datos.

    Args:
        train_dataloader (DataLoader): DataLoader para el conjunto de entrenamiento.
        model (nn.Module): Modelo a entrenar.
        loss_fn (func): Función de pérdida.
        optimizer (Optimizer): Optimizador.
        device (torch.device): Dispositivo (CPU/GPU).

    Returns:
        float: Pérdida promedio durante la época.
    """
    running_loss = 0.0  # Para llevar el seguimiento de la pérdida
    batch_num = len(train_dataloader)  # Número total de batches
    train_dataset_size = len(train_dataloader.dataset)  # Número total de imágenes en el dataset de entrenamiento
    batch_size = train_dataloader.batch_size  # Tamaño del batch
    model.train()

    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        # X = (batch_size, canales, alto, ancho)
        # y = (batch_size, alto, ancho)

        # Compute prediction and loss

        assert y.max() < config.NUM_CLASSES, f"Valor fuera de rango en la máscara: {y.max()}" #.max() mira el valor maximo dentro del tensor

        pred = model(X)
        y = fnn.interpolate(y.unsqueeze(1).float(), size=pred.shape[2:], mode="nearest").squeeze(1).long() # redimensiona las máscaras y para que tengan el mismo tamaño que la predicción pred
        # y.unsqueeze(1) : (batch_size, H, W) pasa a (batch_size, 1, H, W).

        loss = loss_fn(pred, y)

        if loss.item() > 15.0:
            print(f"Loss demasiado alta: {loss.item()}")
            sys.exit()  # Detiene todito el script

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

        # Imprimir datos importantes
        if batch % 100 == 0 or batch == batch_num - 1:
            avg_loss = running_loss / (batch + 1)
            current_processed_data = min((batch + 1) * batch_size, train_dataset_size)
            print(f"Batch [{batch+1} / {batch_num}], Loss: {avg_loss:.4f} \n")
            print(f"[{current_processed_data}/{train_dataset_size}] \n")

    return (running_loss / batch_num)


def test_loop(val_dataloader, model, loss_fn, device, use_tta=False):
    """
    Ejecuta la evaluación del modelo en un conjunto de validación.

    Args:
        val_dataloader (DataLoader): DataLoader para validación.
        model (nn.Module): Modelo a evaluar.
        loss_fn (func): Función de pérdida.
        device (torch.device): Dispositivo (CPU/GPU).
        use_tta (bool): Flag para usar test-time augmentation (no implementado aquí).

    Returns:
        tuple:
            - avg_val_loss (float): Pérdida promedio en validación.
            - pixel_percentage (float): Precisión a nivel de píxel.
            - mean_iou (list): Lista con IoU promedio por clase.
    """
    val_loss = 0.0
    total_pixels, correct_pixels = 0, 0
    batch_num = len(val_dataloader)
    val_dataset_size = len(val_dataloader.dataset)
    batch_size = val_dataloader.batch_size
    iou_per_epoch = t.zeros(config.NUM_CLASSES)
    model.eval()

    with t.no_grad():
        for batch, (X, y) in enumerate(val_dataloader):
            X, y = X.to(device), y.to(device)

            preds = []
            for img in X:
                pred = model(img.unsqueeze(0).to(device)).squeeze(0)
                preds.append(pred)

            pred = t.stack(preds)  # (B, C, H, W)

            if batch == 0:
                cm_batch = compute_confusion_matrix(pred, y)
                visualization.show_confusion_matrix(cm_batch)
                # Mostrar y guardar comparación de máscara
                fig_pred = visualization.show_image_comparison(X[0].cpu(), pred[0].cpu(), y[0].cpu())
                pred_path = os.path.join(config.OUTPUT_DIR, f"comparison_batch_{batch}.png")
                fig_pred.savefig(pred_path)
                plt.close(fig_pred)
                print(f"Comparación visual guardada como {pred_path}")

            y = fnn.interpolate(y.unsqueeze(1).float(), size=pred.shape[2:], mode="nearest").squeeze(1).long()
            loss = loss_fn(pred, y)

            val_loss += loss.item()
            correct_pixels += (pred.argmax(1) == y).type(t.float).sum().item()
            total_pixels += y.numel()

            iou = compute_iou(pred, y)
            iou_per_epoch += t.nan_to_num(iou, nan=0.0)

            if batch % 50 == 0 or batch == batch_num - 1:
                pixel_percentage = correct_pixels / total_pixels
                avg_loss = val_loss / (batch + 1)
                print(f"Validation Batch [{batch+1}/{batch_num}], Loss: {avg_loss:.4f}")
                print(f"Processed: [{min((batch + 1) * batch_size, val_dataset_size)}/{val_dataset_size}]")
                if ((y == 7).any() or (y == 8).any() or (y == 11).any() or (y == 12).any()):
                    print(f"Mostrando predicción de un batch con clase minoritaria (batch {batch})")
                    fig = visualization.show_image_comparison(X[0].cpu(), pred[0].cpu(), y[0].cpu())
                    file_name = f"prediccion_batch_{batch}.png"
                    file_path = os.path.join(config.OUTPUT_DIR, file_name)
                    fig.savefig(file_path)  # guardamos directamente desde el objeto fig
                    plt.close(fig)
                    print(f"Imagen guardada como {file_path}")

    avg_val_loss = val_loss / batch_num
    pixel_percentage = correct_pixels / total_pixels
    mean_iou = (iou_per_epoch / batch_num).tolist()

    print("\n--- Evaluation ---")
    print(f"Average Validation Loss: {avg_val_loss:.4f}")
    print(f"Pixel Accuracy: {pixel_percentage:.4f}")
    print("\nMean IoU por clase:")
    for idx, iou_value in enumerate(mean_iou):
        print(f"Clase {idx:2d}: IoU = {iou_value:.4f}")

    return avg_val_loss, pixel_percentage, mean_iou


class FocalLoss(nn.Module):
    def __init__(self, device, alpha=None, gamma=2.0, reduction='mean'):
        """
        alpha: tensor con pesos por clase o valor escalar (si se quiere dar más peso global)
        gamma: parámetro de enfoque, típico entre 1 y 5
        reduction: 'mean', 'sum' o 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: logits del modelo, tamaño (N, C, H, W)
        targets: etiquetas, tamaño (N, H, W)
        """
        log_probs = fnn.log_softmax(inputs, dim=1)  # Log-Probabilidades
        probs = t.exp(log_probs)              # Probabilidades reales
        targets_one_hot = fnn.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2)

        ce_loss = -targets_one_hot * log_probs  # CrossEntropy clásica
        focal_loss = ((1 - probs) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.view(1, -1, 1, 1).to(inputs.device)
            focal_loss *= alpha


        focal_loss = focal_loss.sum(dim=1)  # Sumamos sobre canales (C)
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

    def set_gamma(self, new_gamma):
      self.gamma = new_gamma


def cosine_gamma(epoch, max_epochs, gamma_min=1.5, gamma_max=3.0):
    """
    Devuelve un gamma entre gamma_min y gamma_max usando Cosine Annealing.
    """
    cos_inner = math.pi * epoch / max_epochs # calculates the inner term of the cosine function
    return gamma_min + 0.5 * (gamma_max - gamma_min) * (1 - math.cos(cos_inner))


def compute_confusion_matrix_incremental(model, dataloader, device, num_classes):
    """
    Calcula la matriz de confusión acumulativa sin almacenar todas las predicciones en memoria.
    """
    matriz_conf = np.zeros((num_classes, num_classes), dtype=int)
    model.eval()

    with t.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)  # (B, H, W)

            pred = model(X)  # (B, C, H, W)
            pred_labels = pred.argmax(dim=1)  # (B, H, W)

            # Asegurar que `y` tenga el mismo tamaño que `pred`
            y = y.unsqueeze(1).float()  # (B, 1, H, W)
            y = fnn.interpolate(y, size=pred_labels.shape[1:], mode="nearest")  # (B, 1, H, W)
            y = y.squeeze(1).long()  # (B, H, W)

            # Flatten all para calcular la matriz batch a batch
            y_np = y.cpu().numpy().flatten()
            pred_np = pred_labels.cpu().numpy().flatten()

            assert y_np.shape == pred_np.shape, f"Shape mismatch: y={y_np.shape}, pred={pred_np.shape}"

            matriz_conf += confusion_matrix(y_np, pred_np, labels=range(num_classes))

    return matriz_conf

#Función que imprime la matriz de confusión dados un batch de predicciones y de imagenes
def compute_confusion_matrix(pred, y_true):
    pred = pred.argmax(dim=1).cpu().numpy()
    y_true = y_true.unsqueeze(1)
    y_true = fnn.interpolate(y_true.float(), size=pred.shape[1:], mode="nearest")
    y_true = y_true.squeeze(1).long()
    y_true = y_true.cpu().numpy()
    cm = confusion_matrix(y_true.flatten(), pred.flatten(), labels=range(13))
    return cm


def compute_iou(y_pred, y_true, num_classes=config.NUM_CLASSES):
    """
    Versión mejorada que:
    1. Ignora completamente las clases no presentes
    2. Maneja explícitamente casos edge
    3. Devuelve métricas más informativas
    """
    iou_per_class = {}  # Diccionario para guardar IoUs por clase
    present_classes = []  # Lista para las clases presentes

    y_pred = y_pred.argmax(dim=1)  # Predicciones
    y_true = y_true.squeeze(1)  # Ground truth

    for cls in range(num_classes):
        # Solo calcular IoU si la clase está presente en el ground truth
        if (y_true == cls).any():  # Si hay píxeles de esta clase en y_true
            pred_inds = (y_pred == cls)
            target_inds = (y_true == cls)

            intersection = (pred_inds & target_inds).sum().float()
            union = (pred_inds | target_inds).sum().float()

            if union > 0:  # Si hay intersección y unión
                iou = (intersection / union).item()
                iou_per_class[cls] = iou  # Guardamos el IoU para esta clase
                present_classes.append(cls)  # Añadimos a las clases presentes
            else:
                iou_per_class[cls] = float('nan')  # No se calcula el IoU para clases sin píxeles

    # Calcular mIoU solo sobre clases que tienen IoU válido
    valid_ious = [iou for iou in iou_per_class.values() if not np.isnan(iou)]
    mean_iou = np.mean(valid_ious) if valid_ious else float('nan')

    return iou_per_class, mean_iou, present_classes
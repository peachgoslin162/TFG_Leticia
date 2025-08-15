import util
import dataset
import model
import train
import visualization
import config
import torch as t
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.colors as mcolors
from torch.optim.lr_scheduler import CosineAnnealingLR
import time


def main():
    start_time = time.time()
    os.makedirs(config.METRICS_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    num_gpus = t.cuda.device_count()
    num_cpus = os.cpu_count()
    util.set_seed(config.SEED)
    normal_transform = dataset.TransformImages(dataset.normal_augmentations)
    special_transform = dataset.TransformImages(dataset.special_augmentations)
    processed_dataset = dataset.CustomImageDataset(dataset_path=config.DATASET_PATH, normal_transform=normal_transform,
                                 special_transform=special_transform)
    dataset_size = len(processed_dataset)

    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(processed_dataset, [train_size, val_size, test_size])

    g = t.Generator()

    g.manual_seed(config.SEED)

    if not t.cuda.is_available():
        train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=min(4, num_gpus),
                                      generator=g)
        val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=min(4, num_cpus), generator=g)
        test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=min(4, num_cpus),
                                     generator=g)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=min(4, num_gpus),
                                      generator=g)
        val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=min(4, num_gpus), generator=g)
        test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=min(4, num_gpus),
                                     generator=g)

    new_model = model.UNet().to(device)

    start_epoch = config.START_EPOCH

    total_epochs = config.TOTAL_EPOCHS

    train_losses = []

    val_losses = []

    pixel_accuracies = []

    mean_ious = []

    best_val_loss = float('inf')

    optimizer = t.optim.Adam(new_model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.TOTAL_EPOCHS,
        eta_min=1e-6
    )

    early_stopper = util.EarlyStopping(
        patience=8,
        min_delta=1e-3,
        mode="min"  # para val_loss
    )

    if config.START_FROM_CHECKPOINT:
        new_model, optimizer, start_epoch, train_losses, val_losses, mean_ious, pixel_accuracies, early_stopper, best_val_loss = util.load_checkpoint(config.LAST_CHECKPOINT_PATH, new_model, optimizer,
                               scheduler, early_stopper, device)
    else:
        #Código para sacar el gráfico de barras para saber cuanto aparece cada clase

        # Contador de píxeles por clase (train + val + test)
        label_pixel_counts = defaultdict(int)

        for subset in [train_dataset, val_dataset, test_dataset]:
            for _, mask in subset:
                unique_labels = t.unique(mask)
                for label in unique_labels:
                    label = int(label.item())
                    label_pixel_counts[label] += t.sum(mask == label).item()

        # Ordenar resultados
        sorted_items = sorted(label_pixel_counts.items(), key=lambda x: x[1])
        ids, values = zip(*sorted_items)
        names = [config.CLASS_NAME_MAPPING.get(i, f"Class {i}") for i in ids]

        # Normalizamos los valores para aplicar el colormap
        values_array = np.array(values)
        norm = plt.Normalize(values_array.min(), values_array.max())
        colors = [config.NORMALIZED_COLORS[i] for i in ids]

        # Dibujar gráfico de barras
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(names, values, color=colors)

        # Añadir los valores a cada barra
        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + 1e6, bar.get_y() + bar.get_height() / 2, str(value), va='center')

        # Títulos y etiquetas
        ax.set_title("Pixels per label (Train + Val + Test)")
        ax.set_xlabel("Number of pixels")
        ax.set_ylabel("Label")
        ax.grid(axis='x', linestyle='--', alpha=0.7)

        # Crear colorbar y asociarla con el eje actual

        cmap = mcolors.ListedColormap(config.NORMALIZED_COLORS)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Pixel Count')

        plt.tight_layout()

        if config.OUTPUT_DIR:
            full_path = os.path.join(config.OUTPUT_DIR, "visualizacion.png")
            plt.savefig(full_path, bbox_inches='tight')
            print(f"Imagen guardada en: {full_path}")
        plt.show()
        plt.close()



    loss_fn = train.FocalLoss(device, alpha=config.ALPHA, gamma=config.GAMMA)

    if t.cuda.is_available():
        t.cuda.empty_cache()



    # EMPIEZA EL ENTRENAMIENTO, VALIDACIÓN Y FINALMENTE TEST

    for epoch in range(start_epoch, total_epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")

        train_dataset.dataset.set_transform_true(normal_transform, special_transform)

        '''
        train_dataset = Un subconjunto (Subset) de otro dataset, con .dataset se accede a el dataset "base".
        '''

        new_gamma = train.cosine_gamma(epoch, total_epochs, gamma_min=1.5, gamma_max=2.5)

        loss_fn.set_gamma(new_gamma)
        print(f"[Epoch {epoch + 1}] Gamma actualizado a: {new_gamma:.4f}")

        # Mostrar el valor actual del learning rate
        current_lrs = [group['lr'] for group in optimizer.param_groups]
        print(f"[Epoch {epoch + 1}] LR actual: {current_lrs}")

        avg_train_loss = train.train_loop(train_dataloader, new_model, loss_fn, optimizer, device)

        val_dataset.dataset.set_transform_none()
        avg_val_loss, pixel_acc, mean_iou = train.test_loop(val_dataloader, new_model, loss_fn, device)

        # ACTUALIZA SCHEDULER SEGÚN EPOCH
        scheduler.step(epoch)

        early_stopper(avg_val_loss)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        pixel_accuracies.append(pixel_acc)
        mean_ious.append(mean_iou)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            t.save({
                'epoch': epoch,
                'model_state_dict': new_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'mean_ious': mean_ious,
                'pixel_accuracies': pixel_accuracies,
                'early_stopper_state': early_stopper.__dict__,
                'best_val_loss': best_val_loss
            }, os.path.join(config.OUTPUT_DIR, f'best_checkpoint_epoch_{epoch + 1}.pth'))
            print("Checkpoint actualizado: best_checkpoint.pth")
            if epoch >= 5:
                visualization.plot_metrics(train_losses, val_losses, pixel_accuracies, mean_ious)
        if early_stopper.early_stop:
            print("Ninguna mejora aparente: early stopping activado.")
            break

        current_time = time.time()

        elapsed_time = current_time - start_time

        if elapsed_time >= config.SECONDS_LIMIT:
            print("Tiempo límite alcanzado. Guardando modelo antes de que Kaggle corte.")
            t.save({
                'epoch': epoch,
                'model_state_dict': new_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'mean_ious': mean_ious,
                'pixel_accuracies': pixel_accuracies,
                'early_stopper_state': early_stopper.__dict__,
                'best_val_loss': best_val_loss
            }, os.path.join(config.OUTPUT_DIR, f'auto_interrupt_checkpoint_epoch_{epoch + 1}.pth'))
            print("Checkpoint guardado automáticamente antes del corte.")
            visualization.plot_metrics(train_losses, val_losses, pixel_accuracies, mean_ious)
            break

    print("Entrenamiento acabado, guardando modelo...")
    t.save({
        'model_state_dict': new_model.state_dict(),
    }, os.path.join(config.OUTPUT_DIR, f'last_model.pth'))
    visualization.plot_metrics(train_losses, val_losses, pixel_accuracies, mean_ious)
    print(f"Modelo guardado!")

    print('Evaluación final!')

    test_dataset.dataset.set_transform_none()

    avg_val_loss, pixel_acc, mean_iou = train.test_loop(test_dataloader, new_model, loss_fn, device)

    cm = train.compute_confusion_matrix_incremental(model, test_dataloader, device, config.NUM_CLASSES)

    visualization.show_confusion_matrix(cm)

    '''
    # Calcular los porcentajes
    correct, false_positives, false_negatives = util.calculate_percentages(cm)

    # Imprimir los resultados para cada clase
    for i in range(len(correct)):
        print(f"Clase {i}:")
        print(f"  - Predicción correcta: {correct[i]:.2f}%")
        print(f"  - Falsos positivos: {false_positives[i]:.2f}%")
        print(f"  - Falsos negativos: {false_negatives[i]:.2f}%\n")
        print()

    # confusion_details(cm)
    '''

    metrics = util.calculate_confusion_metrics(cm)

    # Imprimir resultados
    for i in range(len(cm)):
        print(f"Clase {i}:")
        print(f"- Recall: {metrics['recall'][i]:.2f}%")
        print(f"- FPR: {metrics['fpr'][i]:.2f}%")
        print(f"- FNR: {metrics['fnr'][i]:.2f}%")
        print()

    print(f"Test Loss: {avg_val_loss}")
    print(f"Test Pixel Accuracy: {pixel_acc}")

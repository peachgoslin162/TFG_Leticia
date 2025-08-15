import os
import torch as t
import matplotlib.pyplot as plt
import seaborn as sns
import config
import matplotlib.colors as mcolors


def plot_metrics(train_losses, val_losses, pixel_accuracies, mean_ious):
    epochs_range = range(1, len(train_losses) + 1)

    # --- Gráfico de pérdidas ---
    fig1 = plt.figure(figsize=(15, 8))
    plt.plot(epochs_range, train_losses, label="Training Loss")
    plt.plot(epochs_range, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    loss_path = os.path.join(config.METRICS_DIR, f"losses.png")
    fig1.savefig(loss_path)
    print(f"Gráfico de pérdidas guardado: {loss_path}")
    plt.close(fig1)

    # --- Gráfico de pixel accuracy ---
    fig2 = plt.figure(figsize=(15, 8))
    plt.plot(epochs_range, pixel_accuracies, label="Pixel Accuracy", color="green")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Pixel Accuracy Over Epochs")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    acc_path = os.path.join(config.METRICS_DIR, f"pixel_accuracy.png")
    fig2.savefig(acc_path)
    print(f"Gráfico de precisión guardado: {acc_path}")
    plt.close(fig2)

    # --- Gráfico de IoU ---
    fig3 = plt.figure(figsize=(15, 8))
    plt.plot(epochs_range, mean_ious, label="Mean IoU", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("IoU")
    plt.title("Mean IoU Over Epochs")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    iou_path = os.path.join(config.METRICS_DIR, f"mean_iou.png")
    fig3.savefig(iou_path)
    print(f"Gráfico de IoU guardado: {iou_path}")
    plt.close(fig3)

def show_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False) #sns = seaborn
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout() #This ensures that all titles and labels are properly spaced without overlapping.
    plt.show()
    plt.close()

'''
cm: This is the data being visualized, typically a 2D array or DataFrame (e.g., a confusion matrix).
annot=True: Displays the values of the cells directly on the heatmap, annot=False → Solo verías la escala de colores, sin texto numérico.
fmt="d": Formats the annotations as integers (useful for whole numbers like counts).
cmap="Blues": Sets the color map to shades of blue.
cbar=False: Disables the color bar on the side of the heatmap.
'''

def show_image_comparison(image, pred_mask, true_mask):
    cmap = mcolors.ListedColormap(config.NORMALIZED_COLORS)
    fig, ax = plt.subplots(1, 3, figsize=(15, 8))

    ax[0].imshow(image.cpu().permute(1, 2, 0))
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    pred_mask = pred_mask.argmax(dim=0) # Cuando quieres convertir de probabilidades por clase a clase más probable, usas .argmax() sobre el eje de las clases (C).
    ax[1].imshow(pred_mask.cpu(), cmap=cmap, vmin=0, vmax=config.NUM_CLASSES - 1)
    ax[1].set_title('Predicted Mask')
    ax[1].axis('off')

    ax[2].imshow(true_mask.cpu(), cmap=cmap, vmin=0, vmax=config.NUM_CLASSES - 1)
    ax[2].set_title('Ground Truth Mask')
    ax[2].axis('off')


    # Para crear y mostrar una leyenda personalizada en el gráfico
    unique_classes = t.unique(true_mask.cpu())
    labels = [config.CLASS_NAME_MAPPING[i.item()] for i in unique_classes]
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=10) for i in unique_classes]
    ax[2].legend(handles, labels, title="Clases", loc="upper right", bbox_to_anchor=(1.65, 1))

    '''
    vmin y vmax en imshow de Matplotlib son los valores mínimo y máximo del rango de colores que se van a mapear al colormap (cmap).
    '''
    plt.tight_layout()
    plt.show()  # Muestra en pantalla
    return fig  # Devuelve el objeto para guardarlo

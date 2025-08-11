import sys
import os
import torch as t


#Bucle de entrenamiento para 1 epoch
def train_loop(train_dataloader, model, loss_fn, optimizer, device):
    running_loss = 0.0  # Para llevar el seguimiento de la pérdida
    batch_num = len(train_dataloader) #Número total de batches
    train_dataset_size = len(train_dataloader.dataset) #Número total de imagenes en el dataset de entrenamiento
    batch_size = train_dataloader.batch_size #Tamaño del batch
    model.train()


    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction and loss

        assert y.max() < num_classes, f"Valor fuera de rango en la máscara: {y.max()}"
        pred = model(X)
        y = fnn.interpolate(y.unsqueeze(1).float(), size=pred.shape[2:], mode="nearest").squeeze(1).long()
        loss = loss_fn(pred, y)

        if loss.item() > 15.0:
            print(f"Loss demasiado alta: {loss.item()}")
            sys.exit()  # Detiene todo el script


        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

        #print important data
        if batch % 100 == 0 or batch == batch_num - 1:
            avg_loss = running_loss / (batch + 1)
            current_processed_data = min((batch + 1) * batch_size, train_dataset_size)
            print(f"Batch [{batch+1} / {batch_num}], Loss: {avg_loss:.4f} \n")
            print(f"[{current_processed_data}/{train_dataset_size}] \n")

    return (running_loss / batch_num)


def test_loop(val_dataloader, model, loss_fn, device, use_tta=False):
    val_loss = 0.0
    total_pixels, correct_pixels = 0, 0
    batch_num = len(val_dataloader)
    val_dataset_size = len(val_dataloader.dataset)
    batch_size = val_dataloader.batch_size
    num_classes = 13
    iou_per_epoch = t.zeros(num_classes)
    has_minority = False
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
                show_confusion_matrix(cm_batch)
                # Mostrar y guardar comparación de máscara
                fig_pred = show_image_comparison(X[0].cpu(), pred[0].cpu(), y[0].cpu())
                pred_path = os.path.join(output_path, f"comparison_batch_{batch}.png")
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
                    fig = show_image_comparison(X[0].cpu(), pred[0].cpu(), y[0].cpu())
                    file_name = f"prediccion_batch_{batch}.png"
                    file_path = os.path.join(output_path, file_name)
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



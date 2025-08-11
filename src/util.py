import config
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


def cargar_checkpoint_completo(checkpoint_path=last_checkpoint_path, model=model, optimizer=optimizer,
                               cosine_scheduler=cosine_scheduler, early_stopper=early_stopper,
                               best_val_loss=best_val_loss):
    print(f"Cargando checkpoint desde: {checkpoint_path}")
    checkpoint = t.load(checkpoint_path, map_location=dispositivo, weights_only=False)

    # Modelo y optimizador
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(dispositivo)  # Muy importante
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

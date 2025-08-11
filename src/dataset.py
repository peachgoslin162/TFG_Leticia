import torch as t
from torch.utils.data import Dataset
from torchvision.io import decode_image
import torchvision.transforms as transforms
from torchvision.transforms import v2, InterpolationMode, Resize
import torchvision.transforms.functional as F
import random
import kornia.augmentation as K
import os

# FUNCIONES AUXILIARES PARA EL PREPROCESAMIENTO

#Devuelve la bounding box que contiene everything excepto el fondo (clase 0).
#Añade un pequeño margen para no cortar justo al borde.
def get_foreground_bbox(mask, margin=10):
    mask_copy = mask.clone()

    mask_copy = format_mask(mask_copy)

    foreground = mask_copy != 0

    if foreground.sum() == 0:
        return None  # Imagen totalmente negra

    coords = foreground.nonzero(as_tuple=False)
    top = max(coords[:, 0].min().item() - margin, 0)
    bottom = min(coords[:, 0].max().item() + margin, mask.shape[1] - 1)
    left = max(coords[:, 1].min().item() - margin, 0)
    right = min(coords[:, 1].max().item() + margin, mask.shape[2] - 1)

    # Validación para evitar errores
    if bottom <= top or right <= left:
        return None

    return top, left, bottom, right

def crop_foreground_and_resize(image, mask, output_size=(360, 640)):
    bbox = get_foreground_bbox(mask)
    if bbox is None:
        image = transforms.Resize(output_size, interpolation=InterpolationMode.BILINEAR)(image)
        mask = transforms.Resize(output_size, interpolation=InterpolationMode.NEAREST)(mask)
        return image, mask

    top, left, bottom, right = bbox
    image_cropped = image[:, top:bottom+1, left:right+1]
    mask_cropped = mask[:, top:bottom+1, left:right+1]

    image_resized = transforms.Resize(output_size, interpolation=InterpolationMode.BILINEAR)(image_cropped)
    mask_resized = transforms.Resize(output_size, interpolation=InterpolationMode.NEAREST)(mask_cropped)

    return image_resized, mask_resized

#Función que convierte la máscara watershed con id de clases basados en su RGB a una máscara de clases (0-12)
def format_mask(mask):
    mask = mask.permute(1, 2, 0)[:, :, 0]

    class_mask = t.zeros((mask.shape[0], mask.shape[1]), dtype=t.long)

    for color_value, class_index in id_to_class.items():
        class_mask[mask == color_value] = class_index

    return class_mask

def maybe_apply(transform_fn, p=0.5):
    def wrapped(image, mask):
        if random.random() < p:
            return transform_fn(image, mask)
        return image, mask
    wrapped.__name__ = transform_fn.__name__
    return wrapped

#Devuelve la bounding box [top, left, bottom, right] que contiene las clases dadas.
def get_bounding_box(mask, target_classes = [7,8,11,12]):
    indices = t.zeros_like(mask, dtype=t.bool)
    for cls in target_classes:
        indices |= (mask == cls)

    if indices.sum() == 0:
        return None  # No hay píxeles de esas clases

    coords = indices.nonzero(as_tuple=False)  # (N, 2): filas y columnas
    top = coords[:, 0].min().item()
    bottom = coords[:, 0].max().item()
    left = coords[:, 1].min().item()
    right = coords[:, 1].max().item()

    return top, left, bottom, right


#Reduce la imagen y la máscara a un 75% de su tamaño original
def resize_quarter(image, mask):
    orig_height, orig_width = image.shape[1], image.shape[2] #Shape: (C, H, W)

    new_height = int(orig_height * (3/4))
    new_width = int(orig_width * (3/4))

    resize_transform_img = Resize((new_height, new_width), interpolation=InterpolationMode.BILINEAR)
    image = resize_transform_img(image)

    resize_transform_mask = Resize((new_height, new_width), interpolation=InterpolationMode.NEAREST)
    mask = resize_transform_mask(mask)

    return image, mask

# Clase personalizada para aplicar las transformaciones a imagen y máscara
class TransformImages:
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, image, mask):
        applied_transforms = []  # Aquí guardamos el nombre de las transformaciones
        for transform in self.augmentations:
            transform_name = transform.__name__
            image, mask = transform(image, mask)
            applied_transforms.append(transform_name)
        return image, mask, applied_transforms

# Las transformaciones deben ser definidas manualmente, ya que necesitamos
# aplicar las mismas transformaciones a la imagen y la máscara.
def random_rotation(image, mask, max_angle=25):
    angle = random.uniform(-max_angle, max_angle)
    image = transforms.functional.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
    mask = transforms.functional.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
    '''
    if random.random() < 0.1:
      print(f"Estoy en: {random_rotation.__name__}")
      visualize_with_legend(image, format_mask(mask.clone()))
    '''
    return image, mask

def random_horizontal_flip(image, mask):
    image = transforms.functional.hflip(image)
    mask = transforms.functional.hflip(mask)
    '''
    if random.random() < 0.1:
      print(f"Estoy en: {random_horizontal_flip.__name__}")
      visualize_with_legend(image, format_mask(mask.clone()))
    '''
    return image, mask

def random_vertical_flip(image, mask):
    image = transforms.functional.vflip(image)
    mask = transforms.functional.vflip(mask)
    '''
    if random.random() < 0.1:
      print(f"Estoy en: {random_vertical_flip.__name__}")
      visualize_with_legend(image, format_mask(mask.clone()))
    '''
    return image, mask

# Función para aplicar un random crop con el mismo tamaño a la imagen y la máscara
def random_crop(image, mask, crop_size=[320,320,3]):
    # Obtener la altura y el ancho de la imagen
    height, width = image.shape[1], image.shape[2]

    # Generar los límites del crop aleatorio
    top = random.randint(0, height - crop_size[0])
    left = random.randint(0, width - crop_size[1])

    # Aplicar el mismo crop tanto a la imagen como a la máscara
    image_cropped = image[:, top:top + crop_size[0], left:left + crop_size[1]]
    mask_cropped = mask[:, top:top + crop_size[0], left:left + crop_size[1]]

    # Asegurarnos de que las dimensiones de la imagen y la máscara sean consistentes
    image_cropped = transforms.Resize((image.shape[1], image.shape[2]), interpolation=InterpolationMode.BILINEAR)(image_cropped)
    mask_cropped = transforms.Resize((image.shape[1], image.shape[2]), interpolation=InterpolationMode.NEAREST)(mask_cropped)
    '''
    if random.random() < 0.1:
      print(f"Estoy en: {random_crop.__name__}")
      visualize_with_legend(image_cropped, format_mask(mask_cropped.clone()))
    '''
    return image_cropped, mask_cropped

def strong_random_crop(image, mask, crop_size=[192,192,3]):
    # Obtener la altura y el ancho de la imagen
    height, width = image.shape[1], image.shape[2]

    # Generar los límites del crop aleatorio
    top = random.randint(0, height - crop_size[0])
    left = random.randint(0, width - crop_size[1])

    # Aplicar el mismo crop tanto a la imagen como a la máscara
    image_cropped = image[:, top:top + crop_size[0], left:left + crop_size[1]]
    mask_cropped = mask[:, top:top + crop_size[0], left:left + crop_size[1]]

    # Asegurarnos de que las dimensiones de la imagen y la máscara sean consistentes
    image_cropped = transforms.Resize((image.shape[1], image.shape[2]), interpolation=InterpolationMode.BILINEAR)(image_cropped)
    mask_cropped = transforms.Resize((image.shape[1], image.shape[2]), interpolation=InterpolationMode.NEAREST)(mask_cropped)
    '''
    if random.random() < 0.1:
      print(f"Estoy en: {strong_random_crop.__name__}")
      visualize_with_legend(image_cropped, format_mask(mask_cropped.clone()))
    '''
    return image_cropped, mask_cropped


def crop_around_bbox(image, mask, crop_size=(360, 640)):

    bbox = get_bounding_box(mask, [7,8,11,12])
    top, left, bottom, right = bbox
    center_y = (top + bottom) // 2
    center_x = (left + right) // 2

    half_h = crop_size[0] // 2
    half_w = crop_size[1] // 2

    start_y = max(center_y - half_h, 0)
    start_x = max(center_x - half_w, 0)

    # Evitar que el crop se salga por abajo/derecha
    end_y = min(start_y + crop_size[0], image.shape[1])
    end_x = min(start_x + crop_size[1], image.shape[2])

    # Ajustar si el recorte es más pequeño por los bordes
    start_y = end_y - crop_size[0]
    start_x = end_x - crop_size[1]

    image_cropped = image[:, start_y:end_y, start_x:end_x]
    mask_cropped = mask[:, start_y:end_y, start_x:end_x]

    # Redimensionar al tamaño original
    image_cropped = transforms.Resize((image.shape[1], image.shape[2]), interpolation=InterpolationMode.BILINEAR)(image_cropped)
    mask_cropped = transforms.Resize((image.shape[1], image.shape[2]), interpolation=InterpolationMode.NEAREST)(mask_cropped)
    '''
    if random.random() < 0.1:
      print(f"Estoy en: {crop_around_bbox.__name__}")
      visualize_with_legend(image_cropped, format_mask(mask_cropped.clone()))
    '''
    return image_cropped, mask_cropped

def color_jitter(image, mask, brightness_range=(0.85, 1.1), contrast_range=(0.9, 1.15), saturation_range=(0.95, 1.05)):
    # Random values for brightness, contrast and saturation
    brightness = random.uniform(*brightness_range)
    contrast = random.uniform(*contrast_range)
    saturation = random.uniform(*saturation_range)

    image = F.adjust_brightness(image, brightness)
    image = F.adjust_contrast(image, contrast)
    image = F.adjust_saturation(image, saturation)
    '''
    if random.random() < 0.1:
      print(f"Estoy en: {color_jitter.__name__}")
      visualize_with_legend(image, format_mask(mask.clone()))
    '''
    return image, mask

def strong_color_jitter(image, mask, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3), saturation_range=(0.8, 1.2)):
    # Random values for brightness, contrast and saturation
    brightness = random.uniform(*brightness_range)
    contrast = random.uniform(*contrast_range)
    saturation = random.uniform(*saturation_range)

    image = F.adjust_brightness(image, brightness)
    image = F.adjust_contrast(image, contrast)
    image = F.adjust_saturation(image, saturation)
    '''
    if random.random() < 0.1:
      print(f"Estoy en: {strong_color_jitter.__name__}")
      visualize_with_legend(image, format_mask(mask.clone()))
    '''
    return image, mask

def random_affine(image, mask):
    # 🔧 Ángulo de rotación más limitado
    angle = random.uniform(-25, 25)

    # 🔧 Traslación más suave (5% del tamaño en lugar de 10%)
    max_dx = int(0.05 * image.shape[2])
    max_dy = int(0.05 * image.shape[1])
    translate = (random.randint(-max_dx, max_dx), random.randint(-max_dy, max_dy))

    # 🔧 Escala más cercana a 1 (menos deformación)
    scale = random.uniform(0.9, 1.1)

    image = F.affine(
        image, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR, fill=0
    )
    mask = F.affine(
        mask, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0],
        interpolation=InterpolationMode.NEAREST, fill=0
    )
    '''
    if random.random() < 0.1:
      print(f"Estoy en: {random_affine.__name__}")
      visualize_with_legend(image, format_mask(mask.clone()))
    '''
    return image, mask


def elastic_transform(image, mask, alpha=6.0, sigma=2.5):
    """
    Versión conservadora de elastic_transform.
    Pequeños desplazamientos para no dañar estructuras anatómicas.
    """

    if mask.ndim == 2:
        mask = mask.unsqueeze(0)  # Convertir a (1, H, W)

    stacked = t.cat([image, mask.float()], dim=0).unsqueeze(0)  # (1, C+1, H, W)

    elastic = K.RandomElasticTransform(
        alpha=t.tensor([alpha, alpha]),
        sigma=t.tensor([sigma, sigma]),
        p=1.0,
        same_on_batch=True
    )

    result = elastic(stacked)

    image_deformed = result[0, :3]
    mask_deformed = result[0, 3:].long()

    return image_deformed, mask_deformed



def random_resized_crop(image, mask, scale=(0.6, 1.0), ratio=(0.75, 1.33)):
    """
    Simula RandomResizedCrop manualmente para imagen y máscara.
    """
    _, H, W = image.shape
    area = H * W

    for _ in range(10):  # hasta 10 intentos de encontrar un crop válido
        target_area = random.uniform(*scale) * area
        aspect_ratio = random.uniform(*ratio)

        new_w = int(round((target_area * aspect_ratio) ** 0.5))
        new_h = int(round((target_area / aspect_ratio) ** 0.5))

        if new_w <= W and new_h <= H:
            top = random.randint(0, H - new_h)
            left = random.randint(0, W - new_w)

            image_cropped = image[:, top:top+new_h, left:left+new_w]
            mask_cropped = mask[:, top:top+new_h, left:left+new_w]

            # Redimensionar al tamaño original
            image_resized = F.resize(image_cropped, size=(H, W), interpolation=InterpolationMode.BILINEAR)
            mask_resized = F.resize(mask_cropped, size=(H, W), interpolation=InterpolationMode.NEAREST)

            return image_resized, mask_resized

    # fallback si no encuentra un crop válido
    return image, mask


special_augmentations = [
    crop_around_bbox,
    maybe_apply(random_affine, p=0.8),
    #maybe_apply(elastic_transform, p=0.1),
    maybe_apply(random_horizontal_flip, p=0.5),
    maybe_apply(random_vertical_flip, p=0.5),
    maybe_apply(strong_color_jitter, p=0.9)
]

normal_augmentations = [
    maybe_apply(random_resized_crop, p=0.6),
    maybe_apply(random_rotation, p=0.7),
    maybe_apply(random_horizontal_flip, p=0.5),
    maybe_apply(random_vertical_flip, p=0.5),
    maybe_apply(color_jitter, p=0.8)
]


normal_transform = TransformImages(normal_augmentations)

special_transform = TransformImages(special_augmentations)

#CLASE QUE DEFINE NUESTRO DATASET APLICANDO EL PREPROCESAMIENTO DE ARRIBA

class CustomImageDataset(Dataset):
    def __init__(self, dataset_path, normal_transform=None, special_transform=None):
        self.dataset_path = dataset_path
        self.normal_transform = normal_transform
        self.special_transform = special_transform
        self.image_paths = []
        self.mask_paths = []
        self.minority_indices = []

        for dir in sorted(os.listdir(self.dataset_path)):
            subdir = sorted(os.listdir(os.path.join(self.dataset_path, dir))) # listamos los directorios dentro de cada directorio
            for sb in subdir:
                folder_path = os.path.join(self.dataset_path, dir, sb)
                all_files = os.listdir(folder_path)

                # Buscar todas las imágenes _endo.png
                endo_images = sorted([f for f in all_files if f.endswith('_endo.png')])

                for image in endo_images:
                    mask = image.replace('_endo.png', '_endo_watershed_mask.png')
                    image_path = os.path.join(folder_path, image)
                    mask_path = os.path.join(folder_path, mask)

                    if os.path.exists(mask_path):
                        self.image_paths.append(image_path)
                        self.mask_paths.append(mask_path)
                    else:
                        print(f"No se encontró máscara para: {image}")

        for i in range(len(self.mask_paths)):
            mask = decode_image(self.mask_paths[i])
            image = decode_image(self.image_paths[i])
            _, mask = resize_quarter(image, mask)
            mask = format_mask(mask)
            present_classes = mask.unique().tolist()
            if any(c in present_classes for c in [7, 8, 11, 12]):
                self.minority_indices.extend([i] * 2)  # duplicado mínimo
                if 8 in present_classes:
                    self.minority_indices.extend([i] * 2)
                if 11 in present_classes:
                    self.minority_indices.extend([i] * 3)
                if 12 in present_classes:
                    self.minority_indices.append(i)


    def __len__(self):
        return len(self.image_paths) + len(self.minority_indices)

    def __getitem__(self, i):
        if i < len(self.image_paths):
            real_index = i
        else:
            real_index = self.minority_indices[i - len(self.image_paths)]
        image = self.image_paths[real_index]
        mask = self.mask_paths[real_index]

        image = decode_image(image)
        mask = decode_image(mask)

        image = v2.functional.to_dtype(image, dtype=t.float32, scale=True) #Para normalizar la imagen: (0,1)
        image, mask = crop_foreground_and_resize(image, mask)

        # Usamos la máscara ya cargada y clonada, y evaluamos si contiene clases minoritarias
        has_minority = any(
            cls.item() in [7, 8, 11, 12]
            for cls in t.unique(format_mask(mask.clone()))
        )
        if has_minority and self.special_transform:
            image, mask, applied_transforms = self.special_transform(image, mask)
            '''
            if random.random() < 0.05:
                print("[DEBUG] Ejemplo de transformación de clases minoritarias")
                print(f"[DEBUG] Transformaciones aplicadas: {applied_transforms}")
                visualize_with_legend(image, format_mask(mask.clone()))
             '''
        elif self.normal_transform:
            image, mask, applied_transforms = self.normal_transform(image, mask)
            '''
            if random.random() < 0.05:
                print("[DEBUG] Ejemplo de transformación de clases normales")
                print(f"[DEBUG] Transformaciones aplicadas: {applied_transforms}")
                visualize_with_legend(image, format_mask(mask.clone()))
            '''
        mask = format_mask(mask)
        return image, mask

    def set_transform_none(self):
        self.normal_transform = None
        self.special_transform = None

    def set_transform_true(self):
        self.normal_transform = normal_transform
        self.special_transform = special_transform




import torch as t
from torch.utils.data import Dataset
from torchvision.io import decode_image
import torchvision.transforms as transforms
from torchvision.transforms import v2, InterpolationMode, Resize
import torchvision.transforms.functional as F
import random
import kornia.augmentation as K
import os
import config



def get_foreground_bbox(mask, margin=10):
    """
    Devuelve la bounding box que contiene todito excepto el fondo (clase 0).
    Añade un margen para evitar recortes muy ajustados.

    Parámetros:
        mask (Tensor): máscara con etiquetas de clase.
        margin (int): margen a añadir alrededor de la bounding box.

    Retorna:
        tuple o None: (top, left, bottom, right) si hay primer plano, None si no.
    """
    mask_copy = mask.clone()
    mask_copy = format_mask(mask_copy)

    foreground = mask_copy != 0

    if foreground.sum() == 0:
        return None

    coords = foreground.nonzero(as_tuple=False) # .nonzero(as_tuple=False) en PyTorch busca todas las posiciones donde un tensor es distinto de cero (o True si es booleano) y devuelve las coordenadas de esos elementos.
    '''
    as_tuple=False (por defecto)
    Devuelve un solo tensor con forma (N, D) donde:
    
    N = número de elementos que cumplen la condición (True o ≠ 0)
    
    D = número de dimensiones del tensor.
    
    en cambio, as_tuple=True te devuelve una tupla por cada dimensión con cuantos han dado True
    '''
    top = max(coords[:, 0].min().item() - margin, 0)
    bottom = min(coords[:, 0].max().item() + margin, mask.shape[1] - 1)
    left = max(coords[:, 1].min().item() - margin, 0)
    right = min(coords[:, 1].max().item() + margin, mask.shape[2] - 1)

    if bottom <= top or right <= left:
        return None

    return top, left, bottom, right


def crop_foreground_and_resize(image, mask, output_size=config.DESIRED_IMAGE_SIZE):
    """
    Recorta la imagen y máscara alrededor del primer plano y redimensiona al tamaño deseado.
    Si no hay primer plano, redimensiona toda la imagen y máscara.

    Parámetros:
        image (Tensor): imagen original.
        mask (Tensor): máscara con etiquetas.
        output_size (tuple): tamaño deseado para la salida (alto, ancho).

    Retorna:
        tuple: (imagen redimensionada, máscara redimensionada).
    """
    bbox = get_foreground_bbox(mask)
    if bbox is None:
        image = transforms.Resize(output_size, interpolation=InterpolationMode.BILINEAR)(image)
        mask = transforms.Resize(output_size, interpolation=InterpolationMode.NEAREST)(mask)
        return image, mask

    top, left, bottom, right = bbox
    image_cropped = image[:, top:bottom+1, left:right+1] #+1 para incluir el final en el indexado
    mask_cropped = mask[:, top:bottom+1, left:right+1]

    image_resized = transforms.Resize(output_size, interpolation=InterpolationMode.BILINEAR)(image_cropped)
    mask_resized = transforms.Resize(output_size, interpolation=InterpolationMode.NEAREST)(mask_cropped)

    return image_resized, mask_resized


def format_mask(mask):
    """
    Convierte una máscara RGB en una máscara de etiquetas numéricas según un mapa de colores a clases.

    Parámetros:

        mask (Tensor): máscara en formato RBG: (C, H, W), valores en rango [0, 255] al ser la watershed, los 3 canales RGB tienen el mismo valor

    Retorna:
        Tensor: máscara con valores enteros que representan clases, con forma (H, W) donde cada píxel contiene el índice entero de la clase correspondiente.
    """
    mask = mask.permute(1, 2, 0)[:, :, 0] # pasa a (H, W, C) y se quitan los dos canales sobrantes

    class_mask = t.zeros((mask.shape[0], mask.shape[1]), dtype=t.long) #se crea una máscara copia que tenga los valores a cero

    for color_value, class_index in config.ID_TO_CLASS.items(): # recorremos cada color del diccionario (sabiendo su indice correspondiente)
        class_mask[mask == color_value] = class_index # sustituimos en cada pixel por su indice correspondiente mirando si en la mascara original estaba ese color
    return class_mask


def maybe_apply(transform_fn, p=0.5):
    """
    Envuelve una función de transformación para que se aplique con probabilidad p.

    Parámetros:
        transform_fn (func): función que aplica una transformación.
        p (float): probabilidad de aplicar la transformación.

    Retorna:
        func: función que aplica la transformación con probabilidad p.
    """
    def wrapped(image, mask):
        if random.random() < p:
            return transform_fn(image, mask)
        return image, mask
    wrapped.__name__ = transform_fn.__name__
    return wrapped


def get_bounding_box(mask, target_classes=config.TARGET_CLASSES):
    """
    Devuelve la bounding box que contiene todas las clases especificadas.

    Parámetros:
        mask (Tensor): máscara con etiquetas.
        target_classes (list): lista de clases a incluir en la bbox.

    Retorna:
        tuple o None: (top, left, bottom, right) o None si no hay clases objetivo.
    """
    indices = t.zeros_like(mask, dtype=t.bool)
    for cls in target_classes:
        indices |= (mask == cls)

    if indices.sum() == 0:
        return None

    coords = indices.nonzero(as_tuple=False)
    top = coords[:, 0].min().item()
    bottom = coords[:, 0].max().item()
    left = coords[:, 1].min().item()
    right = coords[:, 1].max().item()

    return top, left, bottom, right


def resize_quarter(image, mask):
    """
    Reduce la imagen y máscara a un 75% de su tamaño original.

    Parámetros:
        image (Tensor): imagen original.
        mask (Tensor): máscara original.

    Retorna:
        tuple: imagen y máscara redimensionadas.
    """
    orig_height, orig_width = image.shape[1], image.shape[2]

    new_height = int(orig_height * 3 / 4)
    new_width = int(orig_width * 3 / 4)

    resize_transform_img = Resize((new_height, new_width), interpolation=InterpolationMode.BILINEAR) #Se crea el objeto transformador de resize
    image = resize_transform_img(image)

    resize_transform_mask = Resize((new_height, new_width), interpolation=InterpolationMode.NEAREST)
    mask = resize_transform_mask(mask)

    return image, mask





def random_rotation(image, mask, max_angle=25):
    """
    Aplica una rotación aleatoria a imagen y máscara con ángulo máximo dado.

    Parámetros:
        image (Tensor): imagen a rotar.
        mask (Tensor): máscara a rotar.
        max_angle (int): ángulo máximo de rotación en grados.

    Retorna:
        tuple: imagen y máscara rotadas.
    """
    angle = random.uniform(-max_angle, max_angle)
    image = transforms.functional.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
    mask = transforms.functional.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
    return image, mask


def random_horizontal_flip(image, mask):
    """
    Aplica un volteo horizontal a imagen y máscara.

    Parámetros:
        image (Tensor): imagen a voltear.
        mask (Tensor): máscara a voltear.

    Retorna:
        tuple: imagen y máscara volteadas horizontalmente.
    """
    image = transforms.functional.hflip(image)
    mask = transforms.functional.hflip(mask)
    return image, mask


def random_vertical_flip(image, mask):
    """
    Aplica un volteo vertical a imagen y máscara.

    Parámetros:
        image (Tensor): imagen a voltear.
        mask (Tensor): máscara a voltear.

    Retorna:
        tuple: imagen y máscara volteadas verticalmente.
    """
    image = transforms.functional.vflip(image)
    mask = transforms.functional.vflip(mask)
    return image, mask


def random_crop(image, mask, crop_size=config.CROP_SIZE):
    """
    Aplica un recorte aleatorio y redimensiona de nuevo al tamaño original.

    Parámetros:
        image (Tensor): imagen a recortar.
        mask (Tensor): máscara a recortar.
        crop_size (tuple): tamaño del recorte (alto, ancho).

    Retorna:
        tuple: imagen y máscara recortadas y redimensionadas.
    """
    height, width = image.shape[1], image.shape[2]

    top = random.randint(0, height - crop_size[0])
    left = random.randint(0, width - crop_size[1])

    image_cropped = image[:, top:top + crop_size[0], left:left + crop_size[1]]
    mask_cropped = mask[:, top:top + crop_size[0], left:left + crop_size[1]]

    image_cropped = transforms.Resize((height, width), interpolation=InterpolationMode.BILINEAR)(image_cropped)
    mask_cropped = transforms.Resize((height, width), interpolation=InterpolationMode.NEAREST)(mask_cropped)

    return image_cropped, mask_cropped


def strong_random_crop(image, mask, crop_size=config.STRONG_CROP_SIZE):
    """
    Aplica un recorte aleatorio más grande que random_crop y redimensiona al tamaño original.

    Parámetros:
        image (Tensor): imagen a recortar.
        mask (Tensor): máscara a recortar.
        crop_size (tuple): tamaño del recorte fuerte (alto, ancho).

    Retorna:
        tuple: imagen y máscara recortadas y redimensionadas.
    """
    height, width = image.shape[1], image.shape[2]

    top = random.randint(0, height - crop_size[0])
    left = random.randint(0, width - crop_size[1])

    image_cropped = image[:, top:top + crop_size[0], left:left + crop_size[1]]
    mask_cropped = mask[:, top:top + crop_size[0], left:left + crop_size[1]]

    image_cropped = transforms.Resize((height, width), interpolation=InterpolationMode.BILINEAR)(image_cropped)
    mask_cropped = transforms.Resize((height, width), interpolation=InterpolationMode.NEAREST)(mask_cropped)

    return image_cropped, mask_cropped


def crop_around_bbox(image, mask, crop_size=config.CROP_SIZE):
    """
    Recorta la imagen y máscara centrados alrededor de la bounding box de ciertas clases.

    Args:
        image (Tensor): Imagen original (C,H,W).
        mask (Tensor): Máscara correspondiente.
        crop_size (tuple): Tamaño del recorte (alto, ancho).

    Returns:
        tuple: Imagen y máscara recortadas y redimensionadas al tamaño original.
    """
    bbox = get_bounding_box(mask, [7,8,11,12])
    top, left, bottom, right = bbox
    center_y = (top + bottom) // 2
    center_x = (left + right) // 2

    half_h = crop_size[0] // 2
    half_w = crop_size[1] // 2

    start_y = max(center_y - half_h, 0)
    start_x = max(center_x - half_w, 0)

    end_y = min(start_y + crop_size[0], image.shape[1])
    end_x = min(start_x + crop_size[1], image.shape[2])

    start_y = end_y - crop_size[0]
    start_x = end_x - crop_size[1]

    image_cropped = image[:, start_y:end_y, start_x:end_x]
    mask_cropped = mask[:, start_y:end_y, start_x:end_x]

    image_cropped = transforms.Resize((image.shape[1], image.shape[2]), interpolation=InterpolationMode.BILINEAR)(image_cropped)
    mask_cropped = transforms.Resize((image.shape[1], image.shape[2]), interpolation=InterpolationMode.NEAREST)(mask_cropped)

    return image_cropped, mask_cropped


def color_jitter(image, mask, brightness_range=(0.85, 1.1), contrast_range=(0.9, 1.15), saturation_range=(0.95, 1.05)):
    """
    Aplica jitter aleatorio de brillo, contraste y saturación a la imagen, sin modificar la máscara.

    Args:
        image (Tensor): Imagen a modificar.
        mask (Tensor): Máscara correspondiente (sin cambios).
        brightness_range (tuple): Rango para brillo.
        contrast_range (tuple): Rango para contraste.
        saturation_range (tuple): Rango para saturación.

    Returns:
        tuple: Imagen modificada y máscara sin cambios.
    """
    brightness = random.uniform(*brightness_range)
    contrast = random.uniform(*contrast_range)
    saturation = random.uniform(*saturation_range)

    image = F.adjust_brightness(image, brightness)
    image = F.adjust_contrast(image, contrast)
    image = F.adjust_saturation(image, saturation)

    return image, mask


def strong_color_jitter(image, mask, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3), saturation_range=(0.8, 1.2)):
    """
    Aplica jitter de color más agresivo (brillo, contraste, saturación) a la imagen.

    Args:
        image (Tensor): Imagen a modificar.
        mask (Tensor): Máscara correspondiente (sin cambios).
        brightness_range (tuple): Rango para brillo.
        contrast_range (tuple): Rango para contraste.
        saturation_range (tuple): Rango para saturación.

    Returns:
        tuple: Imagen modificada y máscara sin cambios.
    """
    brightness = random.uniform(*brightness_range)
    contrast = random.uniform(*contrast_range)
    saturation = random.uniform(*saturation_range)

    image = F.adjust_brightness(image, brightness)
    image = F.adjust_contrast(image, contrast)
    image = F.adjust_saturation(image, saturation)

    return image, mask


def random_affine(image, mask):
    """
    Aplica una transformación afín aleatoria (rotación, traslación, escalado) a imagen y máscara.

    Args:
        image (Tensor): Imagen a transformar.
        mask (Tensor): Máscara a transformar.

    Returns:
        tuple: Imagen y máscara transformadas.
    """
    angle = random.uniform(-25, 25)

    max_dx = int(0.05 * image.shape[2])
    max_dy = int(0.05 * image.shape[1])
    translate = (random.randint(-max_dx, max_dx), random.randint(-max_dy, max_dy))

    scale = random.uniform(0.9, 1.1)

    image = F.affine(
        image, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR, fill=0
    )
    mask = F.affine(
        mask, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0],
        interpolation=InterpolationMode.NEAREST, fill=0
    )

    return image, mask


def elastic_transform(image, mask, alpha=6.0, sigma=2.5):
    """
    Aplica una transformación elástica conservadora para deformar imagen y máscara.

    Args:
        image (Tensor): Imagen a deformar.
        mask (Tensor): Máscara a deformar.
        alpha (float): Parámetro de fuerza del desplazamiento.
        sigma (float): Parámetro de suavizado.

    Returns:
        tuple: Imagen y máscara deformadas.
    """
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)

    stacked = t.cat([image, mask.float()], dim=0).unsqueeze(0)

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
    Realiza un recorte aleatorio redimensionado simulando RandomResizedCrop para imagen y máscara.

    Args:
        image (Tensor): Imagen original.
        mask (Tensor): Máscara correspondiente.
        scale (tuple): Rango de escala del área.
        ratio (tuple): Rango de aspecto (ancho/alto).

    Returns:
        tuple: Imagen y máscara recortadas y redimensionadas o originales si no es posible recortar.
    """
    _, H, W = image.shape
    area = H * W

    for _ in range(10):
        target_area = random.uniform(*scale) * area
        aspect_ratio = random.uniform(*ratio)

        new_w = int(round((target_area * aspect_ratio) ** 0.5))
        new_h = int(round((target_area / aspect_ratio) ** 0.5))

        if new_w <= W and new_h <= H:
            top = random.randint(0, H - new_h)
            left = random.randint(0, W - new_w)

            image_cropped = image[:, top:top+new_h, left:left+new_w]
            mask_cropped = mask[:, top:top+new_h, left:left+new_w]

            image_resized = F.resize(image_cropped, size=(H, W), interpolation=InterpolationMode.BILINEAR)
            mask_resized = F.resize(mask_cropped, size=(H, W), interpolation=InterpolationMode.NEAREST)

            return image_resized, mask_resized

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

class CustomImageDataset(Dataset):
    """
    Dataset personalizado que carga imágenes y máscaras, aplica preprocesamiento y maneja clases minoritarias.

    Args:
        dataset_path (str): Ruta base del dataset.
        normal_transform (func): Transformaciones normales a aplicar.
        special_transform (func): Transformaciones especiales para clases minoritarias.
    """

    def __init__(self, dataset_path, normal_transform=None, special_transform=None):
        """
        Inicializa el dataset, cargando rutas y preparando índices para clases minoritarias.

        Args:
            dataset_path (str): Ruta raíz del dataset.
            normal_transform (func, opcional): Transformaciones normales.
            special_transform (func, opcional): Transformaciones especiales.
        """
        self.dataset_path = dataset_path
        self.normal_transform = normal_transform
        self.special_transform = special_transform
        self.image_paths = []
        self.mask_paths = []
        self.minority_indices = []

        for dir in sorted(os.listdir(self.dataset_path)):
            subdir = sorted(os.listdir(os.path.join(self.dataset_path, dir)))
            for sb in subdir:
                folder_path = os.path.join(self.dataset_path, dir, sb)
                all_files = os.listdir(folder_path)

                endo_images = sorted([f for f in all_files if f.endswith('_endo.png')])

                for image in endo_images:
                    mask = image.replace('_endo.png', '_endo_watershed_mask.png') # replace() en Python es un metodo de strings que busca un fragmento de texto y lo sustituye por otro.
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
            '''
            Decodifica una imagen en un tensor de tipo uint8, ya sea desde una ruta de archivo o desde bytes codificados en crudo.
            
            uint8 → “unsigned integer 8-bit” Solo guarda enteros entre 0 y 255 (porque 2⁸ = 256 valores posibles).
            
            tensor → estructura de datos de PyTorch (como un array de NumPy, pero con soporte para GPU y operaciones optimizadas).
            
            En imágenes, normalmente tendrá forma (C, H, W) → canales, alto y ancho.
            
            Si es RGB, C = 3; si es en escala de grises, C = 1.
            '''

            _, mask = resize_quarter(image, mask)
            mask = format_mask(mask)

            present_classes = mask.unique().tolist() #lo pasamos de tensor a lista
            if any(c in present_classes for c in [7, 8, 11, 12]): # any(...) → devuelve True si alguna condición es cierta.
                                                                  # (c in present_classes for c in [...]) → generador que va evaluando cada valor y se detiene al primer True.
                self.minority_indices.extend([i] * 2) # extend(iterable) → añade cada elemento de un iterable a la lista, “desempaquetándolo”.
                '''
                ejemplo:
                i = 5
                resultado = [i] * 2
                print(resultado)  # [5, 5]
                '''
                if 8 in present_classes:
                    self.minority_indices.extend([i] * 2)
                if 11 in present_classes:
                    self.minority_indices.extend([i] * 3)
                if 12 in present_classes:
                    self.minority_indices.append(i) # append(x) → añade un solo elemento al final de la lista, sin importar si es una lista u otro tipo de objeto.

    def __len__(self):
        """
        Retorna la longitud total del dataset incluyendo duplicados para minorías.
        """
        return len(self.image_paths) + len(self.minority_indices)

    def __getitem__(self, i):
        """
        Obtiene la imagen y máscara transformadas según índice, aplicando transformaciones especiales si corresponde.

        Args:
            i (int): Índice de la muestra.

        Returns:
            tuple: Imagen y máscara procesadas.
        """
        if i < len(self.image_paths):
            real_index = i
        else:
            real_index = self.minority_indices[i - len(self.image_paths)]
        image = self.image_paths[real_index]
        mask = self.mask_paths[real_index]

        image = decode_image(image)
        mask = decode_image(mask)

        image = v2.functional.to_dtype(image, dtype=t.float32, scale=True) # Convierte la imagen a float32 y normaliza los valores de 0–255 a un rango 0–1.
        image, mask = crop_foreground_and_resize(image, mask)

        has_minority = any(
            cls.item() in [7, 8, 11, 12]
            for cls in t.unique(format_mask(mask.clone()))
        )
        if has_minority and self.special_transform:
            image, mask, applied_transforms = self.special_transform(image, mask)
        elif self.normal_transform:
            image, mask, applied_transforms = self.normal_transform(image, mask)

        mask = format_mask(mask)
        return image, mask

    def set_transform_none(self):
        self.normal_transform = None
        self.special_transform = None

    def set_transform_true(self, normal_transform, special_transform):
        self.normal_transform = normal_transform
        self.special_transform = special_transform

class TransformImages:
    """
    Aplica una lista de transformaciones a una imagen y máscara.

    Atributos:
        augmentations (list): lista de funciones de transformación.
    """

    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, image, mask):
        """
        Aplica todas las transformaciones en orden.

        Parámetros:
            image (Tensor): imagen a transformar.
            mask (Tensor): máscara a transformar.

        Retorna:
            tuple: imagen transformada, máscara transformada, lista de nombres de transformaciones aplicadas.
        """
        applied_transforms = []
        for transform in self.augmentations:
            transform_name = transform.__name__
            image, mask = transform(image, mask)
            applied_transforms.append(transform_name)
        return image, mask, applied_transforms

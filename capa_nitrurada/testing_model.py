import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from django.conf import settings
import uuid

# =========================
# FUNCIONES AUXILIARES
# =========================
def generar_patches_img(img, patch_size=256, stride=None):
    """Genera patches con coordenadas relativas (5 canales totales)."""
    h, w, _ = img.shape
    if stride is None:
        stride = patch_size

    y_starts = np.arange(0, h - patch_size + 1, stride)
    if y_starts.size > 0 and y_starts[-1] + patch_size < h:
        y_starts = np.append(y_starts, h - patch_size)
    y_starts = y_starts.astype(int)

    x_starts = np.arange(0, w - patch_size + 1, stride)
    if x_starts.size > 0 and x_starts[-1] + patch_size < w:
        x_starts = np.append(x_starts, w - patch_size)
    x_starts = x_starts.astype(int)

    patches, coords = [], []
    for y in y_starts:
        for x in x_starts:
            patch = img[y:y+patch_size, x:x+patch_size, :]
            yy = np.linspace(y/h, (y+patch_size)/h, patch_size, endpoint=False)
            xx = np.linspace(x/w, (x+patch_size)/w, patch_size, endpoint=False)
            coord_y, coord_x = np.meshgrid(yy, xx, indexing="ij")
            coord_x = coord_x[..., np.newaxis]
            coord_y = coord_y[..., np.newaxis]
            patch_with_coords = np.concatenate([patch, coord_x, coord_y], axis=-1)

            patches.append(patch_with_coords)
            coords.append((y, x))
    return np.array(patches, dtype=np.float32), coords

# =========================
# FUNCIONES PERSONALIZADAS
# =========================
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )

def focal_loss(y_true, y_pred, gamma=2., alpha=.25):
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)
    BCE = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    BCE_EXP = tf.keras.backend.exp(-BCE)
    focal = alpha * tf.keras.backend.pow((1 - BCE_EXP), gamma) * BCE
    return tf.keras.backend.mean(focal)

def iou_metric(y_true, y_pred, smooth=1e-6):
    y_pred = tf.cast(y_pred > 0.5, 'float32')
    intersection = tf.keras.backend.sum(y_true * y_pred)
    union = tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

def gradient_loss(y_true, y_pred):
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    grad_diff = tf.abs(dy_true - dy_pred) + tf.abs(dx_true - dx_pred)
    return tf.reduce_mean(grad_diff)

def shape_aware_loss(y_true, y_pred, lambda_shape=0.9):
    dice = dice_loss(y_true, y_pred)
    focal = focal_loss(y_true, y_pred)
    shape_term = gradient_loss(y_true, y_pred)
    return dice + focal + lambda_shape * shape_term

# =========================
# CARGAR MODELO
# =========================

# =========================
# FUNCIÓN PRINCIPAL
# =========================
def procesar_prediccion(img_path, model, request, visualizar=False):
    """
    Ejecuta el pipeline completo:
    - Carga imagen
    - Genera patches
    - Predice y reconstruye
    - Aplica realce asimétrico (solo esa versión)
    - Guarda y devuelve la URL pública
    """
    print(f"\n=== Iniciando predicción para: {img_path} ===")
    img = img_to_array(load_img(img_path)) / 255.0

    patch_size = 256
    stride = patch_size // 2
    patches, coords = generar_patches_img(img, patch_size=patch_size, stride=stride)

    full_h, full_w = img.shape[:2]
    full_pred = np.zeros((full_h, full_w))
    count = np.zeros((full_h, full_w))

    for coord, patch in zip(coords, patches):
        y, x = coord
        pred = model.predict(patch[np.newaxis, ...], verbose=0)[0, ..., 0]
        full_pred[y:y+patch_size, x:x+patch_size] += pred
        count[y:y+patch_size, x:x+patch_size] += 1

    full_pred = np.divide(full_pred, count, out=np.zeros_like(full_pred), where=count != 0)
    full_pred = np.clip(full_pred, 0, 1)

    # ==========================
    # REALCE ASIMÉTRICO
    # ==========================
    def realce_asimetrico(x, centro=0.45, intensidad=10):
        """
        Realce asimétrico:
        - Empuja valores intermedios hacia abajo
        - Eleva los altos
        """
        x = np.clip(x, 0, 1)
        y = 1 / (1 + np.exp(-intensidad * (x - centro)))
        y = (y - y.min()) / (y.max() - y.min())
        return y

    mapa = realce_asimetrico(full_pred)

    # ==========================
    # GUARDAR RESULTADO
    # ==========================
    nombre_unico = f"output_image_realce_{uuid.uuid4().hex}.png"
    output_path = os.path.join(settings.MEDIA_ROOT, nombre_unico)
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Imagen Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    im = plt.imshow(mapa, cmap="Reds", vmin=0, vmax=1)
    plt.title("Predicción (Realce Asimétrico)")
    plt.axis("off")

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Probabilidad de nitruro")

    legend_elements = [
        Patch(facecolor='white', edgecolor='black', label='Fondo'),
        Patch(facecolor='red', edgecolor='black', label='Nitruro')
    ]
    plt.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    output_url = request.build_absolute_uri(settings.MEDIA_URL + nombre_unico)
    print(f"✅ Imagen guardada en: {output_path}")
    print(f"🌐 URL pública: {output_url}\n")

    return output_url

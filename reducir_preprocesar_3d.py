import os
import numpy as np
import cv2
import scipy.ndimage
from tqdm import tqdm
from pydicom import dcmread

# Parámetros
DICOM_ROOT = "./datos_anonimizados"
OUTPUT_DIR = "./volumenes_preprocesados_3d_reducidos"
HU_MIN = -1000
HU_MAX = 400
TARGET_SLICES = 128
TARGET_SIZE = (256, 256)  

def convert_to_hu(dicom_slice):
    """Convierte una imagen DICOM a Hounsfield Units (HU) y normaliza a [0, 1]."""
    image = dicom_slice.pixel_array.astype(np.float32)

    intercept = getattr(dicom_slice, "RescaleIntercept", 0)
    slope = getattr(dicom_slice, "RescaleSlope", 1)

    if slope != 1:
        image = slope * image
    image += intercept

    image = np.clip(image, HU_MIN, HU_MAX)
    image = (image - HU_MIN) / (HU_MAX - HU_MIN)  # Normalización

    return image

def resize_slices(volume, target_slices):
    """Ajusta el número de slices en el eje Z a `target_slices`."""
    num_slices = volume.shape[0]

    if num_slices > target_slices:
        start = (num_slices - target_slices) // 2
        volume = volume[start:start + target_slices]
    elif num_slices < target_slices:
        zoom_factor = target_slices / num_slices
        volume = scipy.ndimage.zoom(volume, (zoom_factor, 1, 1), order=1)

    return volume

def preprocess_dicom_volumes(dicom_root, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for patient_id in tqdm(os.listdir(dicom_root), desc="Procesando pacientes"):
        patient_dir = os.path.join(dicom_root, patient_id)
        if not os.path.isdir(patient_dir):
            continue

        dicom_files = []
        for filename in os.listdir(patient_dir):
            filepath = os.path.join(patient_dir, filename)
            try:
                ds = dcmread(filepath, force=True)
                dicom_files.append(ds)
            except:
                continue

        if not dicom_files:
            print(f"❌ No se encontraron archivos DICOM en {patient_id}. Se omite.")
            continue

        try:
            dicom_files = sorted(dicom_files, key=lambda s: getattr(s, "SliceLocation", 0))
            volume = np.stack([convert_to_hu(s) for s in dicom_files], axis=0)
            volume = resize_slices(volume, TARGET_SLICES)
            volume = np.array([cv2.resize(slice_, TARGET_SIZE, interpolation=cv2.INTER_LINEAR) for slice_ in volume])

            save_path = os.path.join(output_dir, f"{patient_id}.npy")
            np.save(save_path, volume)
            print(f"✅ Guardado: {save_path} | Shape: {volume.shape}")
        except Exception as e:
            print(f"❌ Error procesando {patient_id}: {e}")

# Ejecutar
if __name__ == "__main__":
    preprocess_dicom_volumes(DICOM_ROOT, OUTPUT_DIR)

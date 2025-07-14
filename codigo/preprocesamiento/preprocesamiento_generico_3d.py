import os
import numpy as np
import pydicom
from skimage.transform import resize
from collections import defaultdict
from tqdm import tqdm

def get_cancer_type(patient_id):
    cancer_code = patient_id.split('-')[1][0].upper()
    cancer_types = {
        'A': 0,  # Adenocarcinoma
        'B': 1,  # Small Cell
        'E': 2,  # Large Cell
        'G': 3   # Squamous
    }
    return cancer_types.get(cancer_code, -1)

def preprocess_and_save_3d_series_one_by_one(root_dir, save_dir, target_shape=(256, 512, 512)):
    os.makedirs(save_dir, exist_ok=True)
    nombre_usados = defaultdict(int)

    # Estructura: (label, patient_id, series_uid) -> [file paths]
    series_dicoms = defaultdict(list)

    print("[INFO] Agrupando rutas por series...")
    for dirpath, _, filenames in tqdm(list(os.walk(root_dir))):
        for file in filenames:
            if not file.lower().endswith(".dcm"):
                continue
            try:
                full_path = os.path.join(dirpath, file)
                dcm = pydicom.dcmread(full_path, stop_before_pixels=True)

                if getattr(dcm, "Modality", None) != "CT":
                    continue

                path_parts = dirpath.split(os.sep)
                patient_id = next(p for p in path_parts if p.startswith("Lung_Dx-"))
                label = get_cancer_type(patient_id)
                if label == -1:
                    continue

                series_uid = getattr(dcm, "SeriesInstanceUID", None)
                if not series_uid:
                    continue

                key = (label, patient_id, series_uid)
                series_dicoms[key].append(full_path)
            except:
                continue

    print(f"[INFO] Se detectaron {len(series_dicoms)} series CT únicas.")

    for (label, patient_id, series_uid), paths in tqdm(series_dicoms.items()):
        try:
            slices = []
            for path in paths:
                dcm = pydicom.dcmread(path)
                if dcm.pixel_array is None:
                    continue
                img = dcm.pixel_array.astype(np.float32)
                if hasattr(dcm, "RescaleIntercept") and hasattr(dcm, "RescaleSlope"):
                    img = img * dcm.RescaleSlope + dcm.RescaleIntercept
                z = dcm.ImagePositionPatient[2] if "ImagePositionPatient" in dcm else len(slices)
                slices.append((z, img))

            if len(slices) < 10:
                print(f"[!] Serie {patient_id}-{series_uid} tiene muy pocos slices. Saltando.")
                continue

            slices.sort(key=lambda x: x[0])
            volume = np.stack([s[1] for s in slices], axis=0)
            volume = (volume + 1000) / 2000
            volume = np.clip(volume, 0, 1)
            volume_resized = resize(volume, target_shape, mode='reflect', anti_aliasing=True).astype(np.float32)

            nombre_usados[(label, patient_id)] += 1
            nombre = f"{label}_{patient_id}_{nombre_usados[(label, patient_id)]}.npy"
            np.save(os.path.join(save_dir, nombre), volume_resized)

            print(f"[✓] Guardado: {nombre}")
            del volume, volume_resized, slices  # liberar memoria

        except Exception as e:
            print(f"[X] Error en {patient_id}-{series_uid}: {e}")
            continue

preprocess_and_save_3d_series_one_by_one(
    root_dir="./dataset_generico_pulmones",
    save_dir="./preprocessed_dataset_generico_3d",
    target_shape=(256, 512, 512)
)

import os
import hashlib
from tqdm import tqdm
import dicom2nifti

# El tipo de cáncer se extrae del ID del paciente
def get_cancer_type(patient_id):
    cancer_code = patient_id.split('-')[1][0].upper()  # Ej: 'A' de 'Lung_Dx-A0001'
    cancer_types = {
        'A': 0,  # Adenocarcinoma
        'B': 1,  # Small Cell
        'E': 2,  # Large Cell
        'G': 3   # Squamous
    }
    return cancer_types.get(cancer_code, -1)

def convertir_dicom_a_nifti_con_label(dicom_root, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print("[INFO] Buscando carpetas de series DICOM...")

    posibles_series = []
    for dirpath, _, filenames in os.walk(dicom_root):
        if any(fname.lower().endswith('.dcm') for fname in filenames):
            posibles_series.append(dirpath)

    print(f"[INFO] Se detectaron {len(posibles_series)} carpetas con archivos DICOM.")

    for serie_path in tqdm(posibles_series):
        try:
            # Buscar el ID del paciente en la ruta
            path_parts = serie_path.split(os.sep)
            patient_id = next((p for p in path_parts if p.startswith("Lung_Dx-")), None)
            if not patient_id:
                print(f"❌ No se encontró ID de paciente en ruta: {serie_path}")
                continue

            label = get_cancer_type(patient_id)
            if label == -1:
                print(f"❌ Tipo de cáncer desconocido para {patient_id}. Saltando.")
                continue

            # Listar los archivos actuales en el output_dir para detectar cambios
            archivos_antes = set(os.listdir(output_dir))

            # Ejecutar la conversión
            dicom2nifti.convert_directory(serie_path, output_dir, compression=True, reorient=True)

            # Detectar nuevo archivo .nii.gz creado
            archivos_despues = set(os.listdir(output_dir))
            nuevos_archivos = archivos_despues - archivos_antes
            nuevos_nii = [f for f in nuevos_archivos if f.endswith(".nii.gz")]

            if not nuevos_nii:
                print(f"❌ No se generó ningún .nii.gz para {serie_path}")
                continue

            archivo_generado = nuevos_nii[0]
            origen = os.path.join(output_dir, archivo_generado)
            nuevo_nombre = f"{label}_{patient_id}.nii.gz"
            destino = os.path.join(output_dir, nuevo_nombre)

            os.rename(origen, destino)
            print(f"✅ Guardado como: {nuevo_nombre}")
        except Exception as e:
            print(f"❌ Error en {serie_path}: {e}")

# USO
dicom_root = "./dataset_generico_pulmones"
output_dir = "./nii_dataset_generico_sin_preprocesar"


convertir_dicom_a_nifti_con_label(dicom_root, output_dir)



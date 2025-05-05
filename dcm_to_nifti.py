import os
import dicom2nifti
from tqdm import tqdm

def convertir_dicom_a_nifti(dicom_root, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for paciente in tqdm(os.listdir(dicom_root)):
        paciente_dir = os.path.join(dicom_root, paciente)
        if not os.path.isdir(paciente_dir):
            continue

        try:
            print(f"🔄 Procesando {paciente}...")
            output_path = os.path.join(output_dir, f"{paciente}.nii.gz")
            dicom2nifti.convert_dicom.dicom_series_to_nifti(
                paciente_dir,
                output_path
            )
            print(f"✅ Guardado: {output_path}")
        except Exception as e:
            print(f"❌ Error con {paciente}: {e}")

# USO
dicom_root = "./datos_anonimizados"               # Carpeta con subcarpetas por paciente
output_dir = "./nifti_convertidos_anonimizados"   # Carpeta de salida para los .nii.gz

convertir_dicom_a_nifti(dicom_root, output_dir)

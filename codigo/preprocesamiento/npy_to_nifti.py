import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

def convertir_npy_a_nifti(npy_path, output_path, spacing=(1.0, 1.0, 1.0)):
    vol = np.load(npy_path)
    affine = np.diag(spacing + (1.0,))
    nifti_img = nib.Nifti1Image(vol, affine)
    nib.save(nifti_img, output_path)

def convertir_carpeta_npy_a_nifti(npy_dir, output_dir, spacing=(1.0, 1.0, 1.0)):
    os.makedirs(output_dir, exist_ok=True)
    for archivo in tqdm(os.listdir(npy_dir)):
        if archivo.endswith(".npy"):
            paciente = archivo.replace(".npy", "")
            npy_path = os.path.join(npy_dir, archivo)
            output_path = os.path.join(output_dir, f"{paciente}.nii.gz")
            try:
                convertir_npy_a_nifti(npy_path, output_path, spacing)
                print(f" Guardado: {output_path}")
            except Exception as e:
                print(f" Error con {paciente}: {e}")

input_dir = "./preprocessed_data_3d" # .npy por paciente
output_dir = "./nifti_preprocesados"  # salida .nii.gz
convertir_carpeta_npy_a_nifti(input_dir, output_dir)

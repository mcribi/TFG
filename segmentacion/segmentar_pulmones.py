import os
from totalsegmentator.python_api import totalsegmentator
from tqdm import tqdm


def segmentar_pulmones_nifti(input_dir, output_dir, force_cpu=False):
    os.makedirs(output_dir, exist_ok=True)

    for archivo in tqdm(os.listdir(input_dir)):
        if archivo.endswith(".nii") or archivo.endswith(".nii.gz"):
            ruta_entrada = os.path.join(input_dir, archivo)
            ruta_salida = os.path.join(output_dir, archivo.replace(".nii.gz", "").replace(".nii", "") + "_seg")

            print(f"Segmentando: {archivo}...")
            try:
                totalsegmentator(
                    input=ruta_entrada,
                    output=ruta_salida,
                    task="total", 
                    ml=True,
                    fast=True,
                    device="cpu" if force_cpu else "gpu"
                )
                print(f"Segmentado: {ruta_salida}")
            except Exception as e:
                print(f"Error segmentando {archivo}: {e}")

# 🧪 USO
input_dir = "/mnt/homeGPU/mcribilles/TFG/nifti_convertidos_anonimizados"# Volúmenes .nii.gz
output_dir = "./segmentaciones_lung"    # Salida con las máscaras

segmentar_pulmones_nifti(input_dir, output_dir, force_cpu=False)

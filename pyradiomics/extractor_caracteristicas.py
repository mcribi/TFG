import os
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from radiomics import featureextractor
import radiomics
import logging

# -----------------------------
# ⚙️ Configuración
# -----------------------------
nifti_dir = "./../nifti_convertidos_anonimizados"
output_csv = "caracteristicas_radiomicas_nifti.csv"

# Logging PyRadiomics
radiomics.setVerbosity(logging.INFO)

# Parametrización eficiente
params = {
    "binWidth": 25,
    "resampledPixelSpacing": [2, 2, 2],  # Reduce complejidad
    "interpolator": "sitkLinear",       # Más rápido que B-spline
    "enableCExtensions": True,
    "label": 1
}
extractor = featureextractor.RadiomicsFeatureExtractor(params)
extractor.enableAllFeatures()

# -----------------------------
# 🚀 Proceso de extracción
# -----------------------------
all_features = []

for file in tqdm(os.listdir(nifti_dir), desc="Procesando NIfTI"):
    if not (file.endswith(".nii") or file.endswith(".nii.gz")):
        continue

    try:
        nifti_path = os.path.join(nifti_dir, file)
        image_full = sitk.ReadImage(nifti_path)

        # Recortar a 64 slices centrales
        array_full = sitk.GetArrayFromImage(image_full)
        z = array_full.shape[0]
        array_cropped = array_full[z//2 - 32:z//2 + 32, :, :]

        image = sitk.GetImageFromArray(array_cropped)
        image.SetSpacing(image_full.GetSpacing())
        image.SetOrigin(image_full.GetOrigin())
        image.SetDirection(image_full.GetDirection())

        # Máscara básica (lo que no sea aire)
        mask_array = (array_cropped > -900).astype("uint8")
        mask = sitk.GetImageFromArray(mask_array)
        mask.CopyInformation(image)

        # Extraer
        features = extractor.execute(image, mask)
        filtered = {k: v for k, v in features.items() if k.startswith("original")}
        filtered["Paciente"] = os.path.splitext(file)[0].replace(".nii", "").replace(".gz", "")
        all_features.append(filtered)

    except Exception as e:
        print(f"[❌] Error con {file}: {e}")

# Guardar
df = pd.DataFrame(all_features)
df.set_index("Paciente", inplace=True)
df.to_csv(output_csv)

print(f"✅ Características guardadas en: {output_csv}")

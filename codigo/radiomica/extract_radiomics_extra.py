import os
import pandas as pd
from tqdm import tqdm
from radiomics import featureextractor
import argparse
import SimpleITK as sitk

# argumentss
parser = argparse.ArgumentParser(description="Extract PyRadiomics features from preprocessed NIfTI images and masks.")
parser.add_argument('--preproc_name', type=str, required=True,
                    help='Name of the preprocessing folder (e.g. resize_medium, resize_small, etc.)')
args = parser.parse_args()

PREPROC_NAME = args.preproc_name

#config
# IMAGES_DIR = "./data/full_data/datos_nifti"
# MASKS_DIR  = "./data/full_data/segmentaciones_nodulos"
# OUTPUT_NAME = "radiomics_features_original"

# preprocesado (nii.gz + nii.gz)
# PREPROC_NAME = "resize_medium"
IMAGES_DIR = f"./data/full_data/preprocesamientos/{PREPROC_NAME}/nifti/images"
MASKS_DIR  = f"./data/full_data/preprocesamientos/{PREPROC_NAME}/nifti/masks"
OUTPUT_NAME = f"./data/radiomic_data/extended_radiomics_features_{PREPROC_NAME}"

#parametros
params = {
    "binWidth": 25,
    "resampledPixelSpacing": None,
    "interpolator": "sitkBSpline",
    "enableCExtensions": True,
    "normalize": True,
    "removeOutliers": 3,
    "verbose": True,

    "imageType": {
        "Original": {},
        "Wavelet": {},
        "LoG": {"sigma": [1.0, 2.0, 3.0]},# tres escalas LoG
        "Square": {},
        "SquareRoot": {},
        "Exponential": {},
        "Logarithm": {},
        "Gradient": {},
        "LocalBinaryPattern3D": {}
    }
}

extractor = featureextractor.RadiomicsFeatureExtractor()

#settings
extractor.settings.update({
    'binWidth': 25,
    'normalize': True,
    'removeOutliers': 3,
    'verbose': True
})

#interpolación y resampling
extractor.settings['resampledPixelSpacing'] = None
extractor.settings['interpolator'] = 'sitkBSpline'

#tipos de imagen
extractor.enableImageTypes(
    Original={},
    Wavelet={},
    LoG={'sigma': [1.0, 2.0, 3.0]},
    Square={},
    SquareRoot={},
    Exponential={},
    Logarithm={},
    Gradient={},
    LBP2D={},
    LBP3D={}
)


def list_files_by_extension(folder, extensions):
    return sorted([
        f for f in os.listdir(folder)
        if any(f.lower().endswith(ext) for ext in extensions)
    ])

def strip_nii_extensions(filename):
    for ext in [".nii.gz", ".nii"]:
        if filename.endswith(ext):
            return filename[:-len(ext)]
    return filename

# images and masks

image_files = list_files_by_extension(IMAGES_DIR, [".nii.gz"])
mask_files  = list_files_by_extension(MASKS_DIR, [".nii", ".nii.gz"])

# creamos un índice por nombre base (sin extensión)
image_dict = {strip_nii_extensions(f): f for f in image_files}
mask_dict  = {strip_nii_extensions(f): f for f in mask_files}

# emparejamos por nombre base
common_keys = sorted(set(image_dict.keys()) & set(mask_dict.keys()))

print(f" Encontradas {len(common_keys)} parejas imagen-máscara.")

#extracion de features
results = []
for patient_id in tqdm(common_keys, desc="Extrayendo features"):
    image_path = os.path.join(IMAGES_DIR, image_dict[patient_id])
    mask_path  = os.path.join(MASKS_DIR, mask_dict[patient_id])

    # try:
    #     feature_vector = extractor.execute(image_path, mask_path)
    #     row = {"patient_id": patient_id}
    #     row.update(feature_vector)
    #     results.append(row)
    # except Exception as e:
    #     print(f"Error procesando {patient_id}: {e}")

    try:
        # leemos imagen y máscara con SimpleITK
        img = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
        arr = sitk.GetArrayFromImage(img)
        
        # 3D (monocanal)
        if arr.ndim == 3:
            features = extractor.execute(img, mask)
            row = {"patient_id": patient_id}
            row.update(features)
            results.append(row)

        # 4D (multicanal)
        elif arr.ndim == 4:
            num_channels = arr.shape[-1]
            print(f"Las dimensiones de la imagen son: {arr.shape} (canales: {num_channels})")

            combined_features = {"patient_id": patient_id}

            for c in range(num_channels):
                print(f"    ➜ Procesando canal {c}")

                # extraer canal correctamente
                img_channel = img[c, :, :, :]
                img_channel.CopyInformation(mask)

                print(f"    ➜ Canal {c} tiene shape: {img_channel.GetSize()}")

                # ejecutar PyRadiomics
                features_channel = extractor.execute(img_channel, mask)

                # añadir prefijo del canal
                for k, v in features_channel.items():
                    combined_features[f"ch{c}_{k}"] = v

            results.append(combined_features)


        else:
            print(f" Formato no soportado para {patient_id}: shape {arr.shape}")

    except Exception as e:
        print(f" Error procesando {patient_id}: {e}")

# guardamos
if results:
    df = pd.DataFrame(results)
    df.to_csv(f"{OUTPUT_NAME}.csv", index=False)
    print(" Features guardadas en radiomics_features.csv")
    print("Tamaño del DataFrame:", df.shape)
else:
    print(" No se generaron features.")

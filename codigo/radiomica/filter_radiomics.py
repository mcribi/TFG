#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold


def extract_labels_from_patient_id(pid):
    """
    Igual que LungCTDataset._parse_filename
    """
    fname = pid
    # Remove .nii.gz suffix if present
    if fname.endswith(".nii.gz"):
        fname = fname[:-7]
    elif fname.endswith(".nii"):
        fname = fname[:-4]

    # Extract sexo (first non-digit)
    for i, c in enumerate(fname):
        if not c.isdigit():
            sexo = c
            tumor_start = i + 1
            break
    else:
        raise ValueError(f"No se encontró sexo en el nombre: {pid}")

    tipo_complicacion = fname[-1]
    complicacion = fname[-2]
    tumor = fname[tumor_start:-2]

    return sexo, tumor, complicacion, tipo_complicacion

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if len(sys.argv) < 2:
    print("Usage: python preprocess_radiomics.py INPUT_FILE.csv")
    sys.exit(1)

INPUT_FILE = sys.argv[1]
OUTPUT_DIR = f"radiomic_data/cv/{os.path.splitext(os.path.basename(INPUT_FILE))[0]}"
ensure_dir(OUTPUT_DIR)

#load data
print(f"Loading {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
if "patient_id" not in df.columns:
    raise ValueError("Input file must contain a 'patient_id' column")

df = df.set_index("patient_id")
print(f"Shape after loading: {df.shape}")

# remove diagnostics_* COLUMNS
diagnostic_cols = [c for c in df.columns if c.startswith("diagnostics_") or "diagnostics" in c]
df = df.drop(columns=diagnostic_cols)
print(f" Removed {len(diagnostic_cols)} diagnostics columns. New shape: {df.shape}")

#add labels
print(f"Extracting labels from patient_id...")
labels = df.index.to_series().apply(lambda x: pd.Series(extract_labels_from_patient_id(x),
                                                        index=["label_sexo", "label_tumor", "label_complicacion", "label_tipo_complicacion"]))
df = pd.concat([df, labels], axis=1)
print(f" Labels added. Columns now: {df.columns.tolist()}")

#correlation plot
feature_cols = [c for c in df.columns if not c.startswith("label_")]
corr = df[feature_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title(f"Correlation plot for {INPUT_FILE}")
corrplot_file = f"plots/corrplot_{os.path.basename(INPUT_FILE)}.png"
plt.savefig(corrplot_file, dpi=300)
plt.close()
print(f" Correlation plot saved as {corrplot_file}")

#remove exactly duplicated features(corr=1 or -1)
to_drop = set()
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        if abs(corr.iloc[i, j]) == 1.0:
            to_drop.add(corr.columns[j])
if to_drop:
    print(f" Dropping {len(to_drop)} features with perfect correlation.")
    df = df.drop(columns=list(to_drop))
else:
    print(" No perfectly correlated feature pairs found.")

# save and clean dataset
clean_csv = os.path.join(OUTPUT_DIR, "cleaned_data.csv")
df.to_csv(clean_csv)
print(f"Cleaned CSV saved to {clean_csv}")
print(f" Final shape of dataset: {df.shape}")

# generate kfold splits
print(f"Generating StratifiedKFold splits...")
label_y = df["label_complicacion"]
X = df.drop(columns=["label_sexo", "label_tumor", "label_complicacion", "label_tipo_complicacion"])
y = label_y

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
    fold_dir = os.path.join(OUTPUT_DIR, f"fold_{fold}")
    ensure_dir(fold_dir)
    X.iloc[train_idx].assign(label_complicacion=y.iloc[train_idx]).to_csv(os.path.join(fold_dir, "train.csv"))
    X.iloc[test_idx].assign(label_complicacion=y.iloc[test_idx]).to_csv(os.path.join(fold_dir, "test.csv"))
    print(f"Fold {fold} saved to {fold_dir}")
    print(f"  Train size: {X.iloc[train_idx].shape}, Test size: {X.iloc[train_idx].shape}")
print(f"5-fold splits saved to {OUTPUT_DIR}")

#PCA variants
pca_thresholds = [0.99, 0.95, 0.9]
for threshold in pca_thresholds:
    print(f"\n Applying PCA with threshold {threshold}")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=threshold, svd_solver='full')
    X_pca = pca.fit_transform(X_scaled)

    pca_cols = [f"PCA_{i+1}" for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=X.index)
    df_pca = df_pca.assign(label_complicacion=y.values)

    pca_dir = f"{OUTPUT_DIR}_pca_{int(threshold*100)}"

    ensure_dir(pca_dir)

    for fold, (train_idx, test_idx) in enumerate(kf.split(df_pca[pca_cols], y), 1):
        fold_dir = os.path.join(pca_dir, f"fold_{fold}")
        ensure_dir(fold_dir)
        df_pca.iloc[train_idx].to_csv(os.path.join(fold_dir, "train.csv"))
        df_pca.iloc[test_idx].to_csv(os.path.join(fold_dir, "test.csv"))
        print(f"PCA-{threshold} Fold {fold} saved to {fold_dir}")
        print(f"  Train size: {df_pca.iloc[train_idx].shape}, Test size: {df_pca.iloc[test_idx].shape}")

    print(f" PCA-{threshold} folds saved to {pca_dir}")

# VarianceThreshold
print("\n Applying VarianceThreshold (threshold=0.01)...")
selector = VarianceThreshold(threshold=0.01)
X_sel = selector.fit_transform(X)
sel_cols = X.columns[selector.get_support()].tolist()

df_sel = pd.DataFrame(X_sel, columns=sel_cols, index=X.index)
df_sel = df_sel.assign(label_complicacion=y.values)

sel_dir = f"{OUTPUT_DIR}_varthresh_0.01"
ensure_dir(sel_dir)

for fold, (train_idx, test_idx) in enumerate(kf.split(df_sel[sel_cols], y), 1):
    fold_dir = os.path.join(sel_dir, f"fold_{fold}")
    ensure_dir(fold_dir)
    df_sel.iloc[train_idx].to_csv(os.path.join(fold_dir, "train.csv"))
    df_sel.iloc[test_idx].to_csv(os.path.join(fold_dir, "test.csv"))
    print(f" VarianceThreshold Fold {fold} saved to {fold_dir}")
    print(f"  Train size: {df_sel.iloc[train_idx].shape}, Test size: {df_sel.iloc[test_idx].shape}")

print(f" VarianceThreshold folds saved to {sel_dir}")

print("\n All done.")

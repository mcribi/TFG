#!/usr/bin/env python3

import os
import sys
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from metric_learn import LMNN

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if len(sys.argv) < 6:
    print("Usage: python apply_metric_learning_single.py <INPUT_EXPERIMENT_DIR> <METHOD> <N_COMPONENTS> <SCALER> <OUTPUT_ROOT>")
    print("Example: python apply_metric_learning_single.py radiomic_data/cv/experiment_1 nca 20 StandardScaler radiomic_data/cv_metric")
    sys.exit(1)

INPUT_EXPERIMENT_DIR = sys.argv[1]
METHOD = sys.argv[2].lower()
N_COMPONENTS_RAW = sys.argv[3]
SCALER_CHOICE = sys.argv[4]
OUTPUT_ROOT = sys.argv[5]
LMNN_NEIGHBORS = int(sys.argv[6]) if len(sys.argv) > 6 else 7

RANDOM_STATE = 42

#check method
if METHOD not in ["nca", "lmnn"]:
    raise ValueError(f"Unsupported method: {METHOD}")

if N_COMPONENTS_RAW.lower() == 'all':
    N_COMPONENTS = None
else:
    N_COMPONENTS = int(N_COMPONENTS_RAW)


experiment_name = os.path.basename(INPUT_EXPERIMENT_DIR.rstrip('/'))
OUTPUT_EXPERIMENT_DIR = os.path.join(
    OUTPUT_ROOT,
    f"{experiment_name}_{METHOD}_{N_COMPONENTS_RAW}_{SCALER_CHOICE}_{LMNN_NEIGHBORS}"
)

print(f" INPUT: {INPUT_EXPERIMENT_DIR}")
print(f" METHOD: {METHOD.upper()}")
print(f" N_COMPONENTS: {N_COMPONENTS}")
print(f" OUTPUT: {OUTPUT_EXPERIMENT_DIR}")
print(f" SCALER: {SCALER_CHOICE}")
print(f" LMNN_NEIGHBORS: {LMNN_NEIGHBORS}")

#process folds
for fold_name in sorted(os.listdir(INPUT_EXPERIMENT_DIR)):
    fold_in_path = os.path.join(INPUT_EXPERIMENT_DIR, fold_name)
    if not os.path.isdir(fold_in_path):
        continue

    print(f"\n Fold: {fold_name}")
    train_file = os.path.join(fold_in_path, "train.csv")
    test_file = os.path.join(fold_in_path, "test.csv")

    if not (os.path.exists(train_file) and os.path.exists(test_file)):
        print(f" Missing train/test in {fold_name}, skipping.")
        continue

    df_train = pd.read_csv(train_file, index_col=0)
    df_test = pd.read_csv(test_file, index_col=0)

    target_col = "label_complicacion"
    feature_cols = [col for col in df_train.columns if not col.startswith("label_")]

    # data
    X_train_raw = df_train[feature_cols].values
    y_train = (df_train[target_col].values == 'S').astype(int)
    X_test_raw = df_test[feature_cols].values
    y_test = (df_test[target_col].values == 'S').astype(int)

    # Standard scaling
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train_raw)
    # X_test = scaler.transform(X_test_raw)

    # Apply chosen scaler
    if SCALER_CHOICE == "StandardScaler":
        scaler = StandardScaler()
    elif SCALER_CHOICE == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif SCALER_CHOICE.lower() in ["none", "null", ""]:
        scaler = None
    else:
        raise ValueError(f"Unsupported scaler: {SCALER_CHOICE}")

    if scaler is not None:
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
    else:
        X_train = X_train_raw
        X_test = X_test_raw


    # Fit model
    if METHOD == "nca":
        model = NeighborhoodComponentsAnalysis(
            n_components=N_COMPONENTS,
            random_state=RANDOM_STATE
        )
    elif METHOD == "lmnn":
        model = LMNN(n_neighbors=LMNN_NEIGHBORS, learn_rate=1e-6, n_components=N_COMPONENTS)

    print(f"   ➜ Fitting {METHOD.upper()}...")
    model.fit(X_train, y_train)
    X_train_trans = model.transform(X_train)
    X_test_trans = model.transform(X_test)

    # Save transformed data
    fold_out_path = os.path.join(OUTPUT_EXPERIMENT_DIR, fold_name)
    ensure_dir(fold_out_path)

    train_out = pd.DataFrame(
        X_train_trans,
        index=df_train.index,
        columns=[f"{METHOD}_feat_{i+1}" for i in range(X_train_trans.shape[1])]
    )
    # train_out[target_col] = y_train
    train_out[target_col] = df_train[target_col].values
    train_out.to_csv(os.path.join(fold_out_path, "train.csv"))

    test_out = pd.DataFrame(
        X_test_trans,
        index=df_test.index,
        columns=[f"{METHOD}_feat_{i+1}" for i in range(X_test_trans.shape[1])]
    )
    # test_out[target_col] = y_test
    test_out[target_col] = df_test[target_col].values
    test_out.to_csv(os.path.join(fold_out_path, "test.csv"))

    print(f" Saved transformed fold to {fold_out_path}")

print("\n All folds processed with metric learning transformation")

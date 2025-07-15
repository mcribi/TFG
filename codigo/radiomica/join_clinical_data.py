#!/usr/bin/env python3
import os
import sys
import pandas as pd

if len(sys.argv) < 3:
    print("Usage: python add_clinical_data.py <CLINICAL_CSV> <CV_INPUT_DIR> <CV_OUTPUT_DIR>")
    print("Example: python add_clinical_data.py clinical_data.csv radiomic_data/cv radiomic_data/cv_clinic")
    sys.exit(1)

CLINICAL_CSV = sys.argv[1]
CV_INPUT_DIR = sys.argv[2]
CV_OUTPUT_DIR = sys.argv[3]

# load clinical data
print(f"Loading clinical data from {CLINICAL_CSV}...")
clinical_df = pd.read_csv(CLINICAL_CSV)
if "patient_id" not in clinical_df.columns:
    raise ValueError("Clinical CSV must have a 'patient_id' column.")

clinical_df = clinical_df.set_index("patient_id")
print(f"Clinical data shape: {clinical_df.shape}")

# walk through all experiments

for experiment_name in sorted(os.listdir(CV_INPUT_DIR)):
    experiment_in_path = os.path.join(CV_INPUT_DIR, experiment_name)
    if not os.path.isdir(experiment_in_path):
        continue

    experiment_out_path = os.path.join(CV_OUTPUT_DIR, experiment_name)
    os.makedirs(experiment_out_path, exist_ok=True)

    print(f"\n Processing experiment: {experiment_name}")

    # for each fold
    for fold_name in sorted(os.listdir(experiment_in_path)):
        fold_in_path = os.path.join(experiment_in_path, fold_name)
        if not os.path.isdir(fold_in_path):
            continue

        fold_out_path = os.path.join(experiment_out_path, fold_name)
        os.makedirs(fold_out_path, exist_ok=True)

        print(f" Fold: {fold_name}")

        for split_file in ["train.csv", "test.csv"]:
            split_in_file = os.path.join(fold_in_path, split_file)
            if not os.path.exists(split_in_file):
                continue

            df_split = pd.read_csv(split_in_file, index_col=0)

            print(f" {split_file} before merge: {df_split.shape}")

            # merge on patient_id (index)
            df_merged = df_split.merge(
                clinical_df,
                left_index=True,
                right_index=True,
                how="left"
            )

            print(f"  {split_file} after merge: {df_merged.shape}")

            # Save
            split_out_file = os.path.join(fold_out_path, split_file)
            df_merged.to_csv(split_out_file)

print("\n All partitions processed and saved with clinical data added")

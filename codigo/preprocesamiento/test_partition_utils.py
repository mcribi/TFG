import os
import shutil
from CT_partitioning import create_partitions, extract_labels_from_filename
from glob import glob

NIFTI_DIR = "../data/datos_nifti"
PARTITIONS_DIR = "../data/partitions"

def clean_dirs():
    if os.path.exists(PARTITIONS_DIR):
        shutil.rmtree(PARTITIONS_DIR)

def count_files_recursively(folder):
    print(f"🔎 Explorando: {folder}")
    total = 0
    for root, _, files in os.walk(folder, followlinks=True):  # 👈 followlinks añadido
        for f in files:
            if f.endswith(".nii.gz"):
                # print(f"✅ Encontrado: {os.path.join(root, f)}")
                total += 1
    return total


def run_tests():
    total_original = len([f for f in os.listdir(NIFTI_DIR) if f.endswith('.nii.gz')])
    print(f"📂 Total archivos originales: {total_original}")

    test_cases = [
        {
            'split_type': 'single_train_test',
            'kwargs': {'test_size': 0.25}
        },
        {
            'split_type': 'single_train_val_test',
            'kwargs': {'test_size': 0.2, 'val_size': 0.1}
        },
        {
            'split_type': 'kfold_train_test',
            'kwargs': {'k': 3}
        },
        {
            'split_type': 'kfold_train_val_test',
            'kwargs': {'k': 4, 'test_size': 0.2, 'val_size': 0.1}
        }
    ]

    stratify_combinations = [
        ['complicacion'],
        ['sexo'],
        ['complicacion', 'sexo']
    ]

    expected_total = 0

    for case in test_cases:
        for strat_keys in stratify_combinations:
            print(f"\n🧪 Testing: {case['split_type']} with stratify_keys = {strat_keys}")
            try:
                output = create_partitions(
                    nifti_dir=NIFTI_DIR,
                    output_dir=PARTITIONS_DIR,
                    split_type=case['split_type'],
                    seed=42,
                    stratify_keys=strat_keys,
                    use_symlinks=True,
                    **case['kwargs']
                )

                # Contar archivos generados en esta partición
                print(output)
                count = count_files_recursively(output)
                expected_total += count
                print(f"   Archivos generados: {count}")

            except ValueError as e:
                print(f"⚠️ Error: {e}")

    actual_total = count_files_recursively(PARTITIONS_DIR)
    print(f"\n📊 Total archivos realmente generados: {actual_total}")
    print(f"📊 Total archivos esperados (sumados por split): {expected_total}")

    assert actual_total == expected_total, "❌ Número de archivos final no coincide con suma parcial"
    print("✅ Test de integridad PASADO")

if __name__ == "__main__":
    clean_dirs()
    run_tests()

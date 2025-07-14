import os
import shutil
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from collections import defaultdict
from itertools import product
from tqdm import tqdm

def extract_labels_from_filename(filename):
    fname = filename
    if fname.endswith(".nii.gz"):
        fname = fname[:-7]
    elif fname.endswith(".nii"):
        fname = fname[:-4]

    for i, c in enumerate(fname):
        if not c.isdigit():
            sexo = c
            start_tumor = i + 1
            break
    else:
        raise ValueError(f"No se encontró sexo en el nombre: {filename}")

    tipo_complicacion = fname[-1]
    complicacion = fname[-2]
    tumor = fname[start_tumor:-2]

    return {
        'sexo': sexo,
        'tumor': tumor,
        'complicacion': complicacion,
        'tipo_complicacion': tipo_complicacion
    }

def format_split_name(split_type, seed, stratify_keys=None, test_size=None, val_size=None, k=None):
    parts = [split_type, f"seed_{seed}"]
    if stratify_keys:
        parts.append(f"stratify_[{','.join(stratify_keys)}]")
    if test_size is not None:
        parts.append(f"test_{test_size}")
    if val_size is not None:
        parts.append(f"val_{val_size}")
    if k is not None:
        parts.append(f"{k}fold")
    return "_".join(parts)

def create_partitions(
    nifti_dir,
    output_dir,
    split_type='single_train_test',
    seed=42,
    test_size=0.2,
    val_size=0.1,
    k=5,
    stratify_keys=['complicacion'],
    use_symlinks=True
):
    random.seed(seed)
    np.random.seed(seed)

    files = sorted([f for f in os.listdir(nifti_dir) if f.endswith('.nii.gz')])
    files = list(sorted(set(files)))
    full_paths = [os.path.join(nifti_dir, f) for f in files]

    labels = [extract_labels_from_filename(os.path.basename(f)) for f in full_paths]
    stratify_values = [
    '_'.join(str(label[k]) for k in stratify_keys)
    for label in labels
    ]



    if split_type == 'single_train_test':
        return _split_single_train_test(files, stratify_values, output_dir, test_size, seed, use_symlinks, nifti_dir, stratify_keys)
    elif split_type == 'single_train_val_test':
        return _split_single_train_val_test(files, stratify_values, output_dir, test_size, val_size, seed, use_symlinks, nifti_dir, stratify_keys)
    elif split_type == 'kfold_train_test':
        return _split_kfold_train_test(files, stratify_values, output_dir, k, seed, use_symlinks, nifti_dir, stratify_keys)
    elif split_type == 'kfold_train_val_test':
        return _split_kfold_train_val_test(files, stratify_values, output_dir, k, test_size, val_size, seed, use_symlinks, nifti_dir, stratify_keys)
    else:
        raise ValueError(f"split_type no reconocido: {split_type}")

def _safe_link_or_copy(src, dst, use_symlinks):
    if os.path.exists(dst):
        return
    if use_symlinks:
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)

def _split_single_train_test(files, stratify_values, output_dir, test_size, seed, use_symlinks, nifti_dir, stratify_keys):
    train_idx, test_idx = train_test_split(
        range(len(files)),
        test_size=test_size,
        random_state=seed,
        stratify=stratify_values
    )

    split_name = format_split_name("single_train_test_split", seed, stratify_keys, test_size)
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    for split_name, idxs in [('train', train_idx), ('test', test_idx)]:
        split_path = os.path.join(split_dir, split_name)
        os.makedirs(split_path, exist_ok=True)
        for i in idxs:
            src = os.path.join(nifti_dir, files[i])
            dst = os.path.join(split_path, files[i])
            _safe_link_or_copy(src, dst, use_symlinks)

    return split_dir

def _split_single_train_val_test(files, stratify_values, output_dir, test_size, val_size, seed, use_symlinks, nifti_dir, stratify_keys):
    trainval_idx, test_idx = train_test_split(
        range(len(files)),
        test_size=test_size,
        random_state=seed,
        stratify=stratify_values
    )
    trainval_strat = [stratify_values[i] for i in trainval_idx]
    rel_val_size = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=rel_val_size,
        random_state=seed,
        stratify=trainval_strat
    )

    split_name = format_split_name("single_train_val_test_split", seed, stratify_keys, test_size, val_size)
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    for split_name, idxs in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        split_path = os.path.join(split_dir, split_name)
        os.makedirs(split_path, exist_ok=True)
        for i in idxs:
            src = os.path.join(nifti_dir, files[i])
            dst = os.path.join(split_path, files[i])
            _safe_link_or_copy(src, dst, use_symlinks)

    return split_dir

def _split_kfold_train_test(files, stratify_values, output_dir, k, seed, use_symlinks, nifti_dir, stratify_keys):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    split_name = format_split_name("kfold_train_test_split", seed, stratify_keys, k=k)
    split_dir = os.path.join(output_dir, split_name)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(files, stratify_values), start=1):
        for split_name, idxs in [('train', train_idx), ('test', test_idx)]:
            fold_path = os.path.join(split_dir, split_name, f"fold_{fold_idx}")
            os.makedirs(fold_path, exist_ok=True)
            for i in idxs:
                src = os.path.join(nifti_dir, files[i])
                dst = os.path.join(fold_path, files[i])
                _safe_link_or_copy(src, dst, use_symlinks)

    return split_dir

def _split_kfold_train_val_test(files, stratify_values, output_dir, k, test_size, val_size, seed, use_symlinks, nifti_dir, stratify_keys):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    split_name = format_split_name("kfold_train_val_test_split", seed, stratify_keys, test_size, val_size, k)
    split_dir = os.path.join(output_dir, split_name)

    for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(files, stratify_values), start=1):
        trainval_strat = [stratify_values[i] for i in trainval_idx]
        rel_val_size = val_size / (1.0 - test_size)
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=rel_val_size,
            random_state=seed,
            stratify=trainval_strat
        )

        for split_name, idxs in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
            fold_path = os.path.join(split_dir, split_name, f"fold_{fold_idx}")
            os.makedirs(fold_path, exist_ok=True)
            for i in idxs:
                src = os.path.join(nifti_dir, files[i])
                dst = os.path.join(fold_path, files[i])
                _safe_link_or_copy(src, dst, use_symlinks)

    return split_dir

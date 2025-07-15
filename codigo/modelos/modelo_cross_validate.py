import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR

from monai.networks.nets import DenseNet121
from monai.transforms import Compose, EnsureType

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from collections import defaultdict

#config
device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 30
# WEIGHT_DECAY = 1e-5

#leer datos
df = pd.read_csv("./../../datos_clinicos/datos_clinicos.csv", na_values="NaN")
labels_dict_numeric = {
    k: 1 if str(v).strip().lower() in ['sí', 'si', 'yes', '1', 'true'] else 0
    for k, v in dict(zip(df['Id_paciente'], df['Complicación'])).items()
}

#gradcam
def get_last_conv_layer(model):
    for layer in reversed(model.features):
        if isinstance(layer, nn.Conv3d):
            return layer
        if hasattr(layer, 'layers'):
            for sub in reversed(layer.layers):
                if isinstance(sub, nn.Conv3d):
                    return sub
    raise ValueError("No se encontró una capa Conv3d válida en el modelo")


def apply_gradcam(model, volume, label, device, output_base_dir, example_id=""):
    model.train()
    target_layers = [get_last_conv_layer(model)]
    cam = GradCAM(model=model, target_layers=target_layers)

    input_tensor = volume.to(device)
    targets = [ClassifierOutputTarget(label)]

    with torch.enable_grad():
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    for idx in range(0, grayscale_cam.shape[0], max(1, grayscale_cam.shape[0] // 6)):
        plt.figure(figsize=(6, 6))
        img = input_tensor.cpu().numpy()[0, 0, idx]
        norm_img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        heatmap = grayscale_cam[idx]
        rgb_img = np.repeat(norm_img[..., np.newaxis], 3, axis=-1)
        result = show_cam_on_image(rgb_img.astype(np.float32), heatmap, use_rgb=True)

        os.makedirs(output_base_dir, exist_ok=True)
        path = os.path.join(output_base_dir, f"{example_id}_slice_{idx}.png")
        plt.imsave(path, result)
        plt.close()


#visualiza solo unos cuantos
# def visualizar_gradcams(model, dataset, val_idx, device, output_base_dir):
#     print("Generando ejemplos de Grad-CAM...")
#     correctos = defaultdict(list)
#     incorrectos = defaultdict(list)

#     loader = DataLoader(Subset(dataset, val_idx), batch_size=1, shuffle=False)
#     model.eval()

#     for i, (vol, label) in enumerate(loader):
#         vol = vol.to(device)
#         real = label.item()
#         pred = torch.argmax(model(vol), dim=1).item()

#         patient_id = dataset.patients[val_idx[i]]

#         if pred == real:
#             correctos[real].append( (vol, patient_id, pred) )
#         else:
#             incorrectos[real].append( (vol, patient_id, pred) )

#     # Generar hasta 2 ejemplos por clase
#     for clase in [0, 1]:
#         for (v, pid, pred) in correctos[clase][:2]:
#             example_id = f"label{clase}_correcto_pred{pred}_pid{pid}"
#             apply_gradcam(model, v, clase, device, output_base_dir, example_id=example_id)

#         for (v, pid, pred) in incorrectos[clase][:2]:
#             example_id = f"label{clase}_fallo_pred{pred}_pid{pid}"
#             apply_gradcam(model, v, clase, device, output_base_dir, example_id=example_id)

def visualizar_gradcams(model, dataset, val_idx, device, output_base_dir):
    print("Generando ejemplos de Grad-CAM...")
    correctos = defaultdict(list)
    incorrectos = defaultdict(list)

    loader = DataLoader(Subset(dataset, val_idx), batch_size=1, shuffle=False)
    model.eval()

    for i, (vol, label) in enumerate(loader):
        vol = vol.to(device)
        real = label.item()
        pred = torch.argmax(model(vol), dim=1).item()

        patient_id = dataset.patients[val_idx[i]]

        if pred == real:
            correctos[real].append( (vol, patient_id, pred) )
        else:
            incorrectos[real].append( (vol, patient_id, pred) )

    #  generamos todos los ejemplos 
    for clase in [0, 1]:
        for (v, pid, pred) in correctos[clase]:
            example_id = f"label{clase}_correcto_pred{pred}_pid{pid}"
            apply_gradcam(model, v, clase, device, output_base_dir, example_id=example_id)

        for (v, pid, pred) in incorrectos[clase]:
            example_id = f"label{clase}_fallo_pred{pred}_pid{pid}"
            apply_gradcam(model, v, clase, device, output_base_dir, example_id=example_id)


#dataset
class LungCTDataset(Dataset):
    def __init__(self, data_dir, mask_dir, labels_dict, transform=None):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.patients = list(labels_dict.keys())
        self.labels = labels_dict
        self.transform = transform

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_id = self.patients[idx]
        volume = np.load(os.path.join(self.data_dir, f"{patient_id}.npy"))

        mask_path = os.path.join(self.mask_dir, f"{patient_id}.npy")
        if os.path.exists(mask_path):
            mask = np.load(mask_path).astype(bool)
            if mask.shape == volume.shape:
                volume = volume * mask

        volume = np.expand_dims(volume, axis=0)
        if self.transform:
            volume = self.transform(volume)
        label = self.labels[patient_id]
        return volume, torch.tensor(label, dtype=torch.long)

transform = Compose([EnsureType()])

#modelo
def build_model(dropout_prob):
    return DenseNet121(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        dropout_prob=dropout_prob
    )

# train
def train_model_with_internal_validation(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    save_path
):
    # scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_gmean = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        print(f"\n Epoch {epoch+1}/{epochs}")
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        tpr, tnr, gmean = calcular_metricas_binarias(all_labels, all_preds)
        print(f" Entrenamiento — Loss: {running_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | G-Mean: {gmean:.4f}")

        # val
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for val_inputs, val_labels_batch in val_loader:
                val_inputs, val_labels_batch = val_inputs.to(device), val_labels_batch.to(device)
                val_outputs = model(val_inputs)
                batch_loss = criterion(val_outputs, val_labels_batch)
                val_loss += batch_loss.item()

                val_preds_batch = torch.argmax(val_outputs, dim=1).cpu().numpy()
                val_preds.extend(val_preds_batch)
                val_labels.extend(val_labels_batch.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        val_tpr, val_tnr, val_gmean = calcular_metricas_binarias(val_labels, val_preds)

        print(f" Validación   — Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | G-Mean: {val_gmean:.4f}")

        # scheduler.step()

        if epoch == 0 or val_gmean > best_val_gmean:
            best_val_gmean = val_gmean
            torch.save(model.state_dict(), save_path)
            print(f" Guardado modelo con mejor G-Mean ({val_gmean:.4f}) en '{save_path}'")

#metricas
def calcular_metricas_binarias(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return 0, 0, 0
    TN, FP, FN, TP = cm.ravel()
    tpr = TP / (TP + FN) if (TP + FN) else 0
    tnr = TN / (TN + FP) if (TN + FP) else 0
    gmean = np.sqrt(tpr * tnr)
    return tpr, tnr, gmean

def evaluar_modelo(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    tpr, tnr, gmean = calcular_metricas_binarias(all_labels, all_preds)
    return acc, f1, tpr, tnr, gmean

#cv
def cross_validate(
    model_class,
    dataset,
    batch_size,
    learning_rate,
    preprocessed_dir,
    k=5,
    device='cuda',
    epochs=10,
    save_path_prefix="modelo_densenet_param_now_solomini_otras_seeds_", 
    weight_decay=1e-5,
    dropout_prob=0.4, 
    seed=42
):
    results = []
    labels = [dataset.labels[pid] for pid in dataset.patients]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(dataset.patients, labels)):
        print("\n===========================================")
        print(f" Fold {fold+1}/{k} | BS={batch_size} | LR={learning_rate}")
        print("===========================================")

        train_val_subset = Subset(dataset, train_val_idx)
        test_subset = Subset(dataset, test_idx)

        # Internal split 80/20
        internal_labels = [dataset.labels[dataset.patients[i]] for i in train_val_idx]
        train_indices, val_indices = train_test_split(
            train_val_idx,
            test_size=0.2,
            stratify=internal_labels,
            random_state=seed
        )
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)

        model = model_class(dropout_prob).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        save_path = f"{save_path_prefix}_bs{batch_size}_lr{learning_rate}_fold{fold+1}.pth"

        # train con validacion interna
        train_model_with_internal_validation(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            epochs,
            save_path
        )

        # evaluacion final en validacion (mejor modelo)
        model.load_state_dict(torch.load(save_path))
        val_acc, val_f1, val_tpr, val_tnr, val_gmean = evaluar_modelo(model, val_loader, device)
        test_acc, test_f1, test_tpr, test_tnr, test_gmean = evaluar_modelo(model, test_loader, device)

        results.append({
            "fold": fold + 1,
            "set": "VALIDATION",
            "accuracy": val_acc,
            "f1": val_f1,
            "tpr": val_tpr,
            "tnr": val_tnr,
            "gmean": val_gmean,
            "batch_size": batch_size,
            "learning_rate": learning_rate, 
            "seed": seed
        })
        results.append({
            "fold": fold + 1,
            "set": "TEST",
            "accuracy": test_acc,
            "f1": test_f1,
            "tpr": test_tpr,
            "tnr": test_tnr,
            "gmean": test_gmean,
            "batch_size": batch_size,
            "learning_rate": learning_rate, 
            "seed": seed
        })

        # visualizamos gradcam
        gradcam_output_dir = f"gradcamConID_outputs_param_k5_now_solomini_otras_seed_titan_/{preprocessed_dir}/fold_{fold+1}_bs{batch_size}_lr{learning_rate}_seed{seed}"
        visualizar_gradcams(model, dataset, test_idx, device, gradcam_output_dir)

    results_df = pd.DataFrame(results)

    # añadimos medias
    for split in ["VALIDATION", "TEST"]:
        means = results_df[results_df["set"] == split][["accuracy", "f1", "tpr", "tnr", "gmean"]].mean()
        mean_row = {
            "fold": "MEAN",
            "set": split,
            "accuracy": means["accuracy"],
            "f1": means["f1"],
            "tpr": means["tpr"],
            "tnr": means["tnr"],
            "gmean": means["gmean"],
            "batch_size": batch_size,
            "learning_rate": learning_rate, 
            "seed": seed
        }
        results_df = pd.concat([results_df, pd.DataFrame([mean_row])], ignore_index=True)

    return results_df

#run grid
# preprocessing_dirs = [
#     "resize_smaller_hu_m300_1400_separadas",
#     "resize_smaller_hu_m600_1500_separadas",
#     "resize_mini_hu_m300_1400_separadas",
#     "resize_mini_hu_m600_1500_separadas"
# ]

# preprocessing_dirs = [
#     "resize_small_hu_m600_1500_separadas",
#     "resize_small_hu_m300_1400_separadas"
# ]

preprocessing_dirs = [
    "resize_mini_hu_m300_1400_separadas",
    "resize_mini_hu_m600_1500_separadas"
]

# batch_sizes_to_try = [2, 4, 8, 16]
# learning_rates_to_try = [1e-5, 1e-4, 1e-3]

# batch_sizes_to_try = [4, 8, 16]
# learning_rates_to_try = [1e-3]
# weight_decays_to_try = [0, 1e-5]
# dropout_probs_to_try = [0, 0.2, 0.4]

batch_sizes_to_try = [4, 16]
learning_rates_to_try = [1e-3]
weight_decays_to_try = [0]
dropout_probs_to_try = [0.3]
seeds_to_try = [8, 40, 123, 777]



all_results = []

for seed in seeds_to_try:
    print("\n##################################################")
    print(f"=== EMPEZANDO EXPERIMENTOS CON SEED: {seed} ===")
    print("##################################################")

    for prep in preprocessing_dirs:
        print("\n##################################################")
        print(f"=== PREPROCESADO: {prep} | SEED={seed} ===")
        print("##################################################")

        data_dir = f"/mnt/homeGPU/mcribilles/TFG/volumenes/preprocesados/preprocesamientos_interesantes/{prep}/npy/images"
        mask_dir = f"/mnt/homeGPU/mcribilles/TFG/volumenes/preprocesados/preprocesamientos_interesantes/{prep}/npy/masks"

        dataset = LungCTDataset(data_dir, mask_dir, labels_dict_numeric, transform=transform)

        for bs in batch_sizes_to_try:
            for lr in learning_rates_to_try:
                for weight_decay in weight_decays_to_try:
                    for dropout_prob in dropout_probs_to_try:

                        print("\n##################################################")
                        print(f"SEED={seed} | Preprocesado={prep} | BS={bs}, LR={lr}, WD={weight_decay}, Dropout={dropout_prob}, Seed={seed}")
                        print("##################################################")

                        # cv
                        df_result = cross_validate(
                            model_class=build_model,
                            dataset=dataset,
                            batch_size=bs,
                            learning_rate=lr,
                            preprocessed_dir=prep,
                            k=5,
                            device=device,
                            epochs=EPOCHS,
                            save_path_prefix=f"modelo_densenet_param_k5_cv_now_solomini_con_otras_seeds_titan_{prep}_bs{bs}_lr{lr}_wd{weight_decay}_drop{dropout_prob}_seed{seed}",
                            weight_decay=weight_decay,
                            dropout_prob=dropout_prob,
                            seed=seed  
                        )

                        df_result["preprocessed_dir"] = prep
                        df_result["weight_decay"] = weight_decay
                        df_result["dropout_prob"] = dropout_prob
                        df_result["seed"] = seed
                        all_results.append(df_result)

final_results = pd.concat(all_results, ignore_index=True)
final_results.to_csv("resultados_param_k5_final_solomini_con_otras_seeds_titan.csv", index=False)
print("\n Todos los resultados guardados en 'resultados_param_k5_final_solomini_con_otras_seeds_titan.csv'")

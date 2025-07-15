import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR

from monai.networks.nets import DenseNet121
from monai.transforms import Compose, EnsureType

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


# config
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5


# leemos datos clinicos
df = pd.read_csv("./../../datos_clinicos/datos_clinicos.csv", na_values="NaN")
labels_dict_numeric = {
    k: 1 if str(v).strip().lower() in ['sí', 'si', 'yes', '1', 'true'] else 0 
    for k, v in dict(zip(df['Id_paciente'], df['Complicación'])).items()
}

# paths
data_dir = "../../volumenes/preprocesados/preprocesamientos_chiquitos/resize_mini_hu_m300_1400_separadas/npy/images"
mask_dir = "../../volumenes/preprocesados/preprocesamientos_chiquitos/resize_mini_hu_m300_1400_separadas/npy/masks"

# chequeamos numero de archivos
print(f"Número de volúmenes en IMAGES: {len([f for f in os.listdir(data_dir) if f.endswith('.npy')])}")
print(f"Número de volúmenes en MASKS: {len([f for f in os.listdir(mask_dir) if f.endswith('.npy')])}")


# dataset
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

        # aplicamos mascaras si existe
        mask_path = os.path.join(self.mask_dir, f"{patient_id}.npy")
        if os.path.exists(mask_path):
            mask = np.load(mask_path).astype(bool)
            if mask.shape == volume.shape:
                volume = volume * mask
            else:
                raise ValueError(f"Dimensiones no coinciden para {patient_id}: {volume.shape} vs {mask.shape}")
        else:
            print(f"No se encontró máscara para {patient_id}, usando volumen completo.")

        # formato para pytorch
        volume = np.expand_dims(volume, axis=0)
        if self.transform:
            volume = self.transform(volume)
        label = self.labels[patient_id]
        return volume, torch.tensor(label, dtype=torch.long)


# transformaciones
transform = Compose([EnsureType()])
dataset = LungCTDataset(data_dir, mask_dir, labels_dict_numeric, transform=transform)


# cplit
labels = [dataset.labels[pid] for pid in dataset.patients]
train_val_pids, test_pids = train_test_split(
    dataset.patients,
    test_size=0.1,
    stratify=labels,
    random_state=42
)

pid_to_index = {pid: idx for idx, pid in enumerate(dataset.patients)}
train_val_idx = [pid_to_index[pid] for pid in train_val_pids]
test_idx = [pid_to_index[pid] for pid in test_pids]

train_val_dataset = Subset(dataset, train_val_idx)
test_dataset = Subset(dataset, test_idx)

print(f"Total pacientes: {len(dataset)}")
print(f"Train+Val (para CV): {len(train_val_dataset)}")
print(f"Test hold-out: {len(test_dataset)}")


#modelo
def build_model():
    return DenseNet121(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        dropout_prob=0.5
    )

#train
def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    device,
    epochs=10,
    save_path=None
):
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_gmean = 0.0
    lrs = []

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

        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        tpr, tnr, gmean = calcular_metricas_binarias(all_labels, all_preds)

        print(f" Entrenamiento — Loss: {running_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | TPR: {tpr:.4f} | TNR: {tnr:.4f} | G-Mean: {gmean:.4f} | LR: {current_lr}")

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

        print(f" Validación   — Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | TPR: {val_tpr:.4f} | TNR: {val_tnr:.4f} | G-Mean: {val_gmean:.4f}")

        scheduler.step()

        if save_path and val_gmean > best_val_gmean:
            best_val_gmean = val_gmean
            torch.save(model.state_dict(), save_path)
            print(f" Guardado modelo con mejor G-Mean ({val_gmean:.4f}) en '{save_path}'")

    # evaluacion final
    if save_path:
        model.load_state_dict(torch.load(save_path))
        print(f"\n Cargado mejor modelo desde '{save_path}'")
        print("\n Evaluación FINAL en VALIDATION (mejor modelo):")
        evaluar_modelo_final(model, val_loader, device, dataset_name="VALIDATION")
        print("\n Evaluación FINAL en TEST (mejor modelo):")
        evaluar_modelo_final(model, test_loader, device, dataset_name="TEST")

    # Plot del LR
    plt.figure(figsize=(6, 4))
    plt.plot(lrs, marker='o')
    plt.title("Evolución del Learning Rate")
    plt.xlabel("Época")
    plt.ylabel("Learning Rate")
    plt.grid()
    plt.show()


# metricas
def calcular_metricas_binarias(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return 0, 0, 0
    TN, FP, FN, TP = cm.ravel()
    tpr = TP / (TP + FN) if (TP + FN) else 0
    tnr = TN / (TN + FP) if (TN + FP) else 0
    gmean = np.sqrt(tpr * tnr)
    return tpr, tnr, gmean


def evaluar_modelo_final(model, data_loader, device, dataset_name=""):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["No Complicación", "Complicación"])
    disp.plot(cmap='Blues')
    plt.title(f"Matriz de Confusión ({dataset_name})")
    plt.show()

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    tpr, tnr, gmean = calcular_metricas_binarias(all_labels, all_preds)

    print(f"\n RESULTADOS en {dataset_name}:")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   F1 Score: {f1:.4f}")
    print(f"   TPR: {tpr:.4f}")
    print(f"   TNR: {tnr:.4f}")
    print(f"   G-Mean: {gmean:.4f}")


#cv
def cross_validate(
    model_class,
    train_val_dataset,
    test_dataset,
    k=5,
    device='cuda',
    epochs=10,
    save_path_prefix="modelo_densenet_fold"
):
    labels = [dataset.labels[dataset.patients[i]] for i in train_val_dataset.indices]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_dataset.indices, labels)):
        print(f"\n==========================\n Fold {fold+1}/{k}\n==========================")

        train_abs_idx = [train_val_dataset.indices[i] for i in train_idx]
        val_abs_idx = [train_val_dataset.indices[i] for i in val_idx]

        train_subset = Subset(dataset, train_abs_idx)
        val_subset = Subset(dataset, val_abs_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        model = model_class().to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        #optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        save_path = f"{save_path_prefix}_fold{fold+1}.pth"

        train_model(
            model,
            train_loader,
            val_loader,
            test_loader,
            criterion,
            optimizer,
            device,
            epochs=epochs,
            save_path=save_path
        )


# ejecutamos
cross_validate(build_model, train_val_dataset, test_dataset, k=5, device=device, epochs=EPOCHS)

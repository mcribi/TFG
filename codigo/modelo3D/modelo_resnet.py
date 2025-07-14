# -------------------- IMPORTS --------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR

from monai.networks.nets import ResNet
from monai.transforms import (
    Compose, EnsureType, RandFlip, RandRotate90, RandAffine
)

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


# -------------------- CONFIG --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16           #cuanto más pequeño mas regulariza
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5       # weight decay para regularizar
ACCUMULATION_STEPS = 1

# -------------------- TRANSFORMACIONES --------------------
transform_train = Compose([
    EnsureType(),
    RandFlip(prob=0.5, spatial_axis=0),
    RandRotate90(prob=0.5, max_k=3),
    RandAffine(prob=0.3, translate_range=10, scale_range=0.1)
])

transform_val_test = Compose([EnsureType()])


# -------------------- DATA --------------------
# Leer CSV
df = pd.read_csv("./../../datos_clinicos/datos_clinicos.csv", na_values="NaN")
labels_dict_numeric = {
    k: 1 if str(v).strip().lower() in ['sí', 'si', 'yes', '1', 'true'] else 0 
    for k, v in dict(zip(df['Id_paciente'], df['Complicación'])).items()
}

# Paths
data_dir = "../../volumenes/preprocesados/preprocesamientos_chiquitos/resize_tiny/npy/images"
mask_dir = "../../volumenes/preprocesados/preprocesamientos_chiquitos/resize_tiny/npy/masks"

# Comprobación de archivos
num_images = len([f for f in os.listdir(data_dir) if f.endswith(".npy")])
num_masks = len([f for f in os.listdir(mask_dir) if f.endswith(".npy")])
print(f" Número de volúmenes en IMAGES: {num_images}")
print(f" Número de volúmenes en MASKS: {num_masks}")

# -------------------- DATASET --------------------
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

# -------------------- SPLIT --------------------
dataset_trainval = LungCTDataset(data_dir, mask_dir, labels_dict_numeric, transform=None)

# Split
labels = [dataset_trainval.labels[pid] for pid in dataset_trainval.patients]
train_val_pids, test_pids = train_test_split(
    dataset_trainval.patients,
    test_size=0.1,
    stratify=labels,
    random_state=42
)
pid_to_index = {pid: idx for idx, pid in enumerate(dataset_trainval.patients)}
train_val_idx = [pid_to_index[pid] for pid in train_val_pids]
test_idx = [pid_to_index[pid] for pid in test_pids]

train_val_dataset = Subset(dataset_trainval, train_val_idx)
test_dataset = Subset(LungCTDataset(data_dir, mask_dir, labels_dict_numeric, transform=transform_val_test), test_idx)

print(f" Total pacientes: {len(dataset_trainval)}")
print(f" Train+Val: {len(train_val_dataset)}")
print(f" Test hold-out: {len(test_dataset)}")


# -------------------- MODELO --------------------
class ResNetWithDropout(nn.Module):
    def __init__(self, dropout_prob=0.3):
        super().__init__()
        def get_inplanes():
            return [64, 128, 256, 512]

        self.resnet = ResNet(
            block='basic',
            layers=(2, 2, 2, 2),
            block_inplanes=get_inplanes(),
            spatial_dims=3,
            n_input_channels=1,
            num_classes=2,
            conv1_t_size=7,
            conv1_t_stride=1,
            no_max_pool=False,
            shortcut_type='B',
            widen_factor=1.0
        )
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        out = self.resnet(x)          # MONAI ResNet hace todo
        out = self.dropout(out)       # Aplicar dropout a logits
        return out



def build_model():
    return ResNetWithDropout(dropout_prob=0.5)


# -------------------- TRAINING LOOP --------------------
def train_model(
    model, train_loader, val_loader, test_loader, 
    criterion, optimizer, device, epochs, save_path, accumulation_steps=1
):
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_gmean = 0.0
    lrs = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        print(f"\n Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad()

        for i, (inputs, labels) in enumerate(tqdm(train_loader)):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)

        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds)
        train_tpr, train_tnr, train_gmean = calcular_metricas(all_labels, all_preds)

        # VALIDACIÓN
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
        val_tpr, val_tnr, val_gmean = calcular_metricas(val_labels, val_preds)

        scheduler.step()

        # ---------- PRINT en DOS líneas ----------
        print(f" Entrenamiento — Loss: {running_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f} | TPR: {train_tpr:.4f} | TNR: {train_tnr:.4f} | G-Mean: {train_gmean:.4f} | LR: {current_lr:.8f}")
        print(f" Validación   — Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | TPR: {val_tpr:.4f} | TNR: {val_tnr:.4f} | G-Mean: {val_gmean:.4f}")

        if save_path and val_gmean > best_val_gmean:
            best_val_gmean = val_gmean
            torch.save(model.state_dict(), save_path)
            print(f" Guardado modelo con mejor G-Mean ({val_gmean:.4f}) en '{save_path}'")

    # EVALUACIÓN FINAL
    if save_path:
        model.load_state_dict(torch.load(save_path))
        print(f"\n Evaluación Final en VALIDATION (mejor modelo):")
        evaluar_modelo(model, val_loader, device, "VALIDATION")
        print(f"\n Evaluación Final en TEST (mejor modelo):")
        evaluar_modelo(model, test_loader, device, "TEST")

    # Learning rate plot
    plt.figure(figsize=(6, 4))
    plt.plot(lrs, marker='o')
    plt.title("Evolución del Learning Rate")
    plt.xlabel("Época")
    plt.ylabel("Learning Rate")
    plt.grid()
    plt.show()


# -------------------- MÉTRICAS --------------------
def calcular_metricas(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return 0, 0, 0
    TN, FP, FN, TP = cm.ravel()
    tpr = TP / (TP + FN) if (TP + FN) else 0
    tnr = TN / (TN + FP) if (TN + FP) else 0
    gmean = np.sqrt(tpr * tnr)
    return tpr, tnr, gmean

def evaluar_modelo(model, data_loader, device, dataset_name=""):
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
    tpr, tnr, gmean = calcular_metricas(all_labels, all_preds)

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Complicación", "Complicación"])
    disp.plot(cmap='Blues')
    plt.title(f"Matriz de Confusión — {dataset_name}")
    plt.show()

    print(f"RESULTADOS en {dataset_name}:")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   F1 Score: {f1:.4f}")
    print(f"   TPR: {tpr:.4f}")
    print(f"   TNR: {tnr:.4f}")
    print(f"   G-Mean: {gmean:.4f}")

    return gmean

# -------------------- CROSS-VALIDATION --------------------
def cross_validate(
    model_class, train_val_dataset, test_dataset, 
    k=5, device='cuda', epochs=EPOCHS, save_path_prefix="modelo_fold_resnet"
):
    dataset_trainval_original = train_val_dataset.dataset
    labels = [dataset_trainval_original.labels[dataset_trainval_original.patients[i]] for i in train_val_dataset.indices]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_dataset.indices, labels)):
        print(f"\n==========================\n Fold {fold+1}/{k}\n==========================")

        train_abs_idx = [train_val_dataset.indices[i] for i in train_idx]
        val_abs_idx = [train_val_dataset.indices[i] for i in val_idx]

        train_subset = Subset(LungCTDataset(data_dir, mask_dir, labels_dict_numeric, transform=transform_train), train_abs_idx)
        val_subset = Subset(LungCTDataset(data_dir, mask_dir, labels_dict_numeric, transform=transform_val_test), val_abs_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = model_class().to(device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        criterion = nn.CrossEntropyLoss()

        save_path = f"{save_path_prefix}_fold{fold+1}.pth"

        train_model(
            model, train_loader, val_loader, test_loader,
            criterion, optimizer, device, epochs, save_path, accumulation_steps=ACCUMULATION_STEPS
        )

# -------------------- EJECUCIÓN --------------------
cross_validate(build_model, train_val_dataset, test_dataset, k=5, device=device, epochs=EPOCHS)

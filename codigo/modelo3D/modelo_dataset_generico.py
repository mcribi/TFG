#Importación de librerías: DICOM, numpy, matplotlib y widgets interactivos
from pydicom import dcmread
import numpy as np
from pydicom.fileset import FileSet
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
from collections import defaultdict
import pandas as pd
import seaborn as sns
import os

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.functional import softmax
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

import pydicom
from pydicom.filereader import dcmread
from pydicom.errors import InvalidDicomError

from torchvision.transforms.functional import resize
from skimage.transform import resize  # Para redimensionar las imágenes
from PIL import Image

import random
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold

import re #para expresiones regulares

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import pydicom

from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose

from monai.networks.nets import DenseNet121
import torch
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, EnsureChannelFirstD, ScaleIntensityD, ToTensor

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

device = "cuda" if torch.cuda.is_available() else "cpu"

#directorio dataset generico
ruta_dataset_generico="./dataset_generico_preprocesado_segementado/resize_small/npy/images"
ruta_mascaras_generico="./dataset_generico_preprocesado_segementado/resize_small/npy/masks"

#El tipo de cancer esta en la carpeta contenida
def get_cancer_type(patient_id):
    """Clasifica automáticamente según la letra en el ID del paciente."""
    cancer_code = patient_id.split('-')[1][0].upper()  # Ej: Extrae 'A' de 'Lung_Dx-A0001'
    cancer_types = {
        'A': 0,  # Adenocarcinoma
        'B': 1,  # Small Cell
        'E': 2,  # Large Cell
        'G': 3   # Squamous
    }
    return cancer_types.get(cancer_code, -1)  # -1 si no coincide

class LungGenericDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.patients = sorted([f.replace(".npy", "") for f in os.listdir(image_dir) if f.endswith(".npy")])
        self.labels = {pid: get_cancer_type(pid) for pid in self.patients if get_cancer_type(pid) != -1}
        self.patients = [pid for pid in self.patients if pid in self.labels]  # filtra válidos

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        pid = self.patients[idx]
        img = np.load(os.path.join(self.image_dir, f"{pid}.npy"))
        mask_path = os.path.join(self.mask_dir, f"{pid}.npy")
        if os.path.exists(mask_path):
            mask = np.load(mask_path).astype(bool)
            if img.shape != mask.shape:
                raise ValueError(f"Shape mismatch: {pid} {img.shape} vs {mask.shape}")
            img = img * mask
        img = np.expand_dims(img, axis=0)  # (1, D, H, W)
        label = self.labels[pid]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

from monai.transforms import Compose, EnsureType
transform = Compose([EnsureType()])

dataset = LungGenericDataset(ruta_dataset_generico, ruta_mascaras_generico, transform=transform)

from sklearn.model_selection import StratifiedShuffleSplit
labels = [dataset.labels[pid] for pid in dataset.patients]

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(dataset.patients, labels))

train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

def calcular_metricas_multiclase(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    return acc, f1_macro, f1_weighted, cm

def evaluar_modelo_final(model, val_loader, device, class_labels=None):
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc, f1_macro, f1_weighted, cm = calcular_metricas_multiclase(all_labels, all_preds)
    print("\n📌 Evaluación final:")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   F1 Macro: {f1_macro:.4f}")
    print(f"   F1 Weighted: {f1_weighted:.4f}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap='Blues')
    plt.title("📊 Matriz de Confusión")
    plt.show()

def train_model_bs_virtual(model, train_loader, val_loader, criterion, optimizer, device,
                           epochs=10, save_path=None, accumulation_steps=4, class_labels=None):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    best_val_f1 = 0
    lrs = []

    for epoch in range(epochs):
        print(f"\n📘 Epoch {epoch+1}/{epochs}")
        model.train()
        optimizer.zero_grad()
        running_loss, all_preds, all_labels = 0.0, [], []

        for i, (inputs, labels) in enumerate(tqdm(train_loader)):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        acc, f1_macro, f1_weighted, _ = calcular_metricas_multiclase(all_labels, all_preds)
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)

        print(f"🔹 Entrenamiento — Loss: {running_loss:.4f} | Acc: {acc:.4f} | F1_macro: {f1_macro:.4f} | F1_weighted: {f1_weighted:.4f} | LR: {current_lr}")

        # Validación
        val_loss, val_preds, val_labels = 0.0, [], []
        model.eval()
        with torch.no_grad():
            for val_inputs, val_labels_batch in val_loader:
                val_inputs, val_labels_batch = val_inputs.to(device), val_labels_batch.to(device)
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels_batch).item()
                val_preds.extend(torch.argmax(val_outputs, dim=1).cpu().numpy())
                val_labels.extend(val_labels_batch.cpu().numpy())

        val_acc, val_f1_macro, val_f1_weighted, _ = calcular_metricas_multiclase(val_labels, val_preds)
        print(f"🔸 Validación   — Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1_macro: {val_f1_macro:.4f} | F1_weighted: {val_f1_weighted:.4f}")
        scheduler.step(val_f1_macro)

        if save_path and val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            torch.save(model.state_dict(), save_path)
            print(f"✅ Guardado modelo (F1_macro {val_f1_macro:.4f}) en '{save_path}'")

    if save_path:
        model.load_state_dict(torch.load(save_path))
        print(f"\n📥 Cargado mejor modelo desde '{save_path}'")
        evaluar_modelo_final(model, val_loader, device, class_labels=class_labels)

    # Visualización del LR
    plt.plot(lrs, marker='o')
    plt.title("Evolución del Learning Rate")
    plt.xlabel("Época")
    plt.ylabel("LR")
    plt.grid()
    plt.show()


model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=4, dropout_prob=0.5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_labels = [dataset.labels[dataset.patients[i]] for i in train_idx]
unique_classes = np.unique(train_labels)
class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)


class_labels = ["Adenocarcinoma", "Small Cell", "Large Cell", "Squamous"]

train_model_bs_virtual(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=20,
    save_path="modelo_generico_segmentado_multiclase.pth",
    accumulation_steps=8,
    class_labels=class_labels
)

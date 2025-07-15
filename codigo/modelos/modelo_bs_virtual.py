#Importación de librerías: DICOM, numpy, matplotlib y widgets interactivos
import os
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import scipy.ndimage

from collections import defaultdict, Counter
from tqdm import tqdm
from PIL import Image

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, Subset, random_split

# MONAI
from monai.networks.nets import DenseNet121, ResNet
from monai.data import DataLoader as MonaiDataLoader, Dataset as MonaiDataset
from monai.transforms import Compose, EnsureType, EnsureChannelFirstD, ScaleIntensityD, ToTensor

# PyTorch Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Pydicom
import pydicom
from pydicom import dcmread
from pydicom.errors import InvalidDicomError
from pydicom.fileset import FileSet

# Skimage
from skimage.transform import resize

# Torchvision
from torchvision.transforms.functional import resize as tv_resize

# Scikit-learn
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    StratifiedShuffleSplit,
    KFold,
    StratifiedGroupKFold
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)



device = "cuda" if torch.cuda.is_available() else "cpu"

train = pd.read_csv("./../../datos_clinicos/datos_clinicos.csv", na_values="NaN", sep = ",") 

# creamos un diccionario que mapee Id_paciente a Complicación
labels_dict = dict(zip(train['Id_paciente'], train['Complicación']))

unique_labels = set(labels_dict.values())

# convertimos a numérico
labels_dict_numeric = {k: 1 if v.lower() in ['sí', 'si', 'yes', '1', 'true'] else 0 
                      for k, v in labels_dict.items()}

unique_labels_numeric = set(labels_dict_numeric.values())


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

        # cargamos volumen
        volume_path = os.path.join(self.data_dir, f"{patient_id}.npy")
        volume = np.load(volume_path)

        # cargamos máscara alineada
        mask_path = os.path.join(self.mask_dir, f"{patient_id}.npy")
        if os.path.exists(mask_path):
            mask = np.load(mask_path).astype(bool)
            if mask.shape != volume.shape:
                raise ValueError(f"Dimensiones no coinciden para {patient_id}: {volume.shape} vs {mask.shape}")
            volume = volume * mask
        else:
            print(f" No se encontró máscara para {patient_id}, usando volumen completo.")

        # expandimos dimensiones para pytorch (C, D, H, W)
        volume = np.expand_dims(volume, axis=0)  
        label = self.labels[patient_id]

        if self.transform:
            volume = self.transform(volume)

        return volume, torch.tensor(label, dtype=torch.long)


transform = Compose([EnsureType()])


data_dir="../../volumenes/preprocesados/preprocesamientos_chiquitos/resize_mini_hu_m300_1400_separadas/npy/images"
# mask_dir="../../volumenes/preprocesados/preprocesamientos2/resize_small_separadas/npy/masks"
mask_dir="../../volumenes/preprocesados/preprocesamientos_chiquitos/resize_mini_hu_m300_1400_separadas/npy/masks"


# contamos archivos .npy en imágenes
image_files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
num_images = len(image_files)

# contamos archivos .npy en máscaras
mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".npy")]
num_masks = len(mask_files)

print(f"Número de volúmenes en IMAGES: {num_images}")
print(f"Número de volúmenes en MASKS: {num_masks}")

dataset = LungCTDataset(data_dir, mask_dir, labels_dict_numeric, transform=transform)

#holdout estratificado


# recuperamos las etiquetas
# labels = [dataset.labels[pid] for pid in dataset.patients]

# # definimos el split estratificado
# sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=9)

# # obtenemos los índices para train y val
# train_idx, val_idx = next(sss.split(dataset.patients, labels))

# # creamos los subsets
# train_dataset = Subset(dataset, train_idx)
# val_dataset = Subset(dataset, val_idx)

# train_labels = [dataset.labels[dataset.patients[i]] for i in train_idx]
# val_labels = [dataset.labels[dataset.patients[i]] for i in val_idx]



# recuperamos etiquetas
labels = [dataset.labels[pid] for pid in dataset.patients]

# Split inicial: test fijo
train_val_pids, test_pids, train_val_labels, test_labels = train_test_split(
    dataset.patients,
    labels,
    test_size=0.1,
    stratify=labels,
    random_state=42
)

# mapeamos a indices del dataset original
pid_to_index = {pid: idx for idx, pid in enumerate(dataset.patients)}
train_val_idx = [pid_to_index[pid] for pid in train_val_pids]
test_idx = [pid_to_index[pid] for pid in test_pids]

# creamos los subsets
train_val_dataset = Subset(dataset, train_val_idx)
test_dataset = Subset(dataset, test_idx)

print(f"Total pacientes: {len(dataset)}")
print(f"Train+Val (para CV): {len(train_val_dataset)}")
print(f"Test hold-out: {len(test_dataset)}")


BATCH_SIZE=64

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)



#modelo MONAI
model = DenseNet121(
    spatial_dims=3,          # 3D
    in_channels=1,           # Canales de entrada
    out_channels=2,          # Salida binaria
    #pretrained=False,        # sin pesos preentrenados
    dropout_prob=0.5         # Regularización
).to(device)

# comprobacion del modelo con un ejemplo
# x = torch.randn(1, 1, 128, 256, 256).to(device)  # (Batch, Channels, Depth, Height, Width)
# output = model(x)
# print("Output shape:", output.shape)  # Debe ser (1, 2)

# optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# entrenamiento sin ponderacion de pesos
criterion = nn.CrossEntropyLoss()

def train_model_bs_virtual(
    model, 
    train_loader, 
    val_loader, 
    test_loader,
    criterion, 
    optimizer, 
    device, 
    epochs=10, 
    save_path=None, 
    accumulation_steps=8
):
    best_val_gmean = 0.0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
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
                val_loss += criterion(val_outputs, val_labels_batch).item()
                val_preds_batch = torch.argmax(val_outputs, dim=1).cpu().numpy()
                val_preds.extend(val_preds_batch)
                val_labels.extend(val_labels_batch.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        val_tpr, val_tnr, val_gmean = calcular_metricas_binarias(val_labels, val_preds)
        print(f" Validación   — Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | TPR: {val_tpr:.4f} | TNR: {val_tnr:.4f} | G-Mean: {val_gmean:.4f}")

        scheduler.step(val_loss)

        if save_path and val_gmean > best_val_gmean:
            best_val_gmean = val_gmean
            torch.save(model.state_dict(), save_path)
            print(f" Guardado modelo (G-Mean {val_gmean:.4f}) en '{save_path}'")

    # evaluamos modelo final (validación + test)
    if save_path:
        model.load_state_dict(torch.load(save_path))
        model.eval()
        print(f"\n Cargado mejor modelo desde '{save_path}'")
        print("\n EVALUACIÓN FINAL en VALIDATION (mejor modelo):")
        evaluar_modelo_final(model, val_loader, device, dataset_name="VALIDATION")

        print("\n EVALUACIÓN FINAL en TEST (mejor modelo):")
        evaluar_modelo_final(model, test_loader, device, dataset_name="TEST")


    # Learning rate plot
    plt.figure(figsize=(6, 4))
    plt.plot(lrs, marker='o')
    plt.title("Evolución del Learning Rate")
    plt.xlabel("Época")
    plt.ylabel("Learning Rate")
    plt.grid()
    plt.show()



def calcular_metricas_binarias(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return 0, 0, 0  #no binario
    TN, FP, FN, TP = cm.ravel()
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
    tnr = TN / (TN + FP) if (TN + FP) > 0 else 0
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



def get_last_conv_layer(model):
    # buscamos la ultima capa convolucional del encoder DenseNet de MONAI
    for layer in reversed(model.features):
        if isinstance(layer, nn.Conv3d):
            return layer
        if hasattr(layer, 'layers'):  # DenseBlock o Transition
            for sub in reversed(layer.layers):
                if isinstance(sub, nn.Conv3d):
                    return sub
    raise ValueError("No se encontró una capa Conv3d válida en el modelo")


def apply_gradcam(model, volume, label, device, example_id=""):
    model.train()  
    target_layers = [get_last_conv_layer(model)]
    cam = GradCAM(model=model, target_layers=target_layers)

    input_tensor = volume.to(device)
    targets = [ClassifierOutputTarget(label)]
    
    with torch.enable_grad():  #  se necesitan gradientes para CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # [D, H, W]

    for idx in range(0, grayscale_cam.shape[0], max(1, grayscale_cam.shape[0] // 6)):
        plt.figure(figsize=(6, 6))
        img = input_tensor.cpu().numpy()[0, 0, idx]
        norm_img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        heatmap = grayscale_cam[idx]
        rgb_img = np.repeat(norm_img[..., np.newaxis], 3, axis=-1)  # Convierte (H,W) a (H,W,3)
        result = show_cam_on_image(rgb_img.astype(np.float32), heatmap, use_rgb=True)


        output_dir = f"gradcam_outputs_100_pacientes_otro_/class_{label}/"
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{example_id}_slice_{idx}.png")
        plt.imsave(path, result)
        plt.close()



def visualizar_gradcams(model, dataset, val_idx, device):
    print("Mostrando ejemplos de Grad-CAM...")

    
    correctos = defaultdict(list)
    incorrectos = defaultdict(list)

    loader = DataLoader(Subset(dataset, val_idx), batch_size=1, shuffle=False)
    model.eval()

    for i, (vol, label) in enumerate(loader):
        vol = vol.to(device)
        pred = torch.argmax(model(vol), dim=1).item()
        real = label.item()

        if pred == real:
            correctos[real].append(vol)
        else:
            incorrectos[real].append(vol)

    for clase in [0, 1]:
        print(f"\n Correctos clase {clase}")
        for v in correctos[clase][:2]:
            apply_gradcam(model, v, clase, device, example_id=f"{clase}_correcto_{i}")

        print(f"\n Fallos clase {clase}")
        for v in incorrectos[clase][:2]:
            apply_gradcam(model, v, clase, device, example_id=f"{clase}_fallo_{i}")


def cross_validate(
    model_class, 
    train_val_dataset, 
    test_dataset, 
    k=5, 
    device='cuda', 
    epochs=10, 
    save_path_prefix="modelo_fold"
):
    

    # saca etiquetas para stratificar
    labels = [dataset.labels[dataset.patients[i]] for i in train_val_dataset.indices]

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    total_cm = np.array([[0, 0], [0, 0]])
    accs, f1s, tprs, tnrs, gmeans = [], [], [], [], []
    model_paths = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_dataset.indices, labels)):
        print(f"\n Fold {fold+1}/{k}")

        # indices absolutos dentro del dataset original
        train_abs_idx = [train_val_dataset.indices[i] for i in train_idx]
        val_abs_idx = [train_val_dataset.indices[i] for i in val_idx]

        train_subset = Subset(dataset, train_abs_idx)
        val_subset = Subset(dataset, val_abs_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        model = model_class().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        save_path = f"{save_path_prefix}_fold{fold+1}.pth"
        train_model_bs_virtual(
            model, 
            train_loader, 
            val_loader, 
            test_loader,
            criterion, 
            optimizer, 
            device, 
            epochs=epochs, 
            save_path=save_path, 
            accumulation_steps=1
        )

        model_paths.append(save_path)

        # evaluamos en val (fold)
        model.load_state_dict(torch.load(save_path))
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels_batch in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels_batch.cpu().numpy())

        # matriz de confusion del fold
        cm = confusion_matrix(all_labels, all_preds)
        total_cm += cm

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        tpr, tnr, gmean = calcular_metricas_binarias(all_labels, all_preds)

        accs.append(acc)
        f1s.append(f1)
        tprs.append(tpr)
        tnrs.append(tnr)
        gmeans.append(gmean)

        print(f"\n Evaluación del Fold {fold+1}/{k}")
        print(f"   Accuracy: {acc:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        print(f"   TPR: {tpr:.4f}")
        print(f"   TNR: {tnr:.4f}")
        print(f"   G-Mean: {gmean:.4f}")

        visualizar_gradcams(model, train_val_dataset, val_idx, device)

    # resumen de val
    print("\n Matriz de Confusión Acumulada (Todos los folds):")
    disp = ConfusionMatrixDisplay(confusion_matrix=total_cm,
                                  display_labels=["No Complicación", "Complicación"])
    disp.plot(cmap='Blues')
    plt.title("Matriz de Confusión Acumulada")
    plt.show()

    print("\n Métricas medias (promedio de los folds):")
    print(f"   Accuracy medio: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"   F1 Score medio: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"   TPR medio: {np.mean(tprs):.4f} ± {np.std(tprs):.4f}")
    print(f"   TNR medio: {np.mean(tnrs):.4f} ± {np.std(tnrs):.4f}")
    print(f"   G-Mean medio: {np.mean(gmeans):.4f} ± {np.std(gmeans):.4f}")

    # evaluamos todos los modelos en test fijo
    print("\n\n Evaluación FINAL en TEST hold-out:")
    for fold, path in enumerate(model_paths):
        print(f"\n Fold {fold+1} — evaluando en TEST con modelo '{path}'")
        model = model_class().to(device)
        model.load_state_dict(torch.load(path))
        model.eval()
        evaluar_modelo_final(model, DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False), device, dataset_name="TEST")



def build_model():
    return DenseNet121(spatial_dims=3, in_channels=1, out_channels=2)

#cross_validate(build_model, dataset, k=5, device=device, epochs=20)     
#train_model_bs_virtual(model, train_loader, val_loader, criterion, optimizer, device, epochs=10, save_path="mejor_modelo_resnet_gmean_mas_epocas_2.pth", accumulation_steps=76)

cross_validate(build_model, train_val_dataset, test_dataset, k=5, device=device, epochs=30)


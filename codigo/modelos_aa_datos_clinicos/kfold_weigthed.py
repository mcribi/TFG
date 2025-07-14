#librerías: DICOM, numpy, matplotlib y widgets interactivos
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
from skimage.transform import resize  #para redimensionar las imágenes
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
from pydicom import dcmread
import scipy.ndimage
import cv2

from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose

from monai.networks.nets import DenseNet121
import torch
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, EnsureChannelFirstD, ScaleIntensityD, ToTensor
from monai.transforms import Compose, EnsureChannelFirst, ScaleIntensity, ToTensor, EnsureType

from torch.utils.data import WeightedRandomSampler

#lectura csv
train = pd.read_csv("datos_filtrados.csv", na_values="NaN", sep = ",") # Definimos na_values para identificar bien los valores perdidos

#borramos el paciente que falta
train = train.drop([63])

import warnings
warnings.filterwarnings("ignore")

# creamos un diccionario que mapee Id_paciente a Complicación
labels_dict = dict(zip(train['Id_paciente'], train['Complicación']))

unique_labels = set(labels_dict.values())
print(f"Labels únicas: {unique_labels}")

# convertimos a numérico
labels_dict_numeric = {k: 1 if v.lower() in ['sí', 'si', 'yes', '1', 'true'] else 0 
                      for k, v in labels_dict.items()}

unique_labels_numeric = set(labels_dict_numeric.values())
print(f"Labels únicas: {unique_labels_numeric}")

#datos
root_path = "./datos_anonimizados/"  
patient_dirs = [os.path.join(root_path, d) for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]

#activamos gpu
device = "cuda" if torch.cuda.is_available() else "cpu"

def apply_3d_window(volume, window_level=-600, window_width=1500):
    """
    Aplica ventana HU a un volumen 3D completo.
    
    Parámetros:
        volume: numpy array 3D (depth, height, width) en unidades Hounsfield (HU)
        window_level: Centro de la ventana (típico pulmonar: -600 HU)
        window_width: Ancho de ventana (típico pulmonar: 1500 HU)
    
    Returns:
        Volumen normalizado a [0,1] según la ventana seleccionada
    """
    min_val = window_level - window_width/2
    max_val = window_level + window_width/2
    
    # Aplicar ventana y normalizar
    windowed = np.clip(volume, min_val, max_val)
    normalized = (windowed - min_val) / (max_val - min_val)
    
    return normalized.astype(np.float32)

#dataset definicion
class LungCTDataset(Dataset):
    def __init__(self, data_dir, labels_dict, transform=None):
        self.data_dir = data_dir
        self.patients = list(labels_dict.keys())
        self.labels = labels_dict
        self.transform = transform
    
    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, idx):
        patient_id = self.patients[idx]
        volume_path = os.path.join(self.data_dir, f"{patient_id}.npy")
        volume = np.load(volume_path)  # volumen preprocesado
        # aplcicamos ventana pulmonar
        volume = apply_3d_window(volume, window_level=-600, window_width=1500)
        volume = np.expand_dims(volume, axis=0)
        label = self.labels[patient_id]
        
        if self.transform:
            volume = self.transform(volume)
        
        return volume, torch.tensor(label, dtype=torch.long)
    

transform = Compose([
    #EnsureChannelFirst(), #para que sea del tipo (1, 256, 512, 512)
    EnsureType()
])

NPY_DIR="preprocessed_data_3d"
BATCH_SIZE = 2
TARGET_SIZE = (512, 512)

dataset = LungCTDataset(NPY_DIR, labels_dict_numeric, transform=transform)
labels = np.array(list(labels_dict_numeric.values()))

# Loop de entrenamiento
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        all_preds, all_labels = [], []
        for inputs, labels in tqdm(train_loader):
            #labels = labels.float().unsqueeze(1)
            inputs, labels = inputs.cuda(), labels.cuda()
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
        print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}, Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
        
        # Validación
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_labels_batch in tqdm(val_loader):
                val_inputs, val_labels_batch = val_inputs.cuda(), val_labels_batch.cuda()
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels_batch).item()
                
                val_preds_batch = torch.argmax(val_outputs, dim=1).cpu().numpy()
                val_preds.extend(val_preds_batch)
                val_labels.extend(val_labels_batch.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}")
        model.train()

#vamos a hacer 5-fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = {}

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"\n{'='*40}\nFold {fold+1}/5\n{'='*40}")
    
    # oversampling de la clase minoritaria solo en train
    train_labels = labels[train_idx]
    minor_class = 1 if sum(train_labels) < len(train_labels)/2 else 0
    minor_indices = train_idx[train_labels == minor_class]
    oversampled_indices = np.concatenate([train_idx, np.random.choice(minor_indices, size=len(minor_indices), replace=True)])
    
    #  subsets
    train_subset = Subset(dataset, oversampled_indices)
    val_subset = Subset(dataset, val_idx)
    
    #  WeightedRandomSampler para el fold actual, solo en train
    def get_sampler(subset):
        subset_labels = labels[subset.indices]
        class_counts = np.bincount(subset_labels)
        weights = 1 / (torch.tensor(class_counts ** 1.7, dtype=torch.float32) + 1e-6)  
        samples_weights = weights[subset_labels]
        return WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)
    
    train_sampler = get_sampler(train_subset)
    
    # dataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # reiniciamos el modelo para cada fold
    model = DenseNet121(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        dropout_prob=0.4
    ).to(device)
    
    class FocalLoss(nn.Module):
        def __init__(self, alpha=0.25, gamma=2):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, inputs, targets):
            BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-BCE_loss)
            loss = self.alpha * (1-pt)**self.gamma * BCE_loss
            return loss.mean()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    criterion = FocalLoss(alpha=0.9, gamma=1.8)
    
    # train
    best_f1 = 0
    for epoch in range(15): 
        model.train()
        epoch_loss = 0
        all_preds, all_labels = [], []
        
        for inputs, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, labels_batch = inputs.to(device), labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  
            optimizer.step()
            
            epoch_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_batch.cpu().numpy())
        
        # val
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0
        with torch.no_grad():
            for inputs, labels_batch in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels_batch.to(device)).item()
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_labels.extend(labels_batch.cpu().numpy())
        
        # metricas
        train_f1 = f1_score(all_labels, all_preds)
        val_f1 = f1_score(val_labels, val_preds)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1} | Train Loss: {epoch_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # guardamos mejor modelo del fold
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), f"best_fold{fold+1}.pth")
            print(f"Mejor modelo del fold {fold+1} guardado (F1: {best_f1:.4f})")
    
    fold_results[f'fold_{fold+1}'] = best_f1

# resultados finales
print("\nResultados de los 5 folds:")
for fold, f1 in fold_results.items():
    print(f"{fold}: F1 = {f1:.4f}")

print(f"\nF1 promedio: {np.mean(list(fold_results.values())):.4f}")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# modelo en evaluación
model.eval()

all_preds = []
all_labels = []

# iteramos sobre el DataLoader de validación
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        
        # clasificación binaria con 2 neuronas de salida
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

# matriz de confusion
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=["No Complicación", "Complicación"])
disp.plot(cmap='Blues')
plt.title("Matriz de Confusión Final (Validación)")
plt.show()

# otras metricas
print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
print(f"F1-Score: {f1_score(all_labels, all_preds):.4f}")

# guardamos el modelo entrenado
torch.save(model.state_dict(), 'modelo_entrenado_WeightedRandomSampler_1.7.pth')
# modelo completo (arquitectura + pesos)
torch.save(model, 'modelo_entrenado_completo_WeightedRandomSampler_1.7.pth')
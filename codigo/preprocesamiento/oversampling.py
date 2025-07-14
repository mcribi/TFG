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

#lectura csv
train = pd.read_csv("datos_filtrados.csv", na_values="NaN", sep = ",") # Definimos na_values para identificar bien los valores perdidos

#borramos el paciente que falta
train = train.drop([63])

import warnings
warnings.filterwarnings("ignore")

# Crear un diccionario que mapee Id_paciente a Complicación
labels_dict = dict(zip(train['Id_paciente'], train['Complicación']))

unique_labels = set(labels_dict.values())
print(f"Labels únicas: {unique_labels}")

# Convertir a numérico
labels_dict_numeric = {k: 1 if v.lower() in ['sí', 'si', 'yes', '1', 'true'] else 0 
                      for k, v in labels_dict.items()}

unique_labels_numeric = set(labels_dict_numeric.values())
print(f"Labels únicas: {unique_labels_numeric}")

#DATOS ANONIMIZADOS
root_path = "./datos_anonimizados/"  # Directorio raíz que contiene los pacientes
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
        volume = np.load(volume_path)  # Cargar volumen preprocesado (256, 512, 512)
        # Aplicar ventana pulmonar
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

# División en 90% entrenamiento y 10% validación
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

#obtenemos los indices de cada conjunto
train_indices = train_dataset.indices  # Índices del conjunto de entrenamiento
val_indices = val_dataset.indices  

from collections import Counter
# 2. Contar etiquetas en cada conjunto
def count_labels(dataset, indices):
    labels = []
    for idx in indices:
        _, label = dataset[idx]  # Asumiendo que tu dataset devuelve (volumen, label, id)
        labels.append(label.item() if torch.is_tensor(label) else label)
    return Counter(labels)

# Contar en train y val
train_counts = count_labels(dataset, train_indices)
val_counts = count_labels(dataset, val_indices)

# 3. Mostrar resultados
print("\nDistribución de clases:")
print(f"Entrenamiento - Total: {train_size} | Clase 0: {train_counts[0]} | Clase 1: {train_counts[1]}")
print(f"Validación - Total: {val_size} | Clase 0: {val_counts[0]} | Clase 1: {val_counts[1]}")

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

from torch.utils.data import WeightedRandomSampler
#calculamos el sampler para cada el train
def get_sampler(subset, power=1.0):
    # Obtener labels del subset
    subset_labels = [dataset.labels[dataset.patients[idx]] for idx in subset.indices]
    class_counts = np.bincount(subset_labels)
    weights = 1. / torch.tensor(class_counts**power, dtype=torch.float) #inverso al cuadrado para darle aun más peso a las clases minoritarias
    samples_weights = weights[subset_labels]
    return WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)


train_sampler = get_sampler(train_dataset, 1.7)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler, # Sampler de entrenamiento
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=4,
    pin_memory=True
)

#modelo MONAI
model = DenseNet121(
    spatial_dims=3,          # 3D
    in_channels=1,           # Canales de entrada
    out_channels=2,          # Salida binaria
    #pretrained=False,        # sin pesos preentrenados
    dropout_prob=0.4         # Regularización
).to(device)


#optimizer = optim.Adam(model.parameters(), lr=1e-4)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

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

# Configuración de entrenamiento (con focal loss)
criterion = FocalLoss(alpha=0.9, gamma=1.8) # alpha ~1/clase_minoritaria

# con power=1.5
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Poner el modelo en modo evaluación
model.eval()

# 2. Listas para almacenar predicciones y etiquetas reales
all_preds = []
all_labels = []

# 3. Iterar sobre el DataLoader de validación
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        
        # Para clasificación binaria con 2 neuronas de salida (out_channels=2)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

# 4. Calcular y mostrar la matriz
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=["No Complicación", "Complicación"])
disp.plot(cmap='Blues')
plt.title("Matriz de Confusión Final (Validación)")
plt.show()

# 5. Métricas adicionales
print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
print(f"F1-Score: {f1_score(all_labels, all_preds):.4f}")

# Guardar el modelo entrenado
torch.save(model.state_dict(), 'modelo_entrenado_WeightedRandomSampler_1.7.pth')
# Guardar el modelo completo (arquitectura + pesos)
torch.save(model, 'modelo_entrenado_completo_WeightedRandomSampler_1.7.pth')
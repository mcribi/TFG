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

from torchvision.models.video import r3d_18
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from monai.transforms import Compose, EnsureType

#config
device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20
WEIGHT_DECAY = 1e-5

# leemos los datos clinicos
df = pd.read_csv("./../../datos_clinicos/datos_clinicos.csv", na_values="NaN")
labels_dict_numeric = {
    k: 1 if str(v).strip().lower() in ['sí', 'si', 'yes', '1', 'true'] else 0 
    for k, v in dict(zip(df['Id_paciente'], df['Complicación'])).items()
}

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

        mask_path = os.path.join(self.mask_dir, f"{patient_id}.npy")
        if os.path.exists(mask_path):
            mask = np.load(mask_path).astype(bool)
            if mask.shape == volume.shape:
                volume = volume * mask
            else:
                raise ValueError(f"Dimensiones no coinciden para {patient_id}: {volume.shape} vs {mask.shape}")
        else:
            print(f"No se encontró máscara para {patient_id}, usando volumen completo.")

        volume = np.expand_dims(volume, axis=0)
        if self.transform:
            volume = self.transform(volume)
        label = self.labels[patient_id]
        return volume, torch.tensor(label, dtype=torch.long)

# transformaciones
transform = Compose([EnsureType()])

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

    print(f"\n Evaluación FINAL en {dataset_name}:")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   F1 Score: {f1:.4f}")
    print(f"   TPR: {tpr:.4f}")
    print(f"   TNR: {tnr:.4f}")
    print(f"   G-Mean: {gmean:.4f}")

#modelo
def build_model(
    pretrained_weights_path=None,
    fine_tune_mode="full",
    num_classes=2,
    in_channels=1
):
    print(f"Construyendo modelo r3d_18 (fine_tune_mode='{fine_tune_mode}')")

    #creamos modelo con 3 canales
    model = r3d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # si queremos 1 canal, sustituir la primera conv3d
    if in_channels == 1:
        print(" Ajustando la primera capa (stem) para in_channels=1")
        old_conv = model.stem[0]
        new_conv = nn.Conv3d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None)
        )
        model.stem[0] = new_conv

    # si hay pesos preentrenados, cargarlos
    if pretrained_weights_path:
        print(f" Cargando pesos preentrenados desde: {pretrained_weights_path}")
        state_dict = torch.load(pretrained_weights_path, map_location=device)

        # adaptamos la primera capa si checkpoint tiene diferente in_channels
        if "stem.0.weight" in state_dict:
            pretrained_stem_weight = state_dict["stem.0.weight"]
            if pretrained_stem_weight.shape[1] != in_channels:
                print(f" Ajustando stem.0.weight de {pretrained_stem_weight.shape} a in_channels={in_channels}")
                # Si checkpoint es (64, 1, 3, 7, 7) y queremos 3 canales
                if pretrained_stem_weight.shape[1] == 1 and in_channels == 3:
                    state_dict["stem.0.weight"] = pretrained_stem_weight.repeat(1, 3, 1, 1, 1) / 3
                # Si checkpoint es (64, 3, 3, 7, 7) y queremos 1 canal
                elif pretrained_stem_weight.shape[1] == 3 and in_channels == 1:
                    state_dict["stem.0.weight"] = pretrained_stem_weight.mean(dim=1, keepdim=True)
                else:
                    print(" No se puede adaptar stem automáticamente. Quitando del checkpoint.")
                    del state_dict["stem.0.weight"]

        # ignoramos la capa final fc
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f" Pesos cargados (missing: {missing}, unexpected: {unexpected})")

    # configuramos fine-tuning
    if fine_tune_mode == "head_only":
        print(" Congelando todos los parámetros excepto la última capa (fc)")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    elif fine_tune_mode == "full":
        print(" Todos los parámetros entrenables")
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Modo de fine-tune no reconocido: {fine_tune_mode}")

    return model


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

        print(f"\n📘 Epoch {epoch+1}/{epochs}")

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

        # validacion
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

        if save_path and (epoch == 0 or val_gmean > best_val_gmean):
            best_val_gmean = val_gmean
            torch.save(model.state_dict(), save_path)
            print(f" Guardado modelo en '{save_path}' con G-Mean: {val_gmean:.4f}")

    if save_path:
        model.load_state_dict(torch.load(save_path))
        print(f"\n Cargado mejor modelo desde '{save_path}'")
        print("\n Evaluación FINAL en VALIDATION (mejor modelo):")
        evaluar_modelo_final(model, val_loader, device, dataset_name="VALIDATION")
        print("\n Evaluación FINAL en TEST (mejor modelo):")
        evaluar_modelo_final(model, test_loader, device, dataset_name="TEST")

# cv
def cross_validate(
    model_class,
    train_val_dataset,
    test_dataset,
    batch_size,
    learning_rate,
    k=5,
    device='cuda',
    epochs=10,
    save_path_prefix="modelo_resnet3d_fold",
    pretrained_weights_path=None,
    fine_tune_mode="full"
):
    results = []
    labels = [dataset.labels[dataset.patients[i]] for i in train_val_dataset.indices]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_dataset.indices, labels)):
        print("\n===========================================")
        print(f" Fold {fold+1}/{k} | BS={batch_size} | LR={learning_rate} | Fine-Tune={fine_tune_mode}")
        print("===========================================")

        train_abs_idx = [train_val_dataset.indices[i] for i in train_idx]
        val_abs_idx = [train_val_dataset.indices[i] for i in val_idx]

        train_subset = Subset(dataset, train_abs_idx)
        val_subset = Subset(dataset, val_abs_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        model = model_class(
            pretrained_weights_path=pretrained_weights_path,
            fine_tune_mode=fine_tune_mode
        ).to(device)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()

        save_path = f"{save_path_prefix}_bs{batch_size}_lr{learning_rate}_fold{fold+1}.pth"

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

#grid
#preprocesados
preprocessing_dirs = [
    "resize_mini_hu_m300_1400_separadas",
    "resize_mini_hu_m600_1500_separadas",
    "resize_mini"
]

batch_sizes_to_try = [4, 8, 16]
learning_rates_to_try = [1e-5, 1e-4, 1e-3]
fine_tune_modes_to_try = ["full","head_only"]

#ruta a pesos preentrenados
PRETRAINED_PATH = "./pretrain/r3d18_medmnist3d_pretrained.pth"

all_results = []

for prep in preprocessing_dirs:
    print("\n##################################################")
    print(f"=== EMPEZANDO EXPERIMENTOS PARA PREPROCESADO: {prep} ===")
    print("##################################################")

    # rutas del preprocesado actual
    data_dir = f"../../volumenes/preprocesados/preprocesamientos_chiquitos/{prep}/npy/images"
    mask_dir = f"../../volumenes/preprocesados/preprocesamientos_chiquitos/{prep}/npy/masks"

    # creamos nuevo dataset para este preprocesado
    dataset = LungCTDataset(data_dir, mask_dir, labels_dict_numeric, transform=transform)

    # Split fijo para test
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

    # grid de BS, LR y fine-tune
    for bs in batch_sizes_to_try:
        for lr in learning_rates_to_try:
            for ft_mode in fine_tune_modes_to_try:
                print("\n##################################################")
                print(f"Preprocesado={prep} | BATCH_SIZE={bs}, LEARNING_RATE={lr}, FINE_TUNE={ft_mode}")
                print("##################################################")

                df_result = cross_validate(
                    build_model,
                    train_val_dataset,
                    test_dataset,
                    batch_size=bs,
                    learning_rate=lr,
                    k=5,
                    device=device,
                    epochs=EPOCHS,
                    save_path_prefix=f"resnet3d_{prep}_bs{bs}_lr{lr}_{ft_mode}",
                    pretrained_weights_path=PRETRAINED_PATH,
                    fine_tune_mode=ft_mode
                )

                # añadimos columnas con configuración
                df_result["preprocessed_dir"] = prep
                df_result["fine_tune_mode"] = ft_mode

                all_results.append(df_result)

# concatenamos y guardamos todos los resultados
final_results = pd.concat(all_results, ignore_index=True)
final_results.to_csv("resultados_pretrain_resnet3d.csv", index=False)
print("\n Todos los resultados guardados en 'resultados_pretrain_resnet3d.csv'")

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from monai.networks.nets import DenseNet121
from monai.transforms import Compose, EnsureType

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# -------------------- CONFIG --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f" Using device: {device}")

# -------------------- DATA LOADING --------------------
print(" Loading labels...")
df = pd.read_csv("./../../datos_clinicos/datos_clinicos.csv", na_values="NaN")
labels_dict_numeric = {
    k: 1 if str(v).strip().lower() in ['sí', 'si', 'yes', '1', 'true'] else 0
    for k, v in dict(zip(df['Id_paciente'], df['Complicación'])).items()
}

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

transform = Compose([EnsureType()])

# -------------------- MODELO --------------------
def get_densenet3d_model(n_classes=2):
    return DenseNet121(spatial_dims=3, in_channels=1, out_channels=n_classes, dropout_prob=0.3)

# -------------------- MÉTRICAS --------------------
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

# -------------------- CROSS-VALIDATION CON SPLIT INTERNO --------------------
def cross_validate_finetune_densenet(
    model,
    dataset,
    batch_size,
    learning_rate,
    weight_decay,
    device,
    epochs,
    output_log,
    model_file_name,
    checkpoint_path,
    k=5
):
    labels = [dataset.labels[pid] for pid in dataset.patients]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    with open(output_log, "a") as logf:
        logf.write(f"\n\n=========================")
        logf.write(f"\n Modelo: {model_file_name}")
        logf.write(f"\nBatchSize: {batch_size}, LR: {learning_rate}, Epochs: {epochs}\n")
        logf.write(f"=========================\n")

        fold_idx = 1
        for trainval_idx, test_idx in skf.split(dataset.patients, labels):
            logf.write(f"\n----- Fold {fold_idx} -----\n")
            print(f"\n Fold {fold_idx}/{k}")

            # Split externo (80-20)
            trainval_subset = Subset(dataset, trainval_idx)
            test_subset = Subset(dataset, test_idx)

            # Split interno (80-20 de TrainVal)
            internal_labels = [dataset.labels[dataset.patients[i]] for i in trainval_idx]
            train_idx, valid_idx = train_test_split(
                trainval_idx,
                test_size=0.2,
                stratify=internal_labels,
                random_state=42
            )

            train_subset = Subset(dataset, train_idx)
            valid_subset = Subset(dataset, valid_idx)

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

            # Recargar pesos preentrenados
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Filtra la capa final para evitar el error de size mismatch
            filtered_checkpoint = {k: v for k, v in checkpoint.items() if "class_layers.out" not in k}

            model.load_state_dict(filtered_checkpoint, strict=False)

            model = model.to(device)

            # Descongelar TODO
            for param in model.parameters():
                param.requires_grad = True

            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            criterion = nn.CrossEntropyLoss()

            best_gmean = 0.0
            best_model_path = f"best_model_fold_densenet_descongelado_dropout_{fold_idx}.pth"
            saved_first = False

            for epoch in range(epochs):
                model.train()
                train_preds, train_labels = [], []
                running_loss = 0.0

                print(f"\n Epoch {epoch+1}/{epochs}")

                # -------- Train --------
                for inputs, labels_batch in tqdm(train_loader, desc="Training"):
                    inputs, labels_batch = inputs.to(device), labels_batch.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels_batch)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    train_preds.extend(preds)
                    train_labels.extend(labels_batch.cpu().numpy())

                train_acc = accuracy_score(train_labels, train_preds)
                train_f1 = f1_score(train_labels, train_preds)
                train_tpr, train_tnr, train_gmean = calcular_metricas_binarias(train_labels, train_preds)
                print(f" Train — Loss: {running_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f} | G-Mean: {train_gmean:.4f}")

                # -------- Validation --------
                model.eval()
                valid_preds, valid_labels = [], []
                valid_loss = 0.0
                with torch.no_grad():
                    for val_inputs, val_labels_batch in valid_loader:
                        val_inputs, val_labels_batch = val_inputs.to(device), val_labels_batch.to(device)
                        val_outputs = model(val_inputs)
                        batch_loss = criterion(val_outputs, val_labels_batch)
                        valid_loss += batch_loss.item()

                        preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
                        valid_preds.extend(preds)
                        valid_labels.extend(val_labels_batch.cpu().numpy())

                val_acc = accuracy_score(valid_labels, valid_preds)
                val_f1 = f1_score(valid_labels, valid_preds)
                val_tpr, val_tnr, val_gmean = calcular_metricas_binarias(valid_labels, valid_preds)
                print(f" Valid — Loss: {valid_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | G-Mean: {val_gmean:.4f}")

                # Guardar primera época siempre
                if epoch == 0 and not saved_first:
                    torch.save(model.state_dict(), best_model_path)
                    saved_first = True
                    best_gmean = val_gmean
                    print(f" Guardado primer modelo (epoch 1) con G-Mean: {val_gmean:.4f}")

                # Guardar mejor en Valid
                elif val_gmean > best_gmean:
                    best_gmean = val_gmean
                    torch.save(model.state_dict(), best_model_path)
                    print(f" Mejor modelo GUARDADO con G-Mean en Valid: {best_gmean:.4f}")

                # -------- Test --------
                test_preds, test_labels = [], []
                with torch.no_grad():
                    for test_inputs, test_labels_batch in test_loader:
                        test_inputs = test_inputs.to(device)
                        outputs = model(test_inputs)
                        preds = torch.argmax(outputs, dim=1).cpu().numpy()
                        test_preds.extend(preds)
                        test_labels.extend(test_labels_batch.numpy())

                test_acc = accuracy_score(test_labels, test_preds)
                test_f1 = f1_score(test_labels, test_preds)
                test_tpr, test_tnr, test_gmean = calcular_metricas_binarias(test_labels, test_preds)
                print(f" Test — Acc: {test_acc:.4f} | F1: {test_f1:.4f} | G-Mean: {test_gmean:.4f}")

            # Evaluación final
            model.load_state_dict(torch.load(best_model_path))
            final_acc, final_f1, final_tpr, final_tnr, final_gmean = evaluar_modelo(model, test_loader, device)
            logf.write(f"\nRESULTADOS FINALES Fold {fold_idx}:\n")
            logf.write(f"Test Acc: {final_acc:.4f}, F1: {final_f1:.4f}, TPR: {final_tpr:.4f}, TNR: {final_tnr:.4f}, G-Mean: {final_gmean:.4f}\n")
            logf.write(f"=========================\n")
            fold_idx += 1

# -------------------- MAIN LOOP --------------------
def fine_tune_all_models_in_folder(
    models_folder,
    data_dir,
    mask_dir,
    output_log,
    batch_size=4,
    learning_rate=1e-4,
    weight_decay=1e-5,
    epochs=10,
    k=5
):
    dataset = LungCTDataset(data_dir, mask_dir, labels_dict_numeric, transform=transform)
    model_files = [f for f in os.listdir(models_folder) if f.endswith('.pth')]

    print(f" Encontrados {len(model_files)} modelos DenseNet preentrenados en la carpeta {models_folder}")

    for model_file in tqdm(sorted(model_files), desc="Processing Models"):
        checkpoint_path = os.path.join(models_folder, model_file)
        print(f"\n\n======================================")
        print(f" Procesando modelo: {model_file}")
        print(f"======================================")

        # Crear modelo DenseNet
        model = get_densenet3d_model(n_classes=2)

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            filtered_checkpoint = {k: v for k, v in checkpoint.items() if "class_layers.out" not in k}
            model.load_state_dict(filtered_checkpoint, strict=False)
            model = model.to(device)
        except Exception as e:
            print(f" Error cargando {model_file}: {e}")
            with open(output_log, "a") as logf:
                logf.write(f"\n=== Modelo: {model_file} ===\n")
                logf.write(f" Error al cargar checkpoint: {e}\n")
            continue


        cross_validate_finetune_densenet(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
            epochs=epochs,
            output_log=output_log,
            model_file_name=model_file,
            checkpoint_path=checkpoint_path,
            k=k
        )

    print("\n TODOS LOS MODELOS PROCESADOS!")

# -------------------- USO --------------------
fine_tune_all_models_in_folder(
    models_folder="/mnt/homeGPU/mcribilles/TFG/pretraining2",
    data_dir="/mnt/homeGPU/mcribilles/TFG/volumenes/preprocesados/preprocesamientos_interesantes/resize_mini_hu_m300_1400_separadas/npy/images",
    mask_dir="/mnt/homeGPU/mcribilles/TFG/volumenes/preprocesados/preprocesamientos_interesantes/resize_mini_hu_m300_1400_separadas/npy/images",
    output_log="resultados_finetuning_densenet_descong_dropout.txt",
    batch_size=16,
    learning_rate=1e-4,
    weight_decay=1e-5,
    epochs=20,
    k=5
)


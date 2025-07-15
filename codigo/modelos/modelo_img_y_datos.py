# librerias
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from monai.networks.nets import DenseNet121
from monai.transforms import Compose, EnsureType

device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 30


#datos clinicos
df = pd.read_csv("./../../datos_clinicos/datos_clinicos_limpio_j.csv", na_values="NaN")

# Limpieza columnas (tipos de complicacion)
columns_to_drop = ["Id_paciente", "Derrame_pleural_leve", "Hemorragia",
                    "Hemorragia_leve", "Neumotórax", "Sin_complicación"]
features_cols = [c for c in df.columns if c not in columns_to_drop + ["Complicacion_binaria"]]

# Generamos un diccionario etiquetas
labels_dict_numeric = dict(zip(df["Id_paciente"], df["Complicacion_binaria"]))

# Diccionario de features
clinical_features_dict = {
    k: df[df["Id_paciente"] == k][features_cols].values.astype(np.float32)[0]
    for k in df["Id_paciente"]
}

# dataset
class LungCTHybridDataset(Dataset):
    def __init__(self, data_dir, mask_dir, labels_dict, features_dict, transform=None):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.labels = labels_dict
        self.features = features_dict
        self.patients = list(labels_dict.keys())
        self.transform = transform

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        pid = self.patients[idx]
        vol = np.load(os.path.join(self.data_dir, f"{pid}.npy"))
        mask_path = os.path.join(self.mask_dir, f"{pid}.npy")
        if os.path.exists(mask_path):
            mask = np.load(mask_path).astype(bool)
            vol = vol * mask
        vol = np.expand_dims(vol, axis=0).astype(np.float32)
        if self.transform:
            vol = self.transform(vol)

        clinical_feat = self.features[pid]
        label = self.labels[pid]

        return (
            torch.tensor(vol, dtype=torch.float32),
            torch.tensor(clinical_feat, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )

transform = Compose([EnsureType()])

#modelo hibrido: datos tabulares + volumenes
class HybridDenseNet(nn.Module):
    def __init__(self, clinical_input_dim, dropout_prob=0.4):
        super().__init__()
        self.img_model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1024, dropout_prob=dropout_prob)
        self.img_model.class_layers = nn.Identity()
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        self.clinical_mlp = nn.Sequential(
            nn.Linear(clinical_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.final_fc = nn.Linear(1024 + 256, 2)

    def forward(self, vol, clin):
        x_img = self.img_model(vol)
        x_img = self.global_pool(x_img).view(x_img.size(0), -1)
        x_clin = self.clinical_mlp(clin)
        x = torch.cat([x_img, x_clin], dim=1)
        return self.final_fc(x)

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

def evaluar_modelo(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for vol, clin, labels in loader:
            vol, clin = vol.to(device), clin.to(device)
            outputs = model(vol, clin)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    tpr, tnr, gmean = calcular_metricas_binarias(all_labels, all_preds)
    return acc, f1, tpr, tnr, gmean

def cross_validate_hybrid_model(
    dataset,
    batch_size,
    learning_rate,
    weight_decay,
    dropout_prob,
    split_seed,
    device,
    epochs,
    save_path_prefix,
    preprocessed_dir,
    k=5
):
    results = []
    labels = [dataset.labels[pid] for pid in dataset.patients]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)  # Outer split fixed

    for fold, (trainval_idx, test_idx) in enumerate(skf.split(dataset.patients, labels)):
        print("\n===========================================")
        print(f" Fold {fold+1}/{k} | BS={batch_size} | LR={learning_rate} | SEED={split_seed}")
        print("===========================================")

        trainval_subset = Subset(dataset, trainval_idx)
        test_subset = Subset(dataset, test_idx)

        # internal split with seed
        internal_labels = [dataset.labels[dataset.patients[i]] for i in trainval_idx]
        train_indices, val_indices = train_test_split(
            trainval_idx,
            test_size=0.2,
            stratify=internal_labels,
            random_state=split_seed
        )

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)

        clinical_dim = next(iter(train_loader))[1].shape[1]
        model = HybridDenseNet(clinical_input_dim=clinical_dim, dropout_prob=dropout_prob).to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        save_path = f"{save_path_prefix}_fold{fold+1}_hybrid.pth"

        # Entrenamiento con validación interna
        train_model_internal_val(
            model, train_loader, val_loader,
            criterion, optimizer,
            device, epochs,
            save_path
        )

        # Evaluación final
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
            "weight_decay": weight_decay,
            "dropout_prob": dropout_prob,
            "seed": split_seed,
            "preprocessed_dir": preprocessed_dir
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
            "weight_decay": weight_decay,
            "dropout_prob": dropout_prob,
            "seed": split_seed,
            "preprocessed_dir": preprocessed_dir
        })

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
            "weight_decay": weight_decay,
            "dropout_prob": dropout_prob,
            "seed": split_seed,
            "preprocessed_dir": preprocessed_dir
        }
        results_df = pd.concat([results_df, pd.DataFrame([mean_row])], ignore_index=True)

    return results_df


# entrenamiento
def train_model_internal_val(
    model, train_loader, val_loader,
    criterion, optimizer, device, epochs,
    save_path
):
    best_gmean = 0.0

    for epoch in range(epochs):
        model.train()
        train_preds, train_labels = [], []
        running_loss = 0.0

        print(f"\n Epoch {epoch+1}/{epochs}")
        for vol, clin, labels in tqdm(train_loader):
            vol, clin, labels = vol.to(device), clin.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(vol, clin)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(train_labels, train_preds)
        f1 = f1_score(train_labels, train_preds)
        tpr, tnr, gmean = calcular_metricas_binarias(train_labels, train_preds)
        print(f" ➤ Train — Loss: {running_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | G-Mean: {gmean:.4f}")

        # val
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for vol, clin, labels in val_loader:
                vol, clin, labels = vol.to(device), clin.to(device), labels.to(device)
                outputs = model(vol, clin)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        val_tpr, val_tnr, val_gmean = calcular_metricas_binarias(val_labels, val_preds)
        print(f" ➤ Valid — Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | G-Mean: {val_gmean:.4f}")

        if epoch == 0 or val_gmean > best_gmean:
            best_gmean = val_gmean
            torch.save(model.state_dict(), save_path)
            print(f"  Guardado modelo con mejor G-Mean ({val_gmean:.4f}) en '{save_path}'")

# main loop
if __name__ == "__main__":

    preprocessing_dirs = [
        "resize_mini_hu_m300_1400_separadas",
        "resize_mini_hu_m600_1500_separadas"
    ]

    batch_sizes_to_try = [4, 16]
    learning_rates_to_try = [1e-3, 1e-4]
    weight_decays_to_try = [0]
    dropout_probs_to_try = [0.4]
    split_seeds_to_try = [9]

    all_results = []

    for seed in split_seeds_to_try:
        print("\n##################################################")
        print(f"=== EMPEZANDO EXPERIMENTOS CON SEED: {seed} ===")
        print("##################################################")

        for prep in preprocessing_dirs:
            print("\n##################################################")
            print(f"=== PREPROCESADO: {prep} | SEED={seed} ===")
            print("##################################################")

            # Load dataset with both images and clinical data
            data_dir = f"/mnt/homeGPU/mcribilles/TFG/volumenes/preprocesados/preprocesamientos_interesantes/{prep}/npy/images"
            mask_dir = f"/mnt/homeGPU/mcribilles/TFG/volumenes/preprocesados/preprocesamientos_interesantes/{prep}/npy/masks"

            dataset = LungCTHybridDataset(data_dir, mask_dir, labels_dict_numeric, clinical_features_dict, transform=transform)

            for bs in batch_sizes_to_try:
                for lr in learning_rates_to_try:
                    for weight_decay in weight_decays_to_try:
                        for dropout_prob in dropout_probs_to_try:

                            print("\n##################################################")
                            print(f"SEED={seed} | Preprocesado={prep} | BS={bs}, LR={lr}, WD={weight_decay}, Dropout={dropout_prob}, Seed={seed}")
                            print("##################################################")

                            # Cross-Validation
                            results_df = cross_validate_hybrid_model(
                                dataset=dataset,
                                batch_size=bs,
                                learning_rate=lr,
                                weight_decay=weight_decay,
                                dropout_prob=dropout_prob,
                                split_seed=seed,
                                device=device,
                                epochs=EPOCHS,
                                save_path_prefix=f"modelo_hybrid_k5_{prep}_bs{bs}_lr{lr}_wd{weight_decay}_drop{dropout_prob}_seed{seed}",
                                preprocessed_dir=prep
                            )

                            # Track metadata for filtering later
                            results_df["preprocessed_dir"] = prep
                            results_df["weight_decay"] = weight_decay
                            results_df["dropout_prob"] = dropout_prob
                            results_df["seed"] = seed
                            all_results.append(results_df)

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv("resultados_hybrid_model_param_grid.csv", index=False)
    print("\n TODOS LOS RESULTADOS GUARDADOS EN 'resultados_hybrid_model_param_grid.csv'")

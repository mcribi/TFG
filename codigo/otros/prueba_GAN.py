import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from monai.transforms import Compose, EnsureType
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
import pandas as pd

print(" Iniciando script GAN")

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
        volume_path = os.path.join(self.data_dir, f"{patient_id}.npy")
        mask_path = os.path.join(self.mask_dir, f"{patient_id}.npy")

        if not os.path.exists(volume_path):
            raise FileNotFoundError(f" Archivo de volumen no encontrado: {volume_path}")

        volume = np.load(volume_path)

        if os.path.exists(mask_path):
            mask = np.load(mask_path).astype(bool)
            volume = volume * mask
        else:
            print(f" Máscara no encontrada para {patient_id}, se usa el volumen completo.")

        volume = np.expand_dims(volume, axis=0)
        if self.transform:
            volume = self.transform(volume)
        return volume, torch.tensor(self.labels[patient_id], dtype=torch.long)


# Generator and Discriminator
class Generator3D(nn.Module):
    def __init__(self, z_dim=100, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose3d(z_dim, 128, 4, 1, 0),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            nn.ConvTranspose3d(128, 64, 4, 2, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.ConvTranspose3d(64, 32, 4, 2, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            nn.ConvTranspose3d(32, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

class Discriminator3D(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, 64, 4, 2, 1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, 4, 2, 1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# gan training
def train_gan(generator, discriminator, data_loader, device, epochs=50, z_dim=100):
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=1e-4)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4)

    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        print(f"\n Epoch {epoch+1}/{epochs}...")
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}", leave=True)
        for step, (real_data, labels) in enumerate(pbar):
            if not torch.any(labels == 1):
                pbar.set_postfix_str(" Sin complicaciones")
                continue

            real_data = real_data[labels == 1].to(device)
            batch_size = real_data.size(0)
            pbar.set_postfix_str(f" Batch con {batch_size} complicaciones")

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            z = torch.randn(batch_size, z_dim, 1, 1, 1, device=device)
            fake_data = generator(z)

            real_output = discriminator(real_data).mean(dim=[2, 3, 4])
            fake_output = discriminator(fake_data.detach()).mean(dim=[2, 3, 4])

            d_loss = criterion(real_output, real_labels) + criterion(fake_output, fake_labels)

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            fake_output = discriminator(fake_data).mean(dim=[2, 3, 4])
            g_loss = criterion(fake_output, real_labels)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

        print(f" Epoch {epoch+1}/{epochs} completado | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")


# Generate Synthetic Samples
def generate_synthetic_samples(generator, n_samples=10, z_dim=100, device="cuda"):
    print(f"\n Generando {n_samples} muestras sintéticas...")
    os.makedirs("synthetic", exist_ok=True)
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, z_dim, 1, 1, 1).to(device)
        generated = generator(z).cpu().numpy()
        for i, sample in enumerate(tqdm(generated, desc="Guardando volúmenes sintéticos")):
            np.save(f"synthetic/complication_{i}.npy", sample[0])

#example
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = "preprocesamientos2/resize_small_separadas/npy/images"
    mask_dir = "preprocesamientos2/resize_small_separadas/npy/masks"

    df = pd.read_csv("./datos_clinicos/datos_filtrados.csv")
    labels_dict = {k: 1 if str(v).lower() in ["sí", "si", "yes", "1", "true"] else 0 for k, v in zip(df.Id_paciente, df.Complicación)}
    transform = Compose([EnsureType()])

    dataset = LungCTDataset(data_dir, mask_dir, labels_dict, transform=transform)
    labels = [dataset.labels[pid] for pid in dataset.patients]


    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, _ = next(sss.split(dataset.patients, labels))

    # solo pacientes con complicación
    train_idx = [i for i in train_idx if dataset[i][1] == 1]
    print(f" Pacientes con complicación en train: {len(train_idx)}")

    train_dataset = Subset(dataset, train_idx)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

    generator = Generator3D().to(device)
    discriminator = Discriminator3D().to(device)

    train_gan(generator, discriminator, train_loader, device, epochs=30)
    generate_synthetic_samples(generator, n_samples=10, device=device)

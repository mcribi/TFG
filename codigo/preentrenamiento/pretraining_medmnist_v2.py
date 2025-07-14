import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.video import r3d_18
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import TripletMarginLoss
import torch.nn.functional as F
from torchvision.models.video import r3d_18
from monai.networks.nets import DenseNet121
import argparse


import medmnist
from medmnist import INFO

def get_model(model_name, n_classes):
    if model_name == 'resnet3d':
        model = r3d_18(pretrained=False)
        model.stem[0] = nn.Conv3d(
            1, 64, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3), bias=False
        )
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif model_name == 'densenet3d':
        model = DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=n_classes
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model

def get_triplet_batch(inputs, targets):
    # simplificado: elige ancla, positivo y negativo aleatorios del batch
    anchors, positives, negatives = [], [], []

    for i in range(len(targets)):
        anchor = inputs[i]
        anchor_label = targets[i]

        # Buscar positivo y negativo
        pos_idx = (targets == anchor_label).nonzero(as_tuple=True)[0]
        neg_idx = (targets != anchor_label).nonzero(as_tuple=True)[0]

        if len(pos_idx) > 1 and len(neg_idx) > 0:
            pos_choice = pos_idx[pos_idx != i][torch.randint(0, len(pos_idx)-1, (1,))]
            neg_choice = neg_idx[torch.randint(0, len(neg_idx), (1,))]
            anchors.append(anchor)  # Eliminar dimensión extra
            positives.append(inputs[pos_choice].squeeze(0))  # Eliminar dimensión extra
            negatives.append(inputs[neg_choice].squeeze(0))  # Eliminar dimensión extra

    if anchors:
        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)
    else:
        return None, None, None



parser = argparse.ArgumentParser(description='MedMNIST3D Pretraining')

parser.add_argument('--experiment_name', type=str, default='prueba')
parser.add_argument('--dataset', type=str, default='organmnist3d')
parser.add_argument('--size', type=int, default=64)
parser.add_argument('--model', type=str, default='resnet3d')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--patience', type=int, default=50)

parser.add_argument('--use_ce', action='store_true')
parser.add_argument('--use_triplet', action='store_true')
parser.add_argument('--use_contrastive', action='store_true')

parser.add_argument('--ce_weight', type=float, default=1.0)
parser.add_argument('--triplet_weight', type=float, default=0.5)
parser.add_argument('--contrastive_weight', type=float, default=0.5)

parser.add_argument('--stop', type=str, default='loss',
                    choices=['loss', 'acc'],
                    help='Early stopping metric: "loss" or "acc"')

args = parser.parse_args()



##########################################
# CONFIGURACIÓN
##########################################

# experiment_name = 'prueba'
# data_flag = 'organmnist3d'   # Cambia el dataset
# desired_size = 64            # ¡AQUÍ CAMBIAS EL TAMAÑO! (28 o 64)
# num_epochs = 50
# batch_size = 16
# learning_rate = 1e-3
# patience = num_epochs  # Guardamos el mejor pero dejamos entrenar hasta el final
# model_choice = 'resnet3d'  # 'resnet3d' or 'densenet3d'

# # Hardcoded options
# use_cross_entropy = True
# use_triplet = True
# use_contrastive = True

# # Loss weights if combining
# ce_weight = 1.0
# triplet_weight = 0.5
# contrastive_weight = 0.5

experiment_name = args.experiment_name
data_flag = args.dataset
desired_size = args.size
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr
patience = args.patience
model_choice = args.model

use_cross_entropy = args.use_ce
use_triplet = args.use_triplet
use_contrastive = args.use_contrastive

ce_weight = args.ce_weight
triplet_weight = args.triplet_weight
contrastive_weight = args.contrastive_weight


str_losses = ""
if use_cross_entropy:
    str_losses += "ceL"
if use_triplet:
    str_losses += "tripletL"
if use_contrastive:
    str_losses += "contrastiveL"

save_path = f'models/pretraining/best_model_{experiment_name}_{model_choice}_{data_flag}_{desired_size}_{str_losses}_stop_{args.stop}.pth'
final_model_path = f'models/pretraining/final_model_{experiment_name}_{model_choice}_{data_flag}_{desired_size}_{str_losses}_stop_{args.stop}.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

##########################################
# DATASET
##########################################

info = INFO[data_flag]
n_classes = len(info['label'])
print(f"Num classes: {n_classes}")

# Import dataset class dynamically
DataClass = getattr(medmnist, info['python_class'])

# MedMNIST v3: supports size argument
transform = None  # The dataset already returns tensors

train_dataset = DataClass(split='train', download=True, size=desired_size, transform=transform)
val_dataset   = DataClass(split='val',   download=True, size=desired_size, transform=transform)
test_dataset  = DataClass(split='test',  download=True, size=desired_size, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2)

##########################################
# MODELO
##########################################

# model = r3d_18(pretrained=False)
# model.stem[0] = nn.Conv3d(
#     1, 64, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3), bias=False
# )
# model.fc = nn.Linear(model.fc.in_features, n_classes)
# model = model.to(device)

model = get_model(model_choice, n_classes).to(device)
print(f"Model: {model_choice} with {n_classes} classes")

##########################################
# LOSS & OPTIMIZER
##########################################

# criterion = nn.CrossEntropyLoss()
# Define losses
cross_entropy_loss = nn.CrossEntropyLoss()
triplet_loss_fn = TripletMarginLoss(margin=1.0, p=2)
contrastive_loss_fn = nn.CosineEmbeddingLoss(margin=0.5)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

##########################################
# EARLY STOPPING
##########################################

class EarlyStopping:
    def __init__(self, patience=5, mode='min', verbose=True, save_path=None):
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.best_score = None
        self.counter = 0
        self.stop = False
        self.save_path = save_path

    def __call__(self, current, model):
        if self.best_score is None:
            self.best_score = current
            if self.save_path:
                torch.save(model.state_dict(), self.save_path)
                if self.verbose:
                    print(f"[EarlyStopping] Saving initial model to {self.save_path}")
            return

        improved = ((self.mode == 'min' and current < self.best_score) or
                    (self.mode == 'max' and current > self.best_score))
        if improved:
            if self.verbose:
                print(f"[EarlyStopping] Improvement found: {self.best_score:.4f} → {current:.4f}")
            self.best_score = current
            self.counter = 0
            if self.save_path:
                torch.save(model.state_dict(), self.save_path)
                if self.verbose:
                    print(f"[EarlyStopping] Saved improved model to {self.save_path}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                if self.verbose:
                    print("[EarlyStopping] Patience exceeded. Stopping training.")
                self.stop = True

##########################################
# TRAIN & EVAL FUNCTIONS
##########################################

# def train_one_epoch(model, dataloader, criterion, optimizer, device):
#     model.train()
#     total_loss = 0
#     correct = 0
#     total = 0

#     progress_bar = tqdm(dataloader, desc="Training", unit="batch")
#     for inputs, targets in progress_bar:
#         inputs = inputs.float().to(device)
#         targets = targets.to(device).long().view(-1)

#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item() * inputs.size(0)
#         _, predicted = torch.max(outputs, 1)
#         correct += (predicted == targets).sum().item()
#         total += targets.size(0)

#         avg_loss = total_loss / total
#         acc = correct / total
#         progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.4f}"})

#     return avg_loss, acc

# def evaluate(model, dataloader, criterion, device):
#     model.eval()
#     total_loss = 0
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         progress_bar = tqdm(dataloader, desc="Validation", unit="batch")
#         for inputs, targets in progress_bar:
#             inputs = inputs.float().to(device)
#             targets = targets.to(device).long().view(-1)

#             outputs = model(inputs)
#             loss = criterion(outputs, targets)

#             total_loss += loss.item() * inputs.size(0)
#             _, predicted = torch.max(outputs, 1)
#             correct += (predicted == targets).sum().item()
#             total += targets.size(0)

#             avg_loss = total_loss / total
#             acc = correct / total
#             progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.4f}"})

#     return avg_loss, acc


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss_sum = 0.0
    ce_sum, triplet_sum, contrastive_sum = 0.0, 0.0, 0.0
    total_samples = 0
    correct = 0

    progress_bar = tqdm(dataloader, desc="Training", unit="batch")
    for inputs, targets in progress_bar:
        inputs = inputs.float().to(device)
        targets = targets.to(device).long().view(-1)

        optimizer.zero_grad()
        loss = 0.0

        # print(f"Input shape before model: {inputs.shape}")

        # Cross-Entropy Loss
        if use_cross_entropy:
            
            outputs = model(inputs)
            ce_loss_value = cross_entropy_loss(outputs, targets)
            loss += ce_weight * ce_loss_value
            ce_sum += ce_loss_value.item() * inputs.size(0)

            # Accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()

        # Triplet Loss
        if use_triplet:
            anc, pos, neg = get_triplet_batch(inputs, targets)
            # print(f"Anchor shape: {anc.shape if anc is not None else 'None'},"
            #         f"Positive shape: {pos.shape if pos is not None else 'None'},"
            #         f" Negative shape: {neg.shape if neg is not None else 'None'}")
            if anc is not None:
                anc_emb = model(anc)
                pos_emb = model(pos)
                neg_emb = model(neg)
                # print(f"Anchor shape: {anc.shape}, Positive shape: {pos.shape}, Negative shape: {neg.shape}")
                triplet_value = triplet_loss_fn(anc_emb, pos_emb, neg_emb)
                loss += triplet_weight * triplet_value
                triplet_sum += triplet_value.item() * len(anc)

        # Contrastive Loss
        if use_contrastive:
            anc, pos, neg = get_triplet_batch(inputs, targets)
            if anc is not None:
                # Positive pairs
                pos_targets = torch.ones(len(anc), device=device)
                anc_emb = model(anc)
                pos_emb = model(pos)
                contrastive_pos = contrastive_loss_fn(anc_emb, pos_emb, pos_targets)
                loss += contrastive_weight * contrastive_pos
                contrastive_sum += contrastive_pos.item() * len(anc)

                # Negative pairs
                neg_targets = -torch.ones(len(anc), device=device)
                neg_emb = model(neg)
                contrastive_neg = contrastive_loss_fn(anc_emb, neg_emb, neg_targets)
                loss += contrastive_weight * contrastive_neg
                contrastive_sum += contrastive_neg.item() * len(anc)

        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss_sum += loss.item() * batch_size
        total_samples += batch_size

        # Logging
        avg_loss = total_loss_sum / total_samples
        acc = correct / total_samples if use_cross_entropy else 0.0

        progress_bar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "acc": f"{acc:.4f}",
            "CE": f"{(ce_sum/total_samples):.4f}" if use_cross_entropy else "N/A",
            "Triplet": f"{(triplet_sum/total_samples):.4f}" if use_triplet else "N/A",
            "Contrast": f"{(contrastive_sum/total_samples):.4f}" if use_contrastive else "N/A"
        })

    avg_loss = total_loss_sum / total_samples
    acc = correct / total_samples if use_cross_entropy else 0.0
    return avg_loss, acc


def evaluate(model, dataloader, device):
    model.eval()
    total_loss_sum = 0.0
    total_samples = 0
    correct = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", unit="batch")
        for inputs, targets in progress_bar:
            inputs = inputs.float().to(device)
            targets = targets.to(device).long().view(-1)

            outputs = model(inputs)
            loss = cross_entropy_loss(outputs, targets)

            total_loss_sum += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total_samples += inputs.size(0)

            avg_loss = total_loss_sum / total_samples
            acc = correct / total_samples

            progress_bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "acc": f"{acc:.4f}"
            })

    avg_loss = total_loss_sum / total_samples
    acc = correct / total_samples
    return avg_loss, acc


##########################################
# TRAINING LOOP
##########################################

# early_stopper = EarlyStopping(
#     patience=patience,
#     mode='min',
#     verbose=True,
#     save_path=save_path
# )

if args.stop == 'loss':
    stop_metric = 'loss'
    mode = 'min'
elif args.stop == 'acc':
    stop_metric = 'acc'
    mode = 'max'
else:
    raise ValueError(f"Invalid --stop argument: {args.stop}")

early_stopper = EarlyStopping(
    patience=patience,
    mode=mode,
    verbose=True,
    save_path=save_path
)


for epoch in range(num_epochs):
    print(f"\n================== Epoch {epoch+1}/{num_epochs} ==================")
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, device)
    scheduler.step()

    print(f"Epoch {epoch+1}:")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    # early_stopper(val_loss, model)

    metric_value = val_loss if args.stop == 'loss' else val_acc
    early_stopper(metric_value, model)

    if early_stopper.stop:
        print("[Training stopped early]")
        break

# final_model_path = f'models/pretraining/final_model_{experiment_name}_{data_flag}_{desired_size}_{str_losses}.pth'
torch.save(model.state_dict(), final_model_path)
print(f"\n[INFO] Last model saved to {final_model_path}")


##########################################
# LOAD BEST MODEL
##########################################

print(f"\nLoading best model from {save_path}")
model.load_state_dict(torch.load(save_path))
model.eval()

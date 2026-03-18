# main_train.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from collections import Counter

# -------------------------
# REPRODUCIBILITY
# -------------------------
torch.manual_seed(42)
np.random.seed(42)

# -------------------------
# PARAMETERS
# -------------------------
WINDOW_SIZE  = 1000
STEP_SIZE    = 500
BATCH_SIZE   = 128   # increased to better utilize GPU throughput
EPOCHS       = 20
LR           = 0.001
VAL_SPLIT    = 0.2
NUM_CHANNELS = 3
NUM_CLASSES  = 3

DATA_PATH = os.environ.get("DATA_PATH", r"C:\Power Pole")

# -------------------------
# DEVICE SETUP
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"  GPU:  {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# -------------------------
# DATA LOADING
# -------------------------
def load_folder(folder, label):
    """Returns a list of (windows, label) grouped per file."""
    files_data = []
    for root, _, files in os.walk(folder):
        for file in sorted(files):
            if not file.endswith(".npy"):
                continue
            full_path = os.path.join(root, file)
            data = np.load(full_path)

            # Validate expected shape: (N, 3)
            assert data.ndim == 2 and data.shape[1] == NUM_CHANNELS, \
                f"Expected shape (N, {NUM_CHANNELS}), got {data.shape} in {full_path}"

            data = data / 1023.0  # normalize to [0, 1]

            if len(data) < WINDOW_SIZE:
                print(f"  Skipped (too short): {full_path}")
                continue

            windows = []
            for i in range(0, len(data) - WINDOW_SIZE + 1, STEP_SIZE):
                windows.append(data[i:i + WINDOW_SIZE])

            files_data.append((windows, label))
            print(f"  Loaded {full_path} → {len(windows)} windows")

    return files_data


print("\nLoading data...")
all_files = []
all_files += load_folder(os.path.join(DATA_PATH, "Good Poles"),   0)
all_files += load_folder(os.path.join(DATA_PATH, "Decent Poles"), 1)
all_files += load_folder(os.path.join(DATA_PATH, "Bad Poles"),    2)

# ── File-level train/val split (no leakage) ──────────────────────────────────
import random
random.seed(42)
random.shuffle(all_files)

val_file_count  = max(1, int(len(all_files) * VAL_SPLIT))
val_files       = all_files[:val_file_count]
train_files     = all_files[val_file_count:]

def files_to_arrays(file_list):
    X, y = [], []
    for windows, label in file_list:
        X.extend(windows)
        y.extend([label] * len(windows))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

X_train, y_train = files_to_arrays(train_files)
X_val,   y_val   = files_to_arrays(val_files)

print(f"\nTrain windows: {len(X_train):,}  |  Val windows: {len(X_val):,}")
print(f"Train files: {len(train_files)}  |  Val files: {len(val_files)}")

# Check class balance
counts = Counter(y_train.tolist())
print(f"Train class distribution: Good={counts[0]}, Decent={counts[1]}, Bad={counts[2]}")

# -------------------------
# DATASET & SPLITS
# -------------------------
class PoleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_set = PoleDataset(X_train, y_train)
val_set   = PoleDataset(X_val,   y_val)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# -------------------------
# CLASS WEIGHTS (imbalance handling)
# -------------------------
class_counts  = torch.tensor([counts[i] for i in range(NUM_CLASSES)], dtype=torch.float32)
class_weights = (1.0 / class_counts)
class_weights = (class_weights / class_weights.sum()).to(device)

# -------------------------
# MODEL
# -------------------------
class PoleNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(NUM_CHANNELS, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 2
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 4
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Global average pooling collapses (batch, 256, T) → (batch, 256)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)       # (batch, time, 3) → (batch, 3, time)
        x = self.features(x)          # (batch, 256, T')
        x = self.gap(x).squeeze(-1)   # (batch, 256)
        return self.classifier(x)     # (batch, num_classes)


model     = PoleNet().to(device)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# -------------------------
# EVALUATION
# -------------------------
def evaluate(model, loader):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            total_loss += loss.item()
            preds    = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return total_loss, correct / total

# -------------------------
# TRAINING LOOP
# -------------------------
print("\nTraining...\n")
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss    = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()

    val_loss, val_acc = evaluate(model, val_loader)
    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc*100:.1f}% | "
          f"LR: {current_lr:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "pole_model_best.pth")
        print("  ✓ Saved best model")

print("\nDone. Best model saved as pole_model_best.pth")
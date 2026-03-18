
import os
import random
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
random.seed(42)

# -------------------------
# PARAMETERS
# -------------------------
WINDOW_SIZE  = 1000
STEP_SIZE    = 500
BATCH_SIZE   = 128
EPOCHS       = 20
LR           = 0.001
NUM_CHANNELS = 3
NUM_CLASSES  = 3

DATA_PATH = os.environ.get("DATA_PATH", r"C:\Users\Temple\Downloads")  # your actual path here")

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
# Load each pole as its own entry: { 'windows': [...], 'label': int, 'name': str }
# -------------------------
def load_folder(folder, label):
    poles = []
    for pole_dir in sorted(os.listdir(folder)):
        pole_path = os.path.join(folder, pole_dir)
        if not os.path.isdir(pole_path):
            continue
        windows = []
        for root, _, files in os.walk(pole_path):
            for file in sorted(files):
                if not file.endswith(".npy"):
                    continue
                full_path = os.path.join(root, file)
                data = np.load(full_path)
                assert data.ndim == 2 and data.shape[1] == NUM_CHANNELS, \
                    f"Expected (N, {NUM_CHANNELS}), got {data.shape} in {full_path}"
                data = data / 1023.0
                if len(data) < WINDOW_SIZE:
                    continue
                for i in range(0, len(data) - WINDOW_SIZE + 1, STEP_SIZE):
                    windows.append(data[i:i + WINDOW_SIZE])
        if windows:
            poles.append({"name": pole_dir, "label": label, "windows": windows})
            print(f"  {pole_dir} ({len(windows)} windows)")
    return poles


print("\nLoading data...")
print("Good Poles:")
good_poles   = load_folder(os.path.join(DATA_PATH, "Good Poles"),   0)
print("Decent Poles:")
decent_poles = load_folder(os.path.join(DATA_PATH, "Decent Poles"), 1)
print("Bad Poles:")
bad_poles    = load_folder(os.path.join(DATA_PATH, "Bad Poles"),    2)

all_poles = good_poles + decent_poles + bad_poles
print(f"\nTotal poles: {len(all_poles)}  "
      f"(Good: {len(good_poles)}, Decent: {len(decent_poles)}, Bad: {len(bad_poles)})")

# -------------------------
# DATASET
# -------------------------
class PoleDataset(Dataset):
    def __init__(self, poles):
        X, y = [], []
        for pole in poles:
            X.extend(pole["windows"])
            y.extend([pole["label"]] * len(pole["windows"]))
        self.X = torch.tensor(np.array(X, dtype=np.float32))
        self.y = torch.tensor(np.array(y, dtype=np.int64))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------------------
# MODEL
# -------------------------
class PoleNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(NUM_CHANNELS, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.features(x)
        x = self.gap(x).squeeze(-1)
        return self.classifier(x)

# -------------------------
# TRAIN + EVAL HELPERS
# -------------------------
def make_criterion(train_poles):
    counts = Counter(l for p in train_poles for l in [p["label"]] * len(p["windows"]))
    class_counts  = torch.tensor([counts[i] for i in range(NUM_CLASSES)], dtype=torch.float32)
    class_weights = (1.0 / class_counts)
    class_weights = (class_weights / class_weights.sum()).to(device)
    return nn.CrossEntropyLoss(weight=class_weights)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss

def evaluate(model, loader, criterion):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total   += labels.size(0)
    return total_loss, correct / total

# -------------------------
# LEAVE-ONE-POLE-OUT CV
# -------------------------
class_names  = ["Good", "Decent", "Bad"]
fold_results = []

print("\n" + "="*60)
print("LEAVE-ONE-POLE-OUT CROSS VALIDATION")
print("="*60)

for fold_idx, val_pole in enumerate(all_poles):
    train_poles = [p for p in all_poles if p is not val_pole]

    print(f"\nFold {fold_idx+1:02d}/{len(all_poles)} — Val pole: "
          f"{val_pole['name']} (label: {class_names[val_pole['label']]})")

    train_set    = PoleDataset(train_poles)
    val_set      = PoleDataset([val_pole])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model     = PoleNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = make_criterion(train_poles)

    best_val_acc  = 0.0
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        train_loss         = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc  = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            torch.save(model.state_dict(), f"pole_model_fold{fold_idx+1}.pth")

        print(f"  Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.3f} | "
              f"Val Loss: {val_loss:.3f} | "
              f"Val Acc: {val_acc*100:.1f}%")

    fold_results.append({
        "pole":     val_pole["name"],
        "label":    class_names[val_pole["label"]],
        "best_acc": best_val_acc,
    })
    print(f"  ✓ Best val acc for this pole: {best_val_acc*100:.1f}%")

# -------------------------
# SUMMARY
# -------------------------
print("\n" + "="*60)
print("CROSS VALIDATION SUMMARY")
print("="*60)
for r in fold_results:
    print(f"  {r['pole']:<30} ({r['label']:<6}) → {r['best_acc']*100:.1f}%")

avg_acc = sum(r["best_acc"] for r in fold_results) / len(fold_results)
print(f"\nMean accuracy across all poles: {avg_acc*100:.1f}%")

for cls_idx, cls_name in enumerate(class_names):
    cls_results = [r["best_acc"] for r in fold_results if r["label"] == cls_name]
    if cls_results:
        print(f"  {cls_name} poles mean acc: {sum(cls_results)/len(cls_results)*100:.1f}%")

print("\nDone.")
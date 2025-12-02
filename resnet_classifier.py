import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, hamming_loss, accuracy_score
from PIL import Image
import json
import csv
from torch.utils.tensorboard import SummaryWriter
import warnings

warnings.filterwarnings('ignore')

# Config

class Config:
    IMG_DIR = "/kaggle/working/png_converted"
    CSV_FILE = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train.csv"
    BATCH_SIZE = 8
    NUM_EPOCHS = 5
    LR = 1e-4
    NUM_CLASSES = 15
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CHECKPOINT_DIR = "/kaggle/working/checkpoints"
    LOG_FILE = "/kaggle/working/training_history.json"
    CSV_RESULTS = "/kaggle/working/results.csv"
    TENSORBOARD_DIR = "/kaggle/working/tensorboard_logs"
    SEED = 42

os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.TENSORBOARD_DIR, exist_ok=True)

torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)

# Dataset

class XRayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        self.class_map = {
            "Aortic enlargement": 0, "Atelectasis": 1, "Calcification": 2,
            "Cardiomegaly": 3, "Consolidation": 4, "ILD": 5, "Infiltration": 6,
            "Lung Opacity": 7, "No finding": 8, "Nodule/Mass": 9, "Other lesion": 10,
            "Pleural effusion": 11, "Pleural thickening": 12, "Pneumothorax": 13,
            "Pulmonary fibrosis": 14
        }

    def __len__(self):
        return self.df['image_id'].nunique()

    def __getitem__(self, idx):
        img_id = self.df['image_id'].unique()[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        labels = self.df[self.df['image_id'] == img_id]['class_name'].tolist()
        multi_hot = np.zeros(Config.NUM_CLASSES, dtype=np.float32)

        for label in labels:
            if label in self.class_map:
                multi_hot[self.class_map[label]] = 1.0

        return {'image': img, 'labels': torch.tensor(multi_hot)}

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# DataLoaders

dataset = XRayDataset(Config.CSV_FILE, Config.IMG_DIR, transform=transform)

# Manual train/validation split
num_train = int(0.8 * len(dataset))
num_val = len(dataset) - num_train
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [num_train, num_val])

train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)

# Model

model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, Config.NUM_CLASSES)
model = model.to(Config.DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# TensorBoard
writer = SummaryWriter(Config.TENSORBOARD_DIR)


# CSV Writer

with open(Config.CSV_RESULTS, "w", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_auc", "hamming_loss", "accuracy"])


# Training

history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_hamming': [], 'val_accuracy': []}
best_val_loss = float("inf")

for epoch in range(Config.NUM_EPOCHS):

    # --- Train ---
    model.train()
    train_loss = 0

    for batch in train_loader:
        imgs = batch['image'].to(Config.DEVICE)
        labels = batch['labels'].to(Config.DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # --- Validation ---
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch['image'].to(Config.DEVICE)
            labels = batch['labels'].to(Config.DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    val_loss /= len(val_loader)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    bin_preds = (all_preds >= 0.5).astype(int)

    # Metrics
    hamming = hamming_loss(all_labels, bin_preds)
    subset_acc = accuracy_score(all_labels, bin_preds)

    aucs = [
        roc_auc_score(all_labels[:, i], all_preds[:, i])
        for i in range(Config.NUM_CLASSES)
        if len(np.unique(all_labels[:, i])) > 1
    ]
    mean_auc = np.mean(aucs) if aucs else 0

    # Scheduler
    scheduler.step(val_loss)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "best_resnet50.pt"))

    # TensorBoard logging
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Metrics/val_auc", mean_auc, epoch)

    # CSV logging
    with open(Config.CSV_RESULTS, "a", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            epoch+1, train_loss, val_loss, mean_auc, hamming, subset_acc
        ])

    print(
        f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
        f"Train {train_loss:.4f} | Val {val_loss:.4f} | AUC {mean_auc:.4f}"
    )

    # Save history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_auc'].append(mean_auc)
    history['val_hamming'].append(hamming)
    history['val_accuracy'].append(subset_acc)

writer.close()

# Save training history JSON
with open(Config.LOG_FILE, "w") as f:
    json.dump(history, f, indent=4)

print("Training complete!")
print(f"Best model saved at {Config.CHECKPOINT_DIR}")
print(f"CSV results stored at {Config.CSV_RESULTS}")
print(f"TensorBoard logs saved to {Config.TENSORBOARD_DIR}")

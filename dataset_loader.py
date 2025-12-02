import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for VinBigData Chest X-ray Dataset
    Supports both classification and detection tasks.
    """
    def __init__(self, img_dir, csv_file, task="classification", img_size=256, transform=None):
        self.img_dir = img_dir
        self.csv_file = csv_file
        self.task = task
        self.img_size = img_size
        self.transform = transform

        # Load CSV
        df = pd.read_csv(csv_file)
        self.annotations = df

        # Unique image IDs
        self.all_image_ids = df["image_id"].unique()

        png_files = [f for f in os.listdir(img_dir) if f.endswith(".png")]
        self.available_ids = [p.replace(".png", "") for p in png_files]

        # Filter annotations to available images only
        self.annotations = df[df["image_id"].isin(self.available_ids)]

        # Class mapping
        self.class_names = sorted(df["class_name"].unique())
        self.class_to_id = {cls: i for i, cls in enumerate(self.class_names)}

        print("===== DATASET LOADED =====")
        print("Images available:", len(self.available_ids))
        print("Total annotations:", len(self.annotations))
        print("Task:", task)
        print("Classes:", self.class_names)
        print("==========================")

    def __len__(self):
        return len(self.available_ids)

    def __getitem__(self, idx):
        image_id = self.available_ids[idx]
        image_path = os.path.join(self.img_dir, image_id + ".png")

        # Load Image
        try:
            img = Image.open(image_path).convert("L")
            img = np.array(img)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        # Normalize and resize
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img = cv2.resize(img, (self.img_size, self.img_size))

        # Convert to 3 channels and torch tensor
        img = np.stack([img] * 3, axis=0)
        img = torch.tensor(img, dtype=torch.float32) / 255.0

        # Get annotations for this image
        ann = self.annotations[self.annotations["image_id"] == image_id]

        if self.task == "classification":
            return self._classification(img, ann)
        else:
            return self._detection(img, ann)

    #  Classification
    def _classification(self, img, ann):
        num_classes = len(self.class_names)
        label_vec = np.zeros(num_classes, dtype=np.float32)

        if len(ann) > 0:
            img_classes = ann["class_name"].unique()
            for cls in img_classes:
                label_vec[self.class_to_id[cls]] = 1.0

        return {
            "image": img,
            "labels": torch.tensor(label_vec),
            "class_names": self.class_names
        }

    # Detection
    def _detection(self, img, ann):
        bboxes = []
        labels = []

        ann = ann.dropna(subset=["x_min", "y_min", "x_max", "y_max"])

        for _, row in ann.iterrows():
            w = row.get("width", 1024)
            h = row.get("height", 1024)

            # Normalize bbox
            x1 = row["x_min"] / w
            y1 = row["y_min"] / h
            x2 = row["x_max"] / w
            y2 = row["y_max"] / h

            bboxes.append([x1, y1, x2, y2])
            labels.append(self.class_to_id[row["class_name"]])

        return {
            "image": img,
            "bboxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "class_names": self.class_names
        }


# Dataloader wrapper
def create_dataloaders(img_dir, csv_file, batch=8, task="classification"):
    dataset = ChestXrayDataset(img_dir, csv_file, task=task)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=2)

    print("\n===== DATALOADERS READY =====")
    print("Train samples:", len(train_ds))
    print("Val samples:", len(val_ds))
    print("Batch size:", batch)

    return train_loader, val_loader


# Test the dataset
if __name__ == "__main__":
    IMG = "/kaggle/working/png_converted"  # PNG folder from dicom_to_png
    CSV = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train.csv"

    ds = ChestXrayDataset(IMG, CSV, task="classification")

    print("\nTesting dataset sample...")
    s = ds[0]
    print("Image shape:", s["image"].shape)
    print("Labels:", s["labels"])

    train_loader, val_loader = create_dataloaders(IMG, CSV, batch=4)

    for batch in train_loader:
        print("Batch images:", batch["image"].shape)
        print("Batch labels:", batch["labels"].shape)
        break

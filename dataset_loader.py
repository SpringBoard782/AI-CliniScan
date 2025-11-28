import os
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image


class VinDrCXRDataset(Dataset):
    def __init__(self,
                 img_dir,
                 csv_file,
                 transform=None,
                 task='classification',
                 img_size=256):
        self.img_dir = img_dir
        self.transform = transform
        self.task = task
        self.img_size = img_size

        # Load CSV annotations
        self.annotations = pd.read_csv(csv_file)
        self.img_names = self.annotations['image_id'].unique()

        # Get available PNG files
        self.available_images = [
            f for f in os.listdir(img_dir)
            if f.endswith('.png')
        ]

        # Get unique class names and create mapping
        self.class_names = sorted(self.annotations['class_name'].unique())
        self.class_to_id = {cls: idx for idx, cls in enumerate(self.class_names)}
        self.id_to_class = {idx: cls for cls, idx in self.class_to_id.items()}

        print(f"Dataset initialized:")
        print(f"  Total unique images: {len(self.img_names)}")
        print(f"  Available PNG files: {len(self.available_images)}")
        print(f"  Total annotations: {len(self.annotations)}")
        print(f"  Number of classes: {len(self.class_names)}")
        print(f"  Task: {task}")
        print(f"  Classes: {self.class_names}")

    def __len__(self):
        return len(self.available_images)
    def __getitem__(self, idx):
        img_file = self.available_images[idx]
        img_path = os.path.join(self.img_dir, img_file)

        # Load image
        try:
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            image = np.array(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return empty sample
            image = np.zeros((512, 512), dtype=np.uint8)

        # Normalize
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Resize if needed
        if image.shape != (self.img_size, self.img_size):
            image = cv2.resize(image, (self.img_size, self.img_size))

        # Convert to 3 channels (for compatibility with ResNet, etc.)
        image = np.stack([image] * 3, axis=0)  # (3, H, W)
        image = torch.from_numpy(image).float() / 255.0

        # Get annotations for this image
        img_id = img_file.replace('.png', '')
        img_annotations = self.annotations[
            self.annotations['image_id'] == img_id
            ]

        if self.task == 'classification':
            return self._get_classification_sample(image, img_annotations)
        else:  # detection
            return self._get_detection_sample(image, img_annotations)

    def _get_classification_sample(self, image, img_annotations):
        # Create binary label vector (one-hot for multi-class, or multi-hot for multi-label)
        num_classes = len(self.class_names)
        labels = np.zeros(num_classes, dtype=np.float32)

        if len(img_annotations) > 0:
            # Get unique classes for this image
            classes_in_image = img_annotations['class_name'].unique()
            for class_name in classes_in_image:
                if class_name in self.class_to_id:
                    class_idx = self.class_to_id[class_name]
                    labels[class_idx] = 1.0

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'labels': torch.from_numpy(labels),
            'class_names': self.class_names
        }

    def _get_detection_sample(self, image, img_annotations):
        """
        For detection: return image, bounding boxes, and class labels
        """
        bboxes = []
        labels = []

        if len(img_annotations) > 0:
            # Filter out rows with missing bounding boxes (No finding class)
            img_annotations_with_bbox = img_annotations.dropna(subset=['x_min', 'y_min', 'x_max', 'y_max'])

            for idx, row in img_annotations_with_bbox.iterrows():
                try:
                    x_min = float(row['x_min'])
                    y_min = float(row['y_min'])
                    x_max = float(row['x_max'])
                    y_max = float(row['y_max'])

                    # Normalize bbox to [0, 1] based on image dimensions
                    width = float(row.get('width', self.img_size))
                    height = float(row.get('height', self.img_size))

                    x_min_norm = x_min / width
                    y_min_norm = y_min / height
                    x_max_norm = x_max / width
                    y_max_norm = y_max / height

                    # Ensure coordinates are within [0, 1]
                    x_min_norm = max(0, min(1, x_min_norm))
                    y_min_norm = max(0, min(1, y_min_norm))
                    x_max_norm = max(0, min(1, x_max_norm))
                    y_max_norm = max(0, min(1, y_max_norm))

                    bboxes.append([x_min_norm, y_min_norm, x_max_norm, y_max_norm])

                    # Get class label
                    class_name = row['class_name']
                    class_id = self.class_to_id.get(class_name, 0)
                    labels.append(class_id)
                except Exception as e:
                    print(f"Error processing bbox: {e}")
                    continue

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'bboxes': torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long),
            'class_names': self.class_names
        }


def create_data_loaders(dataset_dir, csv_file, batch_size=8, num_workers=2, task='classification'):
    # Create dataset
    dataset = VinDrCXRDataset(
        img_dir=dataset_dir,
        csv_file=csv_file,
        task=task,
        img_size=256
    )

    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    from torch.utils.data import random_split
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\n✓ Data loaders created:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of workers: {num_workers}")

    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    IMG_DIR = r"D:\Infosys Springboard Internship\Project\Dataset\vinbigdata-chest-xray-abnormalities-detection-512x512-jp2\vinbigdata_png"
    CSV_FILE = r"D:\Infosys Springboard Internship\Project\Dataset\vinbigdata-chest-xray-abnormalities-detection-512x512-jp2\train.csv"

    # Create dataset and check structure
    print("Creating dataset...")
    dataset = VinDrCXRDataset(IMG_DIR, CSV_FILE, task='classification')

    # Get a sample
    print("\nGetting sample...")
    sample = dataset[0]
    print(f"\nSample batch:")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Labels shape: {sample['labels'].shape}")
    print(f"  Class names: {sample['class_names']}")

    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader = create_data_loaders(
        IMG_DIR,
        CSV_FILE,
        batch_size=8,
        task='classification'
    )

    # Test data loader
    print(f"\nTesting data loader...")
    for batch in train_loader:
        print(f"Batch images shape: {batch['image'].shape}")
        print(f"Batch labels shape: {batch['labels'].shape}")
        print(f"Sample labels: {batch['labels'][0]}")
        break

    print("\n✓ Dataset and DataLoader working correctly!")
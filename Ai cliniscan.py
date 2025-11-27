!pip install pydicom pillow opencv-python pandas numpy torch torchvision -q

import pydicom
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import os

# Step 1: Download VinDr-CXR Dataset
print("📥 Downloading VinDr-CXR dataset...")
os.system("wget -q https://physionet.org/files/vindr-cxr/1.0.0/ -O /tmp/vindr_cxr.zip")
# Note: Requires PhysioNet access. Register at https://physionet.org/

# Step 2: DICOM to PNG Conversion
def convert_dicom_to_png(dicom_path, output_path, resize=(512, 512)):
    """Convert DICOM image to PNG with normalization"""
    try:
        dcm = pydicom.dcmread(dicom_path)
        img = dcm.pixel_array
        
        # Normalize pixel values to 0-255
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        
        # Resize
        img = cv2.resize(img, resize, interpolation=cv2.INTER_CUBIC)
        
        # Save as PNG
        cv2.imwrite(output_path, img)
        return True
    except Exception as e:
        print(f"Error converting {dicom_path}: {e}")
        return False

# Step 3: Batch Processing
def batch_convert_dicom(input_dir, output_dir, limit=None):
    """Convert all DICOM files in directory"""
    os.makedirs(output_dir, exist_ok=True)
    dicom_files = list(Path(input_dir).glob("*.dcm"))
    
    if limit:
        dicom_files = dicom_files[:limit]
    
    success = 0
    for i, dcm_file in enumerate(dicom_files):
        output_file = os.path.join(output_dir, f"{dcm_file.stem}.png")
        if convert_dicom_to_png(str(dcm_file), output_file):
            success += 1
        
        if (i + 1) % 100 == 0:
            print(f"✅ Converted {i + 1}/{len(dicom_files)} images")
    
    print(f"✅ Successfully converted {success}/{len(dicom_files)} images")
    return success

# Step 4: Parse Annotations (CSV to YOLO format)
def parse_vindr_annotations(csv_path):
    """Parse VinDr-CXR annotations CSV"""
    df = pd.read_csv(csv_path)
    print(f"📋 Loaded {len(df)} annotations")
    print(df.head())
    return df

def convert_to_yolo_format(df, output_dir, image_size=512):
    """Convert bounding boxes to YOLO format"""
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, row in df.iterrows():
        image_id = row['image_id']
        x_min, y_min = row['x_min'], row['y_min']
        x_max, y_max = row['x_max'], row['y_max']
        label = row['class_id']  # 0=Aortic enlargement, 1=Atelectasis, etc.
        
        # Convert to YOLO format (center_x, center_y, width, height in [0,1])
        center_x = ((x_min + x_max) / 2) / image_size
        center_y = ((y_min + y_max) / 2) / image_size
        width = (x_max - x_min) / image_size
        height = (y_max - y_min) / image_size
        
        # Save to .txt file
        output_file = os.path.join(output_dir, f"{image_id}.txt")
        with open(output_file, 'a') as f:
            f.write(f"{label} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"✅ Converted {len(df)} annotations to YOLO format")

# Step 5: Data Validation
def validate_dataset(image_dir, annotation_dir):
    """Check dataset integrity"""
    images = set(Path(image_dir).glob("*.png"))
    annotations = set(Path(annotation_dir).glob("*.txt"))
    
    image_names = {f.stem for f in images}
    annotation_names = {f.stem for f in annotations}
    
    print(f"📊 Dataset Statistics:")
    print(f"  Total images: {len(image_names)}")
    print(f"  Total annotations: {len(annotation_names)}")
    print(f"  Matching pairs: {len(image_names & annotation_names)}")
    print(f"  Unmatched images: {len(image_names - annotation_names)}")
    print(f"  Unmatched annotations: {len(annotation_names - image_names)}")

# Usage Example:
# batch_convert_dicom('/path/to/dicom', '/path/to/png')
# df = parse_vindr_annotations('/path/to/annotations.csv')
# convert_to_yolo_format(df, '/path/to/labels')
# validate_dataset('/path/to/png', '/path/to/labels')
!pip install torch torchvision timm pytorch-lightning -q

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# Step 1: Define Custom Dataset
class ChestXrayDataset(Dataset):
    def _init_(self, image_dir, annotations_dir, transform=None):
        self.image_dir = image_dir
        self.annotations_dir = annotations_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        self.transform = transform
    
    def _len_(self):
        return len(self.image_files)
    
    def _getitem_(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Load label from annotation file
        ann_path = os.path.join(self.annotations_dir, img_name.replace('.png', '.txt'))
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                labels = [int(line.split()[0]) for line in f.readlines()]
            label = torch.zeros(14)  # VinDr-CXR has 14 findings
            for l in labels:
                label[l] = 1
        else:
            label = torch.zeros(14)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Step 2: Define Model Architecture
class ChestXrayClassifier(nn.Module):
    def _init_(self, model_name='efficientnet_b0', num_classes=14, pretrained=True):
        super()._init_()
        
        if model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            self.backbone.classifier[1] = nn.Linear(1280, num_classes)
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.backbone.fc = nn.Linear(2048, num_classes)
        
        self.sigmoid = nn.Sigmoid()  # Multi-label classification
    
    def forward(self, x):
        return self.sigmoid(self.backbone(x))

# Step 3: Data Augmentation
transform_train = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Step 4: Training Function
def train_classifier(train_dir, val_dir, epochs=20, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load datasets
    train_dataset = ChestXrayDataset(train_dir, train_dir, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Initialize model
    model = ChestXrayClassifier(model_name='efficientnet_b0', num_classes=14)
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"🚀 Training on {device.upper()}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        scheduler.step()
        print(f"✅ Epoch {epoch+1} completed. Avg Loss: {total_loss/len(train_loader):.4f}")
    
    torch.save(model.state_dict(), 'classifier_model.pth')
    print("💾 Model saved as 'classifier_model.pth'")
    return model

# Step 5: Evaluation
def evaluate_classifier(model, val_dir, device):
    from sklearn.metrics import roc_auc_score, f1_score
    
    model.eval()
    val_dataset = ChestXrayDataset(val_dir, val_dir, transform=transform_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images).cpu()
            
            all_preds.append(outputs.numpy())
            all_labels.append(labels.numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    auc = roc_auc_score(all_labels, all_preds, average='micro')
    f1 = f1_score(all_labels, (all_preds > 0.5).astype(int), average='micro')
    
    print(f"📊 Validation Metrics:")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    return {'auc': auc, 'f1': f1}
  !pip install ultralytics opencv-python -q

from ultralytics import YOLO
import cv2
import torch
import os

# Step 1: Prepare YOLO Dataset Structure
def create_yolo_dataset_structure(train_images, train_labels, val_images, val_labels, output_dir):
    """Create proper YOLO dataset structure"""
    os.makedirs(f'{output_dir}/images/train', exist_ok=True)
    os.makedirs(f'{output_dir}/images/val', exist_ok=True)
    os.makedirs(f'{output_dir}/labels/train', exist_ok=True)
    os.makedirs(f'{output_dir}/labels/val', exist_ok=True)
    
    # Copy training files
    for img, label in zip(train_images, train_labels):
        shutil.copy(img, f'{output_dir}/images/train/')
        shutil.copy(label, f'{output_dir}/labels/train/')
    
    # Copy validation files
    for img, label in zip(val_images, val_labels):
        shutil.copy(img, f'{output_dir}/images/val/')
        shutil.copy(label, f'{output_dir}/labels/val/')
    
    print("✅ Dataset structure created")

# Step 2: Create YAML Configuration
yaml_content = """path: /path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 14
names: ['Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly', 
        'Consolidation', 'Emphysema', 'Effusion', 'Fibrosis', 'Fracture', 
        'Infiltration', 'Lesion', 'Nodule', 'Pleural thickening', 'Pneumothorax']
"""

with open('/content/dataset.yaml', 'w') as f:
    f.write(yaml_content)

print("✅ YAML config created")

# Step 3: Initialize and Train YOLOv8
def train_yolov8(data_yaml, epochs=50, batch_size=16, img_size=640):
    """Train YOLOv8 detection model"""
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    # Load YOLOv8 model
    model = YOLO('yolov8m.pt')  # medium model
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        patience=10,
        save=True,
        amp=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.9,
        flipud=0.0,
        fliplr=0.5,
        perspective=0.0,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0
    )
    
    return model, results

# Step 4: Validation & Metrics
def validate_yolov8(model, data_yaml):
    """Validate model and get metrics"""
    metrics = model.val(data=data_yaml)
    
    print(f"📊 Validation Results:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    return metrics

# Step 5: Inference on Test Images
def run_inference(model_path, image_path, conf_threshold=0.5):
    """Run detection on single image"""
    model = YOLO(model_path)
    results = model.predict(image_path, conf=conf_threshold)
    
    # Visualize
    result = results[0]
    annotated_frame = result.plot()
    
    cv2.imshow('YOLOv8 Detection', annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Print detections
    print(f"🎯 Detected {len(result.boxes)} objects")
    for box in result.boxes:
        class_id = int(box.cls)
        confidence = float(box.conf)
        print(f"  Class: {class_id}, Confidence: {confidence:.2%}")

# Step 6: Batch Inference
def batch_inference(model_path, image_dir, output_dir, conf_threshold=0.5):
    """Run inference on multiple images"""
    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)
    
    for img_name in os.listdir(image_dir):
        if img_name.endswith('.png'):
            img_path = os.path.join(image_dir, img_name)
            results = model.predict(img_path, conf=conf_threshold)
            
            result = results[0]
            annotated = result.plot()
            
            output_path = os.path.join(output_dir, img_name)
            cv2.imwrite(output_path, annotated)
    
    print(f"✅ Batch inference completed. Results saved to {output_dir}")

# Usage:
# model, results = train_yolov8('/content/dataset.yaml', epochs=50)
# metrics = validate_yolov8(model, '/content/dataset.yaml')
# batch_inference('runs/detect/train/weights/best.pt', '/test/images', '/output')
!pip install torch torchvision scikit-learn matplotlib seaborn tensorboard -q

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Step 1: Advanced Training Loop
class TrainerConfig:
    def _init_(self):
        self.epochs = 30
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.patience = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.early_stop_counter = 0
        self.best_val_loss = float('inf')

class ModelTrainer:
    def _init_(self, model, train_loader, val_loader, config):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.device
        
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs
        )
        
        self.history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_f1': []}
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % max(10, len(self.train_loader) // 5) == 0:
                print(f"  Batch [{batch_idx+1}/{len(self.train_loader)}] Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate_epoch(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        avg_loss = total_loss / len(self.val_loader)
        auc = roc_auc_score(all_labels, all_preds, average='micro')
        f1 = f1_score(all_labels, (all_preds > 0.5).astype(int), average='micro')
        
        return avg_loss, auc, f1, all_preds, all_labels
    
    def train(self):
        print(f"🚀 Starting training on {self.device.upper()}")
        
        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch+1}/{self.config.epochs}")
            
            train_loss = self.train_epoch()
            val_loss, val_auc, val_f1, preds, labels = self.validate_epoch()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_auc'].append(val_auc)
            self.history['val_f1'].append(val_f1)
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}")
            
            # Early stopping
            if val_loss < self.config.best_val_loss:
                self.config.best_val_loss = val_loss
                self.config.early_stop_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
                print("💾 Best model saved")
            else:
                self.config.early_stop_counter += 1
                if self.config.early_stop_counter >= self.config.patience:
                    print(f"⏹  Early stopping triggered after {self.config.patience} epochs without improvement")
                    break
            
            self.scheduler.step()
        
        return self.history

# Step 2: Confusion Matrix Visualization
def plot_confusion_matrix(model, val_loader, device, num_classes=14):
    """Generate confusion matrix heatmap"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images).cpu()
            all_preds.append(outputs.numpy())
            all_labels.append(labels.numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Convert to binary predictions
    pred_binary = (all_preds > 0.5).astype(int)
    
    cm = confusion_matrix(all_labels.flatten(), pred_binary.flatten())
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix - Chest X-Ray Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

# Step 3: Training Curves
def plot_training_history(history):
    """Plot loss and metrics curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (BCE)')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Metrics curve
    axes[1].plot(history['val_auc'], label='Val AUC', marker='o')
    axes[1].plot(history['val_f1'], label='Val F1', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Validation Metrics')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

# Step 4: Performance Report
def generate_performance_report(model, val_loader, device, class_names):
    """Generate comprehensive performance report"""
    from sklearn.metrics import classification_report
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images).cpu()
            all_preds.append(outputs.numpy())
            all_labels.append(labels.numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    pred_binary = (all_preds > 0.5).astype(int)
    
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE PERFORMANCE REPORT")
    print("="*60)
    
    for i in range(len(class_names)):
        report = classification_report(all_labels[:, i], pred_binary[:, i],
                                      target_names=['Negative', 'Positive'],
                                      digits=4)
        print(f"\n{class_names[i]}:\n{report}")
    
    return all_preds, all_labels
  !pip install grad-cam torch torchvision matplotlib opencv-python -q

import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Step 1: Grad-CAM Implementation
class GradCAM:
    def _init_(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_backward_hook(self.save_gradients)
    
    def save_activations(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, class_idx=None):
        """Generate Grad-CAM heatmap"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward(retain_graph=True)
        
        # Calculate CAM
        weights = self.gradients.mean(dim=[2, 3])[0]
        cam = (weights[:, None, None] * self.activations[0]).sum(dim=0)
        cam = F.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()

# Step 2: Visualization Function
def visualize_gradcam(model, image_path, target_layer, output_path='gradcam.png'):
    """Generate and visualize Grad-CAM"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    # Load and process image
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    input_tensor = transform(image).unsqueeze(0)
    
    # Generate Grad-CAM
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate_cam(input_tensor)
    
    # Resize CAM to match image size
    cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
    
    # Create heatmap
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Overlay heatmap on original image
    overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_image)
    axes[0].set_title('Original X-Ray')
    axes[0].axis('off')
    
    axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

# Step 3: Bounding Box Visualization (for Detection)
def visualize_detections(image_path, predictions, output_path='detections.png'):
    """Visualize detection bounding boxes"""
    image = cv2.imread(image_path)
    
    for pred in predictions:
        x1, y1, x2, y2 = pred['bbox']
        class_name = pred['class']
        confidence = pred['confidence']
        
        # Draw bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Add label
        label = f"{class_name}: {confidence:.2%}"
        cv2.putText(image, label, (int(x1), int(y1) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imwrite(output_path, image)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Detection Results')
    plt.axis('off')
    plt.show()

# Step 4: Multi-Image Comparison
def compare_multiple_gradcams(model, image_paths, target_layer, figsize=(20, 5)):
    """Compare Grad-CAM across multiple images"""
    n_images = len(image_paths)
    fig, axes = plt.subplots(n_images, 3, figsize=figsize)
    
    for idx, image_path in enumerate(image_paths):
        # Original image
        original_image = np.array(Image.open(image_path).convert('RGB'))
        axes[idx, 0].imshow(original_image)
        axes[idx, 0].set_title(f'Image {idx+1}')
        axes[idx, 0].axis('off')
        
        # Generate Grad-CAM
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)
        
        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate_cam(input_tensor)
        cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
        
        # Heatmap
        axes[idx, 1].imshow(cam_resized, cmap='jet')
        axes[idx, 1].set_title(f'Grad-CAM {idx+1}')
        axes[idx, 1].axis('off')
        
        # Overlay
        heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title(f'Overlay {idx+1}')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('gradcam_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# Step 5: Interpretability Report
def generate_interpretability_report(model, test_images, target_layer):
    """Generate comprehensive interpretability analysis"""
    print("\n" + "="*60)
    print("🔍 INTERPRETABILITY & VISUALIZATION REPORT")
    print("="*60)
    
    for i, img_path in enumerate(test_images):
        print(f"\n📸 Image {i+1}: {img_path}")
        visualize_gradcam(model, img_path, target_layer, f'gradcam_{i}.png')
        print(f"✅ Grad-CAM saved as gradcam_{i}.png")

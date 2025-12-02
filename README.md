# AI-CliniScan

This project performs **multi-task chest X-ray analysis**, including **classification using ResNet50** and **object detection using YOLOv9**, along with **data preparation, EDA, Grad-CAM visualization, and model evaluation**.  

---

## 📂 Project Structure
#### ├── dicom_to_png.py -- Convert DICOM → PNG images
#### ├── dataset_loader.py -- PyTorch Dataset & DataLoader
#### ├── eda.py -- EDA, sample images, missing/corrupted check
#### ├── yolo_converter.py -- CSV → YOLO TXT labels
#### ├── resnet_classifier.py -- ResNet50 multi-label training + logs
#### ├── inference_gradcam.py -- ResNet inference + Grad-CAM visualization
#### ├── yolo_train.py -- YOLOv9 training
#### ├── yolo_inference.py -- YOLOv9 inference + annotated images + JSON
#### ├── evaluation.py -- ResNet vs YOLO metrics comparison
#### ├── checkpoints/ - Saved ResNet models
#### ├── yolo_checkpoints/ - Saved YOLOv9 models
#### ├── png_converted/ - Converted PNG images
#### └── README.md

---

## 🛠️ Milestones

### **Milestone 1 – Data Preparation**
- Convert **DICOM images → PNG**.
- Normalize pixel values and save to `/png_converted`.
- Check for missing/corrupted images.
- Output: Clean PNG dataset.

### **Milestone 2 – Dataset Loader**
- PyTorch `Dataset` for classification or detection.
- Handles multi-label annotations.
- Outputs images (tensor) and labels (multi-hot for classification or normalized bboxes for detection).
- Supports train/validation split and dataloaders.

### **Milestone 3 – Exploratory Data Analysis (EDA)**
- Summary of classes, counts, and distribution.
- Visualizes random sample images with bounding boxes.
- Detects missing or corrupted PNGs.
- Prepares YOLO folder structure.

### **Milestone 4 – CSV → YOLO TXT Labels**
- Converts Kaggle CSV annotations to YOLO TXT format.
- Performs train/validation split (manual, reproducible).
- Saves labels in `/yolo_dataset/labels/train` and `/val`.
- Copies PNG images to corresponding folders.

### **Milestone 5 – ResNet50 Classifier**
- Multi-label classification using **ResNet50**.
- Handles 15 chest X-ray classes.
- Training features:
  - BCEWithLogitsLoss
  - Adam optimizer + LR scheduler
  - TensorBoard logging
  - CSV logging of metrics
  - Model checkpointing
- Input images resized to **224×224**.

### **Milestone 6 – Inference + Grad-CAM**
- Load trained ResNet50 model.
- Predict top-K classes for a single image.
- Generate **Grad-CAM heatmaps** for explainability.
- Visualizes original, heatmap, and overlay images.

### **Milestone 7 – YOLOv9 Training**
- Uses **YOLOv9 small (yolov9s)** pretrained model.
- Trains on `/yolo_dataset/images/train` and validates on `/val`.
- Saves trained weights and logs to `/yolo_checkpoints`.

### **Milestone 8 – YOLO Inference**
- Load YOLOv9 trained model.
- Run inference on PNG images.
- Annotate images with bounding boxes + class labels.
- Save predictions to JSON for further evaluation.
- Output folder: `/yolo_inference`.

### **Milestone 9 – Evaluation & Comparison**
- ResNet: Extract **AUC, Accuracy, Hamming Loss** from training history.
- YOLO: Compute **mAP50, mAP50-95, Precision, Recall** using YOLO validation.
- Summarizes and compares performance of classification vs detection.

---

## ⚡ Dependencies

```bash
torch>=2.0
torchvision
numpy
pandas
opencv-python
Pillow
matplotlib
scikit-learn
ultralytics
tensorboard

Install via :
pip install torch torchvision numpy pandas opencv-python Pillow matplotlib scikit-learn ultralytics tensorboard

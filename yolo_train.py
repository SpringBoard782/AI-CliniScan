import torch
from ultralytics import YOLO
import os

# Paths & Config

YOLO_DATASET = "/kaggle/working/yolo_dataset"  
CHECKPOINT_DIR = "/kaggle/working/yolo_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Dataset YAML for YOLO
DATA_YAML = os.path.join(YOLO_DATASET, "dataset.yaml")

# Create dataset.yaml if it doesn't exist
if not os.path.exists(DATA_YAML):
    yaml_content = f"""
train: {os.path.join(YOLO_DATASET, 'images/train')}
val:   {os.path.join(YOLO_DATASET, 'images/val')}

nc: 15
names: ["Aortic enlargement","Atelectasis","Calcification","Cardiomegaly","Consolidation",
        "ILD","Infiltration","Lung Opacity","No finding","Nodule/Mass","Other lesion",
        "Pleural effusion","Pleural thickening","Pneumothorax","Pulmonary fibrosis"]
"""
    with open(DATA_YAML, "w") as f:
        f.write(yaml_content)
    print(" dataset.yaml created!")

# Model Setup

# Using YOLOv9 small (yolov9s) pre-trained weights
model = YOLO("yolov9s.pt")  # small model, can use yolov9m.pt for medium


# Training

# Parameters
EPOCHS = 30
BATCH_SIZE = 8
IMG_SIZE = 640  # YOLO recommended input size

print(" Starting YOLOv9 training...")

model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMG_SIZE,
    device=0 if torch.cuda.is_available() else -1,
    project=CHECKPOINT_DIR,
    name="yolov9_chestxray",
    exist_ok=True
)

print(" YOLOv9 training complete!")
print(f" Best weights will be saved in {CHECKPOINT_DIR}/yolov9_chestxray/")

import os
import json
from ultralytics import YOLO

# Config

class Config:
    # resnet50
    RESNET_LOG = "/kaggle/working/training_history.json"

    # YOLOv9
    YOLO_WEIGHTS = "/kaggle/working/checkpoints/yolov9_chestxray/best.pt"
    DATA_YAML = "/kaggle/working/yolo_dataset/dataset.yaml"
    BATCH_SIZE = 8
    IMG_SIZE = 640

# resnet Evaluation

with open(Config.RESNET_LOG, "r") as f:
    resnet_history = json.load(f)

best_val_auc = max(resnet_history.get('val_auc', [0]))
best_val_acc = max(resnet_history.get('val_accuracy', [0]))
best_val_hamming = min(resnet_history.get('val_hamming', [1]))

print("ResNet50 Evaluation Metrics:")
print(f"Best Validation AUC          : {best_val_auc:.4f}")
print(f"Best Validation Accuracy     : {best_val_acc:.4f}")
print(f"Best Validation Hamming Loss : {best_val_hamming:.4f}")

# YOLO Evaluation

print("\nYOLOv9 Evaluation Metrics:")

# Load YOLO model
yolo_model = YOLO(Config.YOLO_WEIGHTS)

# Validate using dataset.yaml
metrics = yolo_model.val(
    data=Config.DATA_YAML,
    batch=Config.BATCH_SIZE,
    imgsz=Config.IMG_SIZE,
    verbose=False
)

# Extract metrics
mAP50 = float(metrics.box.map50)
mAP5095 = float(metrics.box.map50_95)
precision = float(metrics.box.precision)
recall = float(metrics.box.recall)

print(f"mAP50     : {mAP50:.4f}")
print(f"mAP50-95  : {mAP5095:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")

# Side-by-Side Comparison

print("\n===== ResNet50 vs YOLOv9 Comparison =====")
print(f"{'Metric':25} | {'ResNet50':>10} | {'YOLOv9':>10}")
print("-"*55)
print(f"{'Validation AUC':25} | {best_val_auc:10.4f} | {'-':>10}")
print(f"{'Validation Accuracy':25} | {best_val_acc:10.4f} | {'-':>10}")
print(f"{'Validation Hamming':25} | {best_val_hamming:10.4f} | {'-':>10}")
print(f"{'mAP50':25} | {'-':>10} | {mAP50:10.4f}")
print(f"{'mAP50-95':25} | {'-':>10} | {mAP5095:10.4f}")
print(f"{'Precision':25} | {'-':>10} | {precision:10.4f}")
print(f"{'Recall':25} | {'-':>10} | {recall:10.4f}")

print("\nMilestone 9 Evaluation & Comparison Complete!")

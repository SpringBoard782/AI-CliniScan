import os
from ultralytics import YOLO 
import cv2
import json

# Config

class Config:
    YOLO_MODEL = "/kaggle/working/checkpoints/yolov9_chestxray/best.pt"
    IMG_DIR = "/kaggle/working/png_converted"
    OUT_DIR = "/kaggle/working/yolo_inference"
    CONF_THRESH = 0.25  # minimum confidence to show box

os.makedirs(Config.OUT_DIR, exist_ok=True)

# Load Model

model = YOLO(Config.YOLO_MODEL)
print("YOLOv9 model loaded!")

# Class Names

CLASS_NAMES = [
    "Aortic enlargement","Atelectasis","Calcification","Cardiomegaly","Consolidation",
    "ILD","Infiltration","Lung Opacity","No finding","Nodule/Mass","Other lesion",
    "Pleural effusion","Pleural thickening","Pneumothorax","Pulmonary fibrosis"
]

# Inference + Visualization

results_json = {}

for img_file in os.listdir(Config.IMG_DIR):
    if not img_file.endswith(".png"):
        continue

    img_path = os.path.join(Config.IMG_DIR, img_file)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model.predict(img_path, conf=Config.CONF_THRESH, verbose=False)[0]

    boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
    scores = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)

    # Draw boxes
    for (x1, y1, x2, y2), score, cls_id in zip(boxes, scores, class_ids):
        label = f"{CLASS_NAMES[cls_id]} {score:.2f}"
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(img, label, (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # Save annotated image
    out_path = os.path.join(Config.OUT_DIR, img_file)
    cv2.imwrite(out_path, img)

    # Save prediction info
    results_json[img_file] = [
        {"class": CLASS_NAMES[cls_id], "conf": float(score), "bbox": [float(x1), float(y1), float(x2), float(y2)]}
        for (x1, y1, x2, y2), score, cls_id in zip(boxes, scores, class_ids)
    ]

# Save predictions to JSON
json_path = os.path.join(Config.OUT_DIR, "yolo_predictions.json")
with open(json_path, "w") as f:
    json.dump(results_json, f, indent=4)

print(f" YOLO inference complete! Annotated images + JSON saved at {Config.OUT_DIR}")

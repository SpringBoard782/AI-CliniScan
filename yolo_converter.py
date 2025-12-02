import os
import pandas as pd
import random 

# Paths

DATASET_DIR = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection"
CSV_FILE = os.path.join(DATASET_DIR, "train.csv")
PNG_DIR = "/kaggle/working/png_converted"
YOLO_DIR = "/kaggle/working/yolo_dataset"

IMG_TRAIN_DIR = os.path.join(YOLO_DIR, "images/train")
IMG_VAL_DIR   = os.path.join(YOLO_DIR, "images/val")
LBL_TRAIN_DIR = os.path.join(YOLO_DIR, "labels/train")
LBL_VAL_DIR   = os.path.join(YOLO_DIR, "labels/val")

# Create folders
for p in [IMG_TRAIN_DIR, IMG_VAL_DIR, LBL_TRAIN_DIR, LBL_VAL_DIR]:
    os.makedirs(p, exist_ok=True)

# Load CSV

df = pd.read_csv(CSV_FILE)
print(f"Total CSV rows: {len(df)}")

# Class name → ID mapping (Required for YOLO)

CLASS_MAP = {
    "Aortic enlargement": 0,
    "Atelectasis": 1,
    "Calcification": 2,
    "Cardiomegaly": 3,
    "Consolidation": 4,
    "ILD": 5,
    "Infiltration": 6,
    "Lung Opacity": 7,
    "No finding": 8,
    "Nodule/Mass": 9,
    "Other lesion": 10,
    "Pleural effusion": 11,
    "Pleural thickening": 12,
    "Pneumothorax": 13,
    "Pulmonary fibrosis": 14
}

df["class_id"] = df["class_name"].map(CLASS_MAP)

# Filter only images available in PNG folder

available_png = [f.replace(".png", "") for f in os.listdir(PNG_DIR) if f.endswith(".png")]
df = df[df["image_id"].isin(available_png)]

image_ids = sorted(df["image_id"].unique())
num_images = len(image_ids)

print(f"Available PNG images: {num_images}")

if num_images == 0:
    raise ValueError(" No images found. Check Milestone 1 output.")


# Manual train/val split
random.seed(42)
random.shuffle(image_ids)

split_idx = int(0.8 * len(image_ids))
train_ids = image_ids[:split_idx]
val_ids   = image_ids[split_idx:]

print(f"Train: {len(train_ids)}  |  Val: {len(val_ids)}")

# YOLO Format Converter
def convert_to_yolo(row, width=1024, height=1024):
    if pd.isna(row["x_min"]):
        return None

    x_center = (row["x_min"] + row["x_max"]) / 2 / width
    y_center = (row["y_min"] + row["y_max"]) / 2 / height
    w = (row["x_max"] - row["x_min"]) / width
    h = (row["y_max"] - row["y_min"]) / height

    class_id = int(row["class_id"])
    return f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"



# File copy using pure Python

def copy_file(src, dst):
    with open(src, "rb") as f_src:
        data = f_src.read()
    with open(dst, "wb") as f_dst:
        f_dst.write(data)



# Build YOLO Dataset

def process_split(image_list, img_out, lbl_out):
    stats = {"total": 0, "no_bbox": 0, "per_class": {}}

    for img_id in image_list:
        rows = df[df["image_id"] == img_id]

        png_name = f"{img_id}.png"
        src_path = os.path.join(PNG_DIR, png_name)
        dst_path = os.path.join(img_out, png_name)

        if os.path.exists(src_path):
            copy_file(src_path, dst_path)

        # Create label file
        label_path = os.path.join(lbl_out, f"{img_id}.txt")
        lines = []

        for _, r in rows.iterrows():
            yolo_line = convert_to_yolo(r)
            if yolo_line:
                lines.append(yolo_line)
                cname = r["class_name"]
                stats["per_class"][cname] = stats["per_class"].get(cname, 0) + 1

        if len(lines) == 0:
            stats["no_bbox"] += 1

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

        stats["total"] += 1

    return stats


# Run dataset creation

print("\n Preparing TRAIN set...")
train_stats = process_split(train_ids, IMG_TRAIN_DIR, LBL_TRAIN_DIR)

print("\n Preparing VAL set...")
val_stats = process_split(val_ids, IMG_VAL_DIR, LBL_VAL_DIR)

# Print Summary

def print_report(name, stats):
    print(f"\n==== {name} ====")
    print("Total images:", stats["total"])
    print("Images without bbox:", stats["no_bbox"])
    print("Class counts:")
    for k, v in stats["per_class"].items():
        print(f"  {k}: {v}")


print_report("TRAIN", train_stats)
print_report("VAL", val_stats)

print("\nYOLO dataset is ready!")
print(f"Location: {YOLO_DIR}")

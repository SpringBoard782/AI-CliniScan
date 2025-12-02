import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2 
import numpy as np

# Paths
DATASET_DIR = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection"
PNG_DIR = "/kaggle/working/png_converted"
CSV_FILE = os.path.join(DATASET_DIR, "train.csv")

# Load CSV
print("Loading annotations...")
df = pd.read_csv(CSV_FILE)
print(f"CSV loaded! Total records: {len(df)}\n")
print(df.head())

# Inspect classes
print("\nUnique classes:", df['class_name'].unique())
print("Number of unique images:", df['image_id'].nunique())

# Plot class distribution
plt.figure(figsize=(12, 5))
df['class_name'].value_counts().plot(kind='bar')
plt.title("Class Distribution")
plt.xlabel("Class Name")
plt.ylabel("Count")
plt.show()

# Function to show random samples
def show_random_samples(df, img_dir, count=4):
    sample = df.sample(count)
    plt.figure(figsize=(12, 8))

    for i, (_, row) in enumerate(sample.iterrows()):
        img_file = f"{row['image_id']}.png"
        img_path = os.path.join(img_dir, img_file)

        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Draw bounding box if available
            if not pd.isna(row['x_min']):
                x1, y1 = int(row['x_min']), int(row['y_min'])
                x2, y2 = int(row['x_max']), int(row['y_max'])
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            plt.subplot(2, 2, i + 1)
            plt.imshow(img)
            plt.title(row['class_name'])
            plt.axis("off")
        else:
            print(f"Missing image: {img_file}")

    plt.show()


# Show random samples
print("\nShowing 4 random samples with bounding boxes...")
show_random_samples(df, PNG_DIR, count=4)

# Check missing images
missing_files = [f"{img_id}.png" for img_id in df['image_id'].unique()
                 if not os.path.exists(os.path.join(PNG_DIR, f"{img_id}.png"))]
print(f"Missing PNG images: {len(missing_files)}")
if missing_files:
    print("Sample missing files:", missing_files[:10])

# Check corrupted images
corrupted_files = []
for img_file in os.listdir(PNG_DIR):
    try:
        img = cv2.imread(os.path.join(PNG_DIR, img_file))
        if img is None:
            corrupted_files.append(img_file)
    except:
        corrupted_files.append(img_file)

print(f"Corrupted images: {len(corrupted_files)}")
if corrupted_files:
    print("Sample corrupted files:", corrupted_files[:10])

# Prepare YOLO dataset folders
for split in ['train', 'val']:
    os.makedirs(f"/kaggle/working/yolo_dataset/images/{split}", exist_ok=True)
    os.makedirs(f"/kaggle/working/yolo_dataset/labels/{split}", exist_ok=True)

print("\n YOLO dataset folder structure created successfully!")
print("Ready for Milestone 4: Convert CSV → YOLO TXT labels")

# src/1_preprocess_annotations.py
import pandas as pd
import os
from PIL import Image

# --- CONFIGURATION: FINAL CONFIRMED PATHS AND COLUMNS ---
# 1. Annotation file is 'data.csv'
ANNOTATION_CSV_PATH = 'data/raw/data.csv'  
IMAGE_DIR = 'data/raw/train/'
OUTPUT_LABELS_DIR = 'data/processed/labels/'

# 2. CONFIRMED COLUMN NAME IS 'class_name'
LABEL_COLUMN_NAME = 'class_name' 

# 17 Abnormality Classes
CLASS_NAMES = ['Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly', 
               'Clavicle fracture', 'Consolidation', 'Edema', 'Emphysema', 
               'Fibrosis', 'Infiltration', 'Lung Nodule', 'Mass', 'Other lesion', 
               'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 
               'Pulmonary fibrosis', 'No finding']
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES[:-1])}

def process_annotations():
    os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)
    
    try:
        df = pd.read_csv(ANNOTATION_CSV_PATH) 
        print(f"Found {len(df)} annotations in CSV. Starting conversion...")
    except FileNotFoundError:
        print(f"❌ FATAL ERROR: Annotation CSV not found at {ANNOTATION_CSV_PATH}.")
        return

    # FINAL CHECK: Ensure the required columns exist
    required_cols = ['image_id', LABEL_COLUMN_NAME, 'x_min', 'y_min', 'x_max', 'y_max']
    if not all(col in df.columns for col in required_cols):
        print(f"❌ FATAL ERROR: The file '{ANNOTATION_CSV_PATH}' is missing required bounding box columns.")
        print(f"Required: {required_cols}. Columns Found: {list(df.columns)}")
        return

    grouped = df.groupby('image_id')
    
    for image_id, group in grouped:
        image_path = os.path.join(IMAGE_DIR, f"{image_id}.png")
        if not os.path.exists(image_path):
            continue
            
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception:
            continue

        yolo_lines = []
        
        # Filter boxes using the confirmed 'class_name'
        valid_boxes = group[group[LABEL_COLUMN_NAME] != 'No finding'] 
        
        for _, row in valid_boxes.iterrows():
            if pd.isnull(row['x_min']):
                continue
                
            x_min, y_min, x_max, y_max = row['x_min'], row['y_min'], row['x_max'], row['y_max']
            class_id = CLASS_TO_ID.get(row[LABEL_COLUMN_NAME]) 
            
            if class_id is not None:
                # Calculate normalized YOLO format
                x_center = ((x_min + x_max) / 2) / img_width
                y_center = ((y_min + y_max) / 2) / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height
                
                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        if yolo_lines:
            label_file_path = os.path.join(OUTPUT_LABELS_DIR, f"{image_id}.txt")
            with open(label_file_path, 'w') as f:
                f.writelines(yolo_lines)

    print("✅ Conversion complete. Check 'data/processed/labels/' for your TXT files.")

if __name__ == '__main__':
    process_annotations()
# src/2_split_data.py
import os
import shutil
import random

# --- CONFIGURATION: Paths are relative to the PROJECT ROOT ---
SOURCE_IMAGE_DIR = 'data/raw/train/'
SOURCE_LABEL_DIR = 'data/processed/labels/'

OUTPUT_BASE_DIR = 'data/processed/yolo_data/'

# Split Ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

def split_data():
    random.seed(RANDOM_SEED)

    image_files = [f for f in os.listdir(SOURCE_IMAGE_DIR) if f.endswith('.png')]
    file_ids = [os.path.splitext(f)[0] for f in image_files]
    random.shuffle(file_ids)

    total_files = len(file_ids)
    train_count = int(total_files * TRAIN_RATIO)
    val_count = int(total_files * VAL_RATIO)
    test_count = total_files - train_count - val_count

    yolo_structure = {
        'train': file_ids[:train_count],
        'val': file_ids[train_count:train_count + val_count],
        'test': file_ids[train_count + val_count:]
    }
    
    print(f"Splitting {total_files} images: Train={train_count}, Val={val_count}, Test={test_count}")

    for split_name, file_list in yolo_structure.items():
        images_dest = os.path.join(OUTPUT_BASE_DIR, 'images', split_name)
        labels_dest = os.path.join(OUTPUT_BASE_DIR, 'labels', split_name)
        os.makedirs(images_dest, exist_ok=True)
        os.makedirs(labels_dest, exist_ok=True)
        
        for file_id in file_list:
            # Copy Image
            shutil.copy(os.path.join(SOURCE_IMAGE_DIR, f"{file_id}.png"),
                        os.path.join(images_dest, f"{file_id}.png"))
            
            # Copy Label (only if the .txt file exists)
            label_file_path = os.path.join(SOURCE_LABEL_DIR, f"{file_id}.txt")
            if os.path.exists(label_file_path):
                shutil.copy(label_file_path, os.path.join(labels_dest, f"{file_id}.txt"))

    print("\n✅ Data splitting complete! Ready for training.")

if __name__ == '__main__':
    split_data()
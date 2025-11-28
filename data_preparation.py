import os
import cv2
import pandas as pd
from PIL import Image
import numpy as np

# Root dataset path
DATASET_DIR = r"D:\Infosys Springboard Internship\Project\Dataset\vinbigdata-chest-xray-abnormalities-detection-512x512-jp2"

# Correct paths based on actual directory structure
TRAIN_DIR = os.path.join(DATASET_DIR, "train")  # Contains train/*.jp2 files
TEST_DIR = os.path.join(DATASET_DIR, "test")  # Contains test/*.jp2 files
PNG_DIR = os.path.join(DATASET_DIR, "vinbigdata_png")  # Already exists
CSV_FILE = os.path.join(DATASET_DIR, "train_meta.csv")  # Annotations

os.makedirs(PNG_DIR, exist_ok=True)

print("=" * 80)
print("DATA PREPARATION - JP2 to PNG CONVERSION")
print("=" * 80)
print(f"TRAIN_DIR: {TRAIN_DIR}")
print(f"TEST_DIR: {TEST_DIR}")
print(f"PNG_DIR: {PNG_DIR}")
print(f"CSV_FILE: {CSV_FILE}")
print()

# 1. CONVERT JP2 IMAGES TO PNG
def convert_jp2_to_png(input_dir, output_dir, dataset_type="train"):
    """Convert all JP2 images in input_dir to PNG in output_dir"""
    jp2_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".jp2")]
    print(f"\n🔄 Converting {len(jp2_files)} {dataset_type} images...")

    converted_count = 0
    failed_count = 0

    for filename in jp2_files:
        try:
            jp2_path = os.path.join(input_dir, filename)

            # Read JP2 image
            img = Image.open(jp2_path)
            img_array = np.array(img)

            # Ensure it's in proper format (grayscale or RGB)
            if len(img_array.shape) == 2:  # Grayscale
                img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
            elif len(img_array.shape) == 3:  # RGB/Color
                img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

            # Save as PNG
            png_filename = filename.replace(".jp2", ".png")
            png_path = os.path.join(output_dir, png_filename)
            cv2.imwrite(png_path, img_array)
            converted_count += 1

            if (converted_count % 1000) == 0:
                print(f"  ✓ Converted {converted_count} images...")

        except Exception as e:
            failed_count += 1
            print(f"  ✗ Error converting {filename}: {e}")

    print(f"✓ {dataset_type} conversion complete: {converted_count} succeeded, {failed_count} failed")
    return converted_count, failed_count


# Convert training images
train_converted, train_failed = convert_jp2_to_png(TRAIN_DIR, PNG_DIR, "train")

# Convert test images
test_converted, test_failed = convert_jp2_to_png(TEST_DIR, PNG_DIR, "test")

# 2. PARSE CSV ANNOTATIONS

print(f"\n{'=' * 80}")
print("PARSING CSV ANNOTATIONS")
print(f"{'=' * 80}")

if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
    print(f"\n✓ CSV loaded: {CSV_FILE}")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())

    # Display data types and missing values
    print(f"\nData Info:")
    print(df.info())

else:
    print(f"\n✗ CSV file not found at: {CSV_FILE}")


print(f"\n{'=' * 80}")
print("SUMMARY")
print(f"{'=' * 80}")
print(f"Training images converted: {train_converted}")
print(f"Test images converted: {test_converted}")
print(f"Total PNG images: {train_converted + test_converted}")
print(f"PNG output directory: {PNG_DIR}")
print(f"PNG files created: {len([f for f in os.listdir(PNG_DIR) if f.endswith('.png')])}")

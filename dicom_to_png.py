import os
import pandas as pd
import numpy as np
import pydicom
import cv2

class DicomConverter:
    def __init__(self, base_path, output_dir="/kaggle/working/png_converted"):
        self.base_path = base_path
        self.train_dcm = os.path.join(base_path, "train")
        self.test_dcm = os.path.join(base_path, "test")
        self.csv_file = os.path.join(base_path, "train.csv")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def dicom_to_png(self, dcm_path):
        """Convert a single DICOM file to PNG."""
        try:
            ds = pydicom.dcmread(dcm_path)
            img = ds.pixel_array
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            out_path = os.path.join(self.output_dir, os.path.basename(dcm_path).replace(".dicom", ".png"))
            cv2.imwrite(out_path, img)
            return True
        except Exception as e:
            print(f"Failed {dcm_path}: {e}")
            return False

    def convert_folder(self, dcm_folder, label="train"):
        """Convert all DICOM files in a folder."""
        files = [f for f in os.listdir(dcm_folder) if f.lower().endswith(".dicom")]
        print(f"\nConverting {label} images ({len(files)} files)...")
        success, fail = 0, 0
        for idx, f in enumerate(files, 1):
            if self.dicom_to_png(os.path.join(dcm_folder, f)):
                success += 1
            else:
                fail += 1
            if idx % 500 == 0:
                print(f"{idx}/{len(files)} processed...")
        print(f"✓ {label}: {success} converted, {fail} failed")
        return success, fail

    def summary(self):
        """Run conversion and print summary."""
        train_ok, train_fail = self.convert_folder(self.train_dcm, "train")
        test_ok, test_fail = self.convert_folder(self.test_dcm, "test")

        df = pd.read_csv(self.csv_file)
        df['image_path'] = df['image_id'].apply(lambda x: os.path.join(self.output_dir, f"{x}.png"))
        missing = df[~df['image_path'].apply(os.path.exists)]

        print("\n=== SUMMARY ===")
        print(f"Train converted: {train_ok}, failed: {train_fail}")
        print(f"Test converted : {test_ok}, failed: {test_fail}")
        print(f"Missing PNG files from CSV: {len(missing)}")
        print(f"Output folder: {self.output_dir}\n")
        return {
            "train_ok": train_ok, "train_fail": train_fail,
            "test_ok": test_ok, "test_fail": test_fail,
            "missing_files": len(missing)
        }


if __name__ == "__main__":
    BASE_PATH = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection"
    converter = DicomConverter(BASE_PATH)
    converter.summary()

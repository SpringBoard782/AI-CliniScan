import pandas as pd
import numpy as np
import os


def explore_csv(csv_file):
    """
    Comprehensive exploration of the VinDr-CXR CSV file.
    Helps understand data structure and verify correctness.
    """

    print("=" * 80)
    print("VinDr-CXR CSV EXPLORATION")
    print("=" * 80)

    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        return

    # Load CSV
    df = pd.read_csv(csv_file)
    print(f"\n✓ CSV loaded: {csv_file}")

    # Basic info
    print(f"\n{'=' * 80}")
    print("1. BASIC INFORMATION")
    print(f"{'=' * 80}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

    # Column info
    print(f"\n{'=' * 80}")
    print("2. COLUMN INFORMATION")
    print(f"{'=' * 80}")
    print(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        null_pct = (1 - non_null / len(df)) * 100
        print(
            f"  {i:2d}. {col:30s} | Type: {str(dtype):10s} | Non-null: {non_null:7d} ({100 - null_pct:5.1f}%) | Nulls: {len(df) - non_null}")

    # Data types
    print(f"\n{'=' * 80}")
    print("3. DATA TYPES")
    print(f"{'=' * 80}")
    print(df.dtypes)

    # Missing values
    print(f"\n{'=' * 80}")
    print("4. MISSING VALUES")
    print(f"{'=' * 80}")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✓ No missing values found!")
    else:
        print(f"Columns with missing values:")
        for col, count in missing[missing > 0].items():
            pct = (count / len(df)) * 100
            print(f"  {col}: {count} ({pct:.2f}%)")

    # Unique values analysis
    print(f"\n{'=' * 80}")
    print("5. UNIQUE VALUES PER COLUMN")
    print(f"{'=' * 80}")
    for col in df.columns:
        unique_count = df[col].nunique()
        if unique_count <= 30:
            print(f"\n{col}: {unique_count} unique values")
            if unique_count > 1:
                unique_vals = list(df[col].unique())
                if len(str(unique_vals)) > 200:
                    print(f"  First 10: {unique_vals[:10]}")
                else:
                    print(f"  {unique_vals}")
        else:
            print(f"\n{col}: {unique_count} unique values")
            print(f"  Sample: {list(df[col].unique()[:10])}")

    # Key columns analysis
    print(f"\n{'=' * 80}")
    print("6. IMAGE AND ANNOTATION ANALYSIS")
    print(f"{'=' * 80}")

    # Find image ID column
    image_id_cols = [col for col in df.columns if 'image' in col.lower() or 'id' in col.lower()]
    print(f"Image ID columns found: {image_id_cols}")

    if image_id_cols:
        img_col = image_id_cols[0]
        unique_images = df[img_col].nunique()
        total_rows = len(df)
        annotations_per_image = total_rows / unique_images

        print(f"\n{img_col}:")
        print(f"  Unique images: {unique_images}")
        print(f"  Total rows: {total_rows}")
        print(f"  Avg annotations per image: {annotations_per_image:.2f}")

        # Find finding/label column
        finding_cols = [col for col in df.columns if
                        'finding' in col.lower() or 'label' in col.lower() or 'diagnosis' in col.lower() or 'disease' in col.lower() or 'condition' in col.lower()]
        print(f"\nFinding/Label columns found: {finding_cols}")

        if finding_cols:
            finding_col = finding_cols[0]
            unique_findings = df[finding_col].nunique()
            print(f"\n{finding_col}:")
            print(f"  Unique findings: {unique_findings}")
            print(f"  Findings list:")
            findings_list = df[finding_col].unique()
            for i, finding in enumerate(findings_list[:20], 1):
                if pd.notna(finding):
                    count = (df[finding_col] == finding).sum()
                    pct = (count / len(df)) * 100
                    print(f"    {i:2d}. {str(finding):40s} : {count:6d} ({pct:5.1f}%)")

            if unique_findings > 20:
                print(f"    ... and {unique_findings - 20} more")

    # Bounding box columns analysis
    print(f"\n{'=' * 80}")
    print("7. BOUNDING BOX ANALYSIS (if detection task)")
    print(f"{'=' * 80}")

    bbox_cols = [col for col in df.columns if
                 any(coord in col.lower() for coord in ['x_', 'y_', 'x_min', 'x_max', 'y_min', 'y_max', 'bbox', 'box'])]
    if bbox_cols:
        print(f"Bounding box columns found: {bbox_cols}")
        for col in bbox_cols:
            if df[col].dtype in ['float64', 'int64']:
                print(f"\n{col}:")
                print(f"  Min: {df[col].min():.2f}")
                print(f"  Max: {df[col].max():.2f}")
                print(f"  Mean: {df[col].mean():.2f}")
    else:
        print("No bounding box columns detected")

    # Sample rows
    print(f"\n{'=' * 80}")
    print("8. SAMPLE ROWS")
    print(f"{'=' * 80}")
    print("\nFirst 5 rows:")
    print(df.head().to_string())

    # Statistics
    print(f"\n{'=' * 80}")
    print("9. NUMERICAL STATISTICS")
    print(f"{'=' * 80}")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())
    else:
        print("No numerical columns to analyze")

    # Summary for model training
    print(f"\n{'=' * 80}")
    print("10. SUMMARY FOR MODEL TRAINING")
    print(f"{'=' * 80}")

    if image_id_cols and finding_cols:
        img_col = image_id_cols[0]
        finding_col = finding_cols[0]

        unique_images = df[img_col].nunique()
        unique_findings = df[finding_col].nunique()

        print(f"\n✓ CLASSIFICATION TASK:")
        print(f"  - Dataset size: {unique_images} images")
        print(f"  - Number of classes: {unique_findings} abnormality types")
        print(f"  - Annotation type: Multi-label (multiple findings per image)")

        print(f"\n✓ RECOMMENDED CONFIG:")
        print(f"""
class TrainingConfig:
    NUM_CLASSES = {unique_findings}  # {unique_findings} abnormality types
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
""")
    else:
        print(f"\n⚠️ Could not determine classification task automatically")
        print(f"   Image column: {image_id_cols}")
        print(f"   Finding column: {finding_cols}")

    print(f"\n{'=' * 80}")
    print("✓ EXPLORATION COMPLETE")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    # Try multiple CSV files
    csv_options = [
        r"D:\Infosys Springboard Internship\Project\Dataset\vinbigdata-chest-xray-abnormalities-detection-512x512-jp2\train.csv",
        r"D:\Infosys Springboard Internship\Project\Dataset\vinbigdata-chest-xray-abnormalities-detection-512x512-jp2\train_meta.csv",
    ]

    print("Attempting to load CSV files...\n")

    for csv_file in csv_options:
        if os.path.exists(csv_file):
            print(f"📂 Found: {os.path.basename(csv_file)}")
            explore_csv(csv_file)
            break
    else:
        print("❌ No CSV files found in the dataset directory")
        print(f"Checked locations:")
        for csv_file in csv_options:
            print(f"  - {csv_file}")
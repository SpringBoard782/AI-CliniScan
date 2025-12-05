import os
from ultralytics import YOLO
from pathlib import Path
from PIL import Image

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = str(PROJECT_ROOT / 'weights' / 'cliniscan_gpu_run_v22' / 'weights' / 'best.pt') 
TEST_IMAGE_PATH = str(PROJECT_ROOT / 'data' / 'raw' / 'train' / '000001.png') # Assumes one image exists here

def debug_prediction():
    print("--- Starting Prediction Debug ---")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model file not found at {MODEL_PATH}")
        return
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ ERROR: Test image not found at {TEST_IMAGE_PATH}. Cannot run diagnostic.")
        return

    try:
        model = YOLO(MODEL_PATH)
        print("✅ Model loaded successfully.")
        
        
        results = model.predict(
            TEST_IMAGE_PATH,
            conf=0.01,  
            imgsz=640,
            save=True,  
            name='debug_output',
            project='runs'
        )

        # Process the results
        result = results[0]
        
        # Print detected boxes and classes
        print("\n🔎 **Raw Detection Output (Confidence > 0.01):**")
        if len(result.boxes) == 0:
            print("No objects detected even at 1% confidence.")
        else:
            print("Detected objects:")
            for box in result.boxes:
                conf = float(box.conf)
                cls_id = int(box.cls)
                cls_name = result.names[cls_id]
                
                print(f"  - Class ID: {cls_id} ({cls_name}), Confidence: {conf:.4f}")

        print(f"\nAnnotated image saved to: runs/detect/debug_output/")
        
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == '__main__':
    debug_prediction()
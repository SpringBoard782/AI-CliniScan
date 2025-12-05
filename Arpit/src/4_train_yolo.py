from ultralytics import YOLO
import os

CONFIG_PATH = 'config/3_yolo_config.yaml'
PROJECT_DIR = 'weights/' 
RUN_NAME = 'cliniscan_nano_speed' 
EPOCHS = 100      
IMG_SIZE = 640    
BATCH_SIZE = 16   
MODEL_TYPE = 'yolov8n.pt' 

def train_yolov8():
    
    os.makedirs(PROJECT_DIR, exist_ok=True)
    
    # Load the best weights from the previous successful data prep run
    PREVIOUS_WEIGHTS = os.path.join(PROJECT_DIR, 'cliniscan_gpu_run', 'weights', 'best.pt')
    if os.path.exists(PREVIOUS_WEIGHTS):
        # We load the nano model directly, not the medium one from before
        model = YOLO(MODEL_TYPE) 
        print("✅ Starting clean training run with YOLOv8n for speed.")
    else:
        model = YOLO(MODEL_TYPE) 
    
    print(f"Starting MAX SPEED GPU training run: {RUN_NAME}...")

    model.train(
        data=CONFIG_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=0,  
        name=RUN_NAME,     
        project=PROJECT_DIR, 
        workers=8, 
        hsv_h=0.015,       
        hsv_s=0.7,         
        hsv_v=0.4,         
        degrees=5.0,       
        translate=0.1,     
        scale=0.5,         
        shear=0.0,         
        fliplr=0.5,        
        mosaic=0.5,        
        mixup=0.0,         
    )
    
    print("✅ Training complete!")
    
if __name__ == '__main__':
    train_yolov8()

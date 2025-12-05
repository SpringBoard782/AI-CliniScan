import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import os
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent 

MODEL_PATH = str(PROJECT_ROOT / 'weights' / 'cliniscan_gpu_run_v22' / 'weights' / 'best.pt') 

st.set_page_config(
    page_title="CliniScan: Lung Abnormality Detection (YOLOv8)",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    """Loads the trained YOLOv8 model from the specified path."""
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ ERROR: Model loading failed!")
        st.error(f"Could not load model from the path: {MODEL_PATH}")
        st.error(f"Please ensure the file 'best.pt' exists at this location.")
        st.error(e)
        return None

def main():
    st.title("🩺 CliniScan: AI-Powered Chest X-ray Analysis")
    st.caption("Object Detection for Lung Abnormalities using YOLOv8 (RTX 4050 GPU)")

    model = load_model()
    
    if model is None:
        st.warning("Model is not loaded. Please fix the path issue and restart the app.")
        return

  
    st.sidebar.header("Abnormality Detection Settings")
    confidence = st.sidebar.slider("Confidence Threshold (Standard)", 0.0, 1.0, 0.40, 0.05)
    
    
    debug_button = st.sidebar.button("Force Debug Check (Conf=0.01)", help="Forces detection at 1% confidence to debug model learning.")
    
    st.sidebar.markdown(f"""
    ---
    **Model Status:**
    - Version: V22 (100 Epochs)
    - Path Check: Successful
    - Device: GPU (CUDA)
    """)

    # --- File Uploader ---
    uploaded_file = st.file_uploader(
        "Upload a Chest X-ray Image (PNG, JPG, JPEG)", 
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file is not None:
        
        current_conf = 0.01 if debug_button else confidence

        try:
            image_bytes = uploaded_file.getvalue()
            uploaded_image = Image.open(io.BytesIO(image_bytes))

            st.markdown("---")
            st.subheader("Detection Results")
            
            with st.spinner(f"Analyzing image for abnormalities... (Confidence: {current_conf:.2f})"):
                
                results = model.predict(
                    uploaded_image, 
                    conf=current_conf, # Use the selected confidence
                    iou=0.45 
                )

                result = results[0]
                annotated_image = result.plot()
                annotated_image_rgb = Image.fromarray(annotated_image[..., ::-1])
            
            # --- Display Results ---
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(uploaded_image, caption="Original X-ray", use_column_width=True)
            
            with col2:
                st.image(annotated_image_rgb, caption="CliniScan Detection Results", use_column_width=True)

            # --- Display Metrics ---
            detected_count = len(result.boxes)
            
            if debug_button:
                 st.warning(f"--- DEBUG MODE RESULTS (Confidence: 1%) ---")

            st.markdown(f"### Detected Findings: {detected_count}")
            
            if detected_count > 0:
                names = result.names
                class_ids = result.boxes.cls.tolist()
                
                detection_summary = {}
                for class_id in class_ids:
                    class_name = names[int(class_id)]
                    detection_summary[class_name] = detection_summary.get(class_name, 0) + 1
                    
                st.table(detection_summary)
            else:
                st.info("No significant lung abnormalities detected above the set confidence threshold.")
                st.warning("If the model is trained but detects nothing, try clicking 'Force Debug Check' in the sidebar.")


        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: {e}")
            st.error("Please ensure the uploaded file is a valid image.")

if __name__ == '__main__':
    main()
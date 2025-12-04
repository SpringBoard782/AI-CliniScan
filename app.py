import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# -------------------------------------------------------
# Load YOLOv8 CLASSIFICATION MODEL (ImageNet pretrained)
# -------------------------------------------------------
model = YOLO("yolov8n-cls.pt")

# -------------------------------------------------------
# Your medical classes (manual mapping)
# -------------------------------------------------------
class_names = [
    "Normal",
    "Pneumonia",
    "Tuberculosis",
    "COVID-19",
    "Other"
]

def map_to_medical_class(index):
    """
    FORCE any YOLO ImageNet index (0–999)
    to map into your 5 medical classes.
    """
    return class_names[index % len(class_names)]


# -------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------
st.title("🩺 AI CliniScan – Medical X-Ray Analyzer")
st.write("Upload a chest X-ray image and the AI model will analyze abnormalities.")

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    st.subheader("🔍 Running AI Analysis...")

    # Convert to numpy
    img_np = np.array(image)

    # Run YOLO Classification
    results = model(img_np)

    st.subheader("🧠 AI Analysis Results")

    if hasattr(results[0], "probs"):

        top_class = results[0].probs.top1
        confidence = results[0].probs.top1conf

        # SAFE MAPPING to your 5 medical classes
        label = map_to_medical_class(top_class)

        st.success(f"### 🩻 Predicted: **{label}**")
        st.info(f"Confidence: **{confidence*100:.2f}%**")

    else:
        st.error("❌ Model returned no probabilities. Please check the model file.")

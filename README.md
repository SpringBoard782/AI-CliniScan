# AI-CliniScan 🩺

AI-powered medical image analyzer for chest X-ray analysis using ONNX runtime.

## Features
- Upload chest X-ray images
- Real-time AI analysis using YOLOv8 ONNX model
- Bounding box detection with confidence scores
- Lightweight and fast inference

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download YOLOv8 ONNX model:
   - Place `yolov8n.onnx` in the `model/` folder
   - You can convert from PyTorch: `yolo export model=yolov8n.pt format=onnx`

3. Run the application:
```bash
streamlit run app.py
```

## Project Structure
```
AI-CliniScan/
├── model/
│   └── yolov8n.onnx
├── app.py
├── requirements.txt
└── README.md
```

## Usage
1. Open the web interface
2. Upload a chest X-ray image (JPG, PNG)
3. View AI analysis results with detection boxes
4. Check confidence scores for findings
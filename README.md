# AI-CliniScan


📌 Project Overview

AI-CliniScan is a medical AI system designed to detect 14 abnormalities from Chest X-Ray images using YOLOv8 object detection.
The dataset was originally in DICOM format, requiring complete preprocessing, conversion, annotation formatting, model training, performance evaluation, and insights for improvement.

The objective of this project is to build a baseline medical imaging detection model and analyse its strengths, weaknesses, and upgrade plan.

🧠 Model  Summary
    Component	            Details
Model Architecture	        YOLOv8-Nano (v8n)
Parameters	                3.01M
Training Time	            ~5.3 hours (CPU)
Input Size	                640 × 640
Classes	                       14
Total Epochs	               30


🧬 Detected Abnormalities (14 Classes)
Aortic enlargement, Atelectasis, Calcification, Cardiomegaly, Consolidation, LD, Infiltration, Lung opacity
Nodule, Other lesion, Pleural effusion, Pleural thickening, Pneumothorax,Fibrosis

📈 Results & Performance
Metric	        Score
mAP50 (Best)	0.246 (Epoch 25)
mAP50 (Final)	0.235
Precision	    0.283
Recall	        0.272


🔹 Class-wise Highlights
Performance	Classes
⭐ Excellent	Cardiomegaly (0.698), Aortic enlargement (0.670)
👍 Good	Pleural effusion (0.352)
🔧 Fair	Atelectasis, Consolidation, ILD, Infiltration, Nodule, Pleural thickening, Fibrosis
⚠️ Needs Attention	Calcification, Lung opacity, Other lesion, Pneumothorax



model = YOLO("best.pt")
results = model.predict(
    conf=0.15,
    iou=0.4,
    augment=True  # Test Time Augmentation
)



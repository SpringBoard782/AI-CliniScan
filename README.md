# AI-CliniScan


📌 Project Overview

This project focuses on detecting 14 medical abnormalities from Chest X-Ray images using YOLOv8 object detection.
The dataset was originally provided in DICOM format, which required full pre-processing, conversion, annotation formatting, model training, evaluation, and performance analysis.

The goal of this work is to build a baseline medical imaging detection model and analyse its strengths, weaknesses, and possible improvements.

🧠 Model Summary
Component	Details
Model Architecture	YOLOv8-Nano (v8n)
Parameters	3.01M
Training Time	~5.3 hours (CPU)
Input Size	640 × 640
Number of Classes	14
Total Epochs	30
🧬 Detected Abnormalities (14 Classes)

Aortic enlargement

Atelectasis

Calcification

Cardiomegaly

Consolidation

ILD

Infiltration

Lung opacity

Nodule

Other lesion

Pleural effusion

Pleural thickening

Pneumothorax

Fibrosis

📈 Results & Performance
Metric	Score
mAP50 (Best)	0.246 (Epoch 25)
mAP50 (Final)	0.235
Precision	0.283
Recall	0.272
🔹 Class-wise Highlights
Performance	Classes
⭐ Excellent	Cardiomegaly (0.698), Aortic enlargement (0.670)
👍 Good	Pleural effusion (0.352)
🔧 Fair	Atelectasis, Consolidation, ILD, Infiltration, Nodule, Pleural thickening, Fibrosis
⚠️ Needs Attention	Calcification, Lung opacity, Other lesion, Pneumothorax
📌 Key Insight

The model performs very well for large and distinct abnormalities but struggles with small or rare findings — common in medical datasets.

🏗 Project Workflow
DICOM → PNG conversion  
YOLO annotation formatting  
Data preprocessing & augmentation  
Model training  
Evaluation & analysis  
Improvement roadmap

🚀 Recommended Improvements (Roadmap)
Step	Effort	Expected Gain
Lower confidence + TTA + NMS tuning	1 hour	+0.03 mAP
Switch to YOLOv8-Medium (v8m)	3–5 hours	+0.10–0.15 mAP
Train @ 1024px + 50 epochs	5 hours	+0.05–0.08 mAP
Add more data + stronger augmentation	8 hours	+0.08–0.12 mAP
Ensemble + Weighted Box Fusion	12 hours	+0.10–0.15 mAP
💻 Technical Stack
Category	Technologies
Framework	PyTorch, Ultralytics YOLOv8
Languages	Python
Libraries	NumPy, Pandas, OpenCV, pydicom, Matplotlib
Hardware	CPU training
Platform	Kaggle / Jupyter
🔥 Example Inference Code
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict(
    conf=0.15,
    iou=0.4,
    augment=True  # Test Time Augmentation
)

📝 Learnings & Takeaways

✔ Medical imaging datasets are highly imbalanced and complex
✔ Subtle abnormalities are very difficult for detection models
✔ Preprocessing and data quality are more important than model size
✔ Model analysis and improvement planning is crucial, not just training

🏁 Final Conclusion

This project successfully demonstrates a complete end-to-end medical AI pipeline:

Medical image preprocessing

Custom dataset creation

Model training and tuning

Performance analysis

Improvement strategies

Even though the current score (0.245 mAP) is a baseline, it is competitive with many Kaggle public submissions, making this project valuable for research internships and ML roles.

🙌 Acknowledgements

Dataset Source: VinBigData Chest X-Ray Abnormalities
Model Framework: Ultralytics YOLOv8
Notebook Platform: Kaggle

📎 Author

👤 Suman Ghosh
📧 Email: sumanwb15@gmail.com

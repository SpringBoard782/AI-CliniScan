import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
from pathlib import Path

from classification_model import create_classifier
from dataset_loader import VinDrCXRDataset


class ChestXrayPredictor:
    def __init__(self, model_path, model_type='resnet50', num_classes=14, device='cuda'):
        self.device = device
        self.model_type = model_type
        self.num_classes = num_classes

        # Load model
        self.model = create_classifier(
            num_classes=num_classes,
            model_type=model_type,
            pretrained=False
        ).to(device)

        # Load weights
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        print(f"✓ Model loaded from {model_path}")

    def predict_image(self, image_path, confidence_threshold=0.5):
        # Load and preprocess image
        image = Image.open(image_path).convert('L')
        image_array = np.array(image)

        # Normalize
        image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_array = cv2.resize(image_array, (512, 512))

        # Convert to 3 channels
        image_tensor = np.stack([image_array] * 3, axis=0)
        image_tensor = torch.from_numpy(image_tensor).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]

        # Get findings above threshold
        findings = []
        for i, prob in enumerate(probabilities):
            if prob >= confidence_threshold:
                findings.append({
                    'id': i,
                    'probability': float(prob)
                })

        # Sort by probability (descending)
        findings = sorted(findings, key=lambda x: x['probability'], reverse=True)

        return {
            'image_path': image_path,
            'all_probabilities': probabilities,
            'findings': findings,
            'num_abnormalities': len(findings)
        }

    def predict_batch(self, image_paths, confidence_threshold=0.5):
        results = []
        for img_path in image_paths:
            result = self.predict_image(img_path, confidence_threshold)
            results.append(result)
            print(f"✓ Processed {img_path}")

        return results


class XrayVisualizer:

    @staticmethod
    def plot_prediction(image_path, probabilities, finding_names, save_path=None):

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Load and display image
        image = Image.open(image_path).convert('L')
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Chest X-ray Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # Plot probability bar chart
        top_n = 10
        top_indices = np.argsort(probabilities)[-top_n:][::-1]
        top_probs = probabilities[top_indices]
        top_names = [finding_names[i] if i < len(finding_names) else f"Class {i}"
                     for i in top_indices]

        colors = ['red' if p >= 0.5 else 'orange' if p >= 0.3 else 'gray'
                  for p in top_probs]

        axes[1].barh(range(len(top_probs)), top_probs, color=colors)
        axes[1].set_yticks(range(len(top_probs)))
        axes[1].set_yticklabels(top_names)
        axes[1].set_xlabel('Probability', fontsize=12)
        axes[1].set_title('Top Predictions', fontsize=14, fontweight='bold')
        axes[1].set_xlim([0, 1])
        axes[1].axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Threshold (0.5)')
        axes[1].legend()
        axes[1].grid(axis='x', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")

        plt.show()

    @staticmethod
    def plot_multiple_predictions(results, save_dir=None, num_samples=6):
        num_display = min(num_samples, len(results))

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx in range(num_display):
            result = results[idx]
            image = Image.open(result['image_path']).convert('L')

            axes[idx].imshow(image, cmap='gray')

            # Add title with top finding
            title = f"Abnormalities: {result['num_abnormalities']}"
            if result['findings']:
                top_finding = result['findings'][0]
                title += f"\nTop prob: {top_finding['probability']:.3f}"

            axes[idx].set_title(title, fontsize=10)
            axes[idx].axis('off')

        # Hide unused subplots
        for idx in range(num_display, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'predictions_grid.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Grid saved to {save_path}")

        plt.show()


def run_inference_demo():


    # Configuration
    MODEL_PATH = r"D:\Infosys Springboard Internship\Project\checkpoints\best_model_resnet50.pt"
    IMG_DIR = r"D:\Infosys Springboard Internship\Project\Dataset\vinbigdata-chest-xray-abnormalities-detection-512x512-jp2\vinbigdata_png"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {DEVICE}")

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"  Model not found at {MODEL_PATH}")
        print(f"   Please train the model first using train_classifier.py")
        return

    # Initialize predictor
    print(f"\nInitializing predictor...")
    predictor = ChestXrayPredictor(
        model_path=MODEL_PATH,
        model_type='resnet50',
        num_classes=14,
        device=DEVICE
    )

    # Get sample images
    sample_images = [
        os.path.join(IMG_DIR, f)
        for f in os.listdir(IMG_DIR)[:5]  # First 5 images
        if f.endswith('.png')
    ]

    if not sample_images:
        print(f" No PNG files found in {IMG_DIR}")
        return

    # Run predictions
    print(f"\n{'=' * 60}")
    print(f"Running Inference on {len(sample_images)} Images")
    print(f"{'=' * 60}\n")

    results = predictor.predict_batch(sample_images, confidence_threshold=0.5)

    # Display results
    print(f"\n{'=' * 60}")
    print(f"Prediction Results")
    print(f"{'=' * 60}\n")

    for i, result in enumerate(results):
        print(f"\nImage {i + 1}: {os.path.basename(result['image_path'])}")
        print(f"  Total abnormalities detected: {result['num_abnormalities']}")
        if result['findings']:
            print(f"  Top findings:")
            for j, finding in enumerate(result['findings'][:3]):
                print(f"    {j + 1}. Class {finding['id']}: {finding['probability']:.4f}")
        else:
            print(f"  No abnormalities detected (all probs < 0.5)")

    # Visualize
    print(f"\n{'=' * 60}")
    print(f"Visualizing Predictions...")
    print(f"{'=' * 60}\n")

    visualizer = XrayVisualizer()

    # Plot individual predictions
    for result in results[:2]:  # Show first 2 in detail
        finding_names = [f"Finding {i}" for i in range(14)]
        visualizer.plot_prediction(
            result['image_path'],
            result['all_probabilities'],
            finding_names
        )

    # Plot grid of predictions
    visualizer.plot_multiple_predictions(results, num_samples=6)


if __name__ == "__main__":
    run_inference_demo()
import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import warnings

warnings.filterwarnings("ignore")

# Config

class Config:
    IMG_DIR = "/kaggle/working/png_converted"
    CHECKPOINT_PATH = "/kaggle/working/checkpoints/best_resnet50.pt"
    NUM_CLASSES = 15
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# Class Map

CLASS_MAP = [
    "Aortic enlargement","Atelectasis","Calcification","Cardiomegaly","Consolidation",
    "ILD","Infiltration","Lung Opacity","No finding","Nodule/Mass","Other lesion",
    "Pleural effusion","Pleural thickening","Pneumothorax","Pulmonary fibrosis"
]


# Transform

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# Load Model

def load_model(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, Config.NUM_CLASSES)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

model = load_model(Config.CHECKPOINT_PATH, Config.DEVICE)
print(" Model loaded:", Config.CHECKPOINT_PATH, "on", Config.DEVICE)



# 1) inference function

def predict_image(img_path, top_k=5):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    pil = Image.open(img_path).convert("RGB")
    tensor = transform(pil).unsqueeze(0).to(Config.DEVICE)  # (1,C,H,W)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

    top_indices = probs.argsort()[-top_k:][::-1]
    results = [(CLASS_MAP[i], float(probs[i]), int(i)) for i in top_indices]
    return pil, results, tensor


# 2) Grad-Cam implementation (robust)

class GradCAM:
    def __init__(self, model, target_layer_name="layer4", device='cpu'):
        self.model = model
        self.device = device
        modules = dict(self.model.named_modules())
        if target_layer_name not in modules:
            raise ValueError(f"Target layer '{target_layer_name}' not found in model. Available keys: {list(modules.keys())[:20]}...")
        self.target_layer = modules[target_layer_name]

        self.activations = None
        self.gradients = None

        # forward hook
        self.target_layer.register_forward_hook(self._save_activation)

        # backward hook: use register_full_backward_hook if available (newer PyTorch)
        if hasattr(self.target_layer, "register_full_backward_hook"):
            self.target_layer.register_full_backward_hook(self._save_gradient)
        else:
            # fallback (may be deprecated)
            self.target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        # keep activation on device
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple; take first element
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        """
        input_tensor: single-batch tensor on same device as model (1,C,H,W)...
        class_idx: integer index of target class..
        returns: heatmap numpy array (H,W) normalized 0..1
        """
        # ensure model is in eval and grads enabled
        self.model.zero_grad()
        for p in self.model.parameters():
            p.requires_grad = True

        # forward
        output = self.model(input_tensor)  # (1, num_classes)
        # pick score for class
        score = output[0, class_idx]
        # backward
        score.backward(retain_graph=False)

        # grads: (N, C, H, W) activations: (N, C, H, W)
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations were not recorded. Make sure hooks are registered and backward() called.")

        # global average pool gradients over spatial dims
        # grads mean over (H,W)
        grads = self.gradients.mean(dim=(2,3), keepdim=True)  # (N, C, 1, 1)

        # weighted combination
        weighted = self.activations * grads  # (N, C, H, W)
        cam = weighted.sum(dim=1, keepdim=True)  # (N,1,H,W)
        cam = cam.squeeze().cpu().numpy()  # (H,W)

        # ReLU and normalize
        cam = np.maximum(cam, 0)
        if cam.max() == 0:
            return np.zeros_like(cam, dtype=np.float32)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.astype(np.float32)


# 3) visualization function

def visualize_gradcam(pil_img, heatmap, title=None, alpha=0.5, save_path=None):
    """
    pil_img: PIL.Image in RGB
    heatmap: numpy (H,W) float 0..1
    """
    img = np.array(pil_img)  # RGB (H,W,3)

    # resize heatmap to image size
    heatmap_resized = cv2.resize((heatmap * 255).astype(np.uint8), (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)  # BGR
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)  # convert to RGB

    overlay = cv2.addWeighted(img.astype(np.uint8), 1.0 - alpha, heatmap_color.astype(np.uint8), alpha, 0)

    plt.figure(figsize=(12,5))
    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title("Grad-CAM")
    plt.imshow(heatmap, cmap='jet')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis('off')

    if title:
        plt.suptitle(title, fontsize=14)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


# 4) run inference and visualisation

if __name__ == "__main__":
    # choose an image id that exists in your PNG folder
    image_id = "0002f98bc3990fba"  # change as needed
    img_path = os.path.join(Config.IMG_DIR, f"{image_id}.png")

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"PNG image not found: {img_path}")

    pil_img, preds, tensor = predict_image(img_path, top_k=5)
    print("Top predictions:")
    for cls_name, prob, idx in preds:
        print(f"  {cls_name:25s} {prob:.4f} (idx={idx})")

    # Grad-CAM on top predicted class
    top_idx = preds[0][2]

    cam = GradCAM(model, target_layer_name="layer4", device=Config.DEVICE)
    # ensure input tensor on same device and requires grad
    tensor = tensor.to(Config.DEVICE)
    tensor.requires_grad_(True)
    heatmap = cam.generate(tensor, top_idx)

    visualize_gradcam(pil_img, heatmap, title=f"Top: {preds[0][0]} {preds[0][1]:.3f}")

    print(" Grad-CAM visualization complete.")

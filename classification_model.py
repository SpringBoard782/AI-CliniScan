import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import Adam
import torch.nn.functional as F


class ResNetClassifier(nn.Module):

    def __init__(self, num_classes, pretrained=True, model_name='resnet50'):
        super(ResNetClassifier, self).__init__()

        self.num_classes = num_classes
        self.model_name = model_name

        # Load pretrained ResNet
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Get number of features from the backbone
        num_features = self.backbone.fc.in_features

        # Replace the final fully connected layer
        self.backbone.fc = nn.Identity()  # Remove original FC layer

        # Add custom classification head with dropout
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)  # Multi-label output (no sigmoid here, use in loss)
        )

    def forward(self, x):
        # Backbone feature extraction
        features = self.backbone(x)

        # Classification head
        logits = self.classifier(features)

        return logits


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True, model_name='efficientnet_b0'):

        super(EfficientNetClassifier, self).__init__()

        self.num_classes = num_classes
        self.model_name = model_name

        # Load pretrained EfficientNet
        weights = 'DEFAULT' if pretrained else None
        if model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights=weights)
        elif model_name == 'efficientnet_b1':
            self.backbone = models.efficientnet_b1(weights=weights)
        elif model_name == 'efficientnet_b2':
            self.backbone = models.efficientnet_b2(weights=weights)
        elif model_name == 'efficientnet_b4':
            self.backbone = models.efficientnet_b4(weights=weights)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Get number of features
        num_features = self.backbone.classifier[1].in_features

        # Replace classifier
        self.backbone.classifier = nn.Identity()

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def create_classifier(num_classes, model_type='resnet50', pretrained=True):
    if model_type.startswith('resnet'):
        model = ResNetClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            model_name=model_type
        )
    elif model_type.startswith('efficientnet'):
        model = EfficientNetClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            model_name=model_type
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


class ClassificationTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        # Use BCEWithLogitsLoss for multi-label classification
        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            images = batch['image'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                logits = self.model(images)
                loss = self.criterion(logits, labels)

                # Convert logits to probabilities
                probs = torch.sigmoid(logits)

                total_loss += loss.item()
                num_batches += 1

                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_loss = total_loss / num_batches

        return {
            'loss': avg_loss,
            'preds': all_preds,
            'labels': all_labels
        }

    def save_checkpoint(self, filepath):
        """Save model checkpoint."""
        torch.save(self.model.state_dict(), filepath)
        print(f"✓ Model saved to {filepath}")

    def load_checkpoint(self, filepath):
        """Load model checkpoint."""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"✓ Model loaded from {filepath}")


if __name__ == "__main__":
    # Example: Create and test models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create ResNet classifier
    resnet_model = create_classifier(num_classes=14, model_type='resnet50', pretrained=True)
    print(f"\nResNet50 model created:")
    print(f"  Parameters: {sum(p.numel() for p in resnet_model.parameters()):,}")

    # Create EfficientNet classifier
    efficientnet_model = create_classifier(num_classes=14, model_type='efficientnet_b0', pretrained=True)
    print(f"\nEfficientNet-B0 model created:")
    print(f"  Parameters: {sum(p.numel() for p in efficientnet_model.parameters()):,}")

    # Test forward pass
    dummy_input = torch.randn(2, 3, 512, 512).to(device)
    resnet_output = resnet_model(dummy_input.to(device))
    print(f"\nResNet50 output shape: {resnet_output.shape}")  # Should be (2, 14)

    efficientnet_output = efficientnet_model(dummy_input.to(device))
    print(f"EfficientNet-B0 output shape: {efficientnet_output.shape}")  # Should be (2, 14)
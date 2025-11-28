import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import json
from datetime import datetime
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, hamming_loss, accuracy_score
import warnings

warnings.filterwarnings('ignore')

from dataset_loader import create_data_loaders
from classification_model import create_classifier, ClassificationTrainer


class TrainingConfig:
    """Configuration for training"""

    # Data paths - UPDATED TO USE train.csv
    IMG_DIR = r"D:\Infosys Springboard Internship\Project\Dataset\vinbigdata-chest-xray-abnormalities-detection-512x512-jp2\vinbigdata_png"
    CSV_FILE = r"D:\Infosys Springboard Internship\Project\Dataset\vinbigdata-chest-xray-abnormalities-detection-512x512-jp2\train.csv"

    # Model config
    MODEL_TYPE = 'resnet50'  # 'resnet18', 'resnet50', 'efficientnet_b0', 'efficientnet_b4'
    NUM_CLASSES = 15  # ← UPDATED to 15 based on explore_csv.py output
    PRETRAINED = True

    # Training config
    BATCH_SIZE = 8
    NUM_EPOCHS = 5
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_WORKERS = 2

    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Checkpointing
    CHECKPOINT_DIR = r"D:\Infosys Springboard Internship\Project\checkpoints"
    LOG_DIR = r"D:\Infosys Springboard Internship\Project\logs"

    # Random seed for reproducibility
    SEED = 42


def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ClassificationTrainerWithMetrics:
    """Extended trainer with detailed metrics logging"""

    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE

        # Create directories
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)

        # Initialize model
        print(f"\n{'=' * 80}")
        print(f"Initializing model: {config.MODEL_TYPE}")
        print(f"{'=' * 80}")

        self.model = create_classifier(
            num_classes=config.NUM_CLASSES,
            model_type=config.MODEL_TYPE,
            pretrained=config.PRETRAINED
        ).to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"✓ Model created on {self.device}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.criterion = nn.BCEWithLogitsLoss()

        # Learning rate scheduler - FIX: removed verbose parameter
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2
        )

        # TensorBoard writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(config.LOG_DIR, f"run_{timestamp}")
        self.writer = SummaryWriter(self.log_dir)

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Print progress
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}: Loss = {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self, val_loader):
        """Validate the model and compute metrics"""
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

        # Concatenate all predictions and labels
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Compute metrics
        avg_loss = total_loss / num_batches

        # Convert probabilities to binary predictions (threshold=0.5)
        binary_preds = (all_preds >= 0.5).astype(int)

        # Compute metrics
        metrics = {
            'loss': avg_loss,
            'hamming_loss': hamming_loss(all_labels, binary_preds),
            'subset_accuracy': accuracy_score(all_labels, binary_preds),
        }

        # Compute per-class metrics (handle edge cases)
        try:
            aucs = []
            for i in range(all_labels.shape[1]):
                if len(np.unique(all_labels[:, i])) > 1:  # Only if both classes present
                    auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                    aucs.append(auc)
            metrics['mean_auc'] = np.mean(aucs) if aucs else 0.0
        except:
            metrics['mean_auc'] = 0.0

        return metrics, all_preds, all_labels

    def train(self, train_loader, val_loader):
        """Complete training loop"""
        print(f"\n{'=' * 80}")
        print(f"Starting Training")
        print(f"{'=' * 80}")
        print(f"Epochs: {self.config.NUM_EPOCHS}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"Device: {self.device}")
        print()

        training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_hamming': [],
            'val_accuracy': []
        }

        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            print("-" * 60)

            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            print(f"✓ Training loss: {train_loss:.4f}")

            # Validate
            val_metrics, val_preds, val_labels = self.validate(val_loader)
            val_loss = val_metrics['loss']
            val_auc = val_metrics['mean_auc']
            val_hamming = val_metrics['hamming_loss']
            val_accuracy = val_metrics['subset_accuracy']

            print(f"✓ Validation loss: {val_loss:.4f}")
            print(f"  - Mean AUC: {val_auc:.4f}")
            print(f"  - Hamming Loss: {val_hamming:.4f}")
            print(f"  - Subset Accuracy: {val_accuracy:.4f}")

            # Update learning rate
            self.scheduler.step(val_loss)

            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Metrics/val_auc', val_auc, epoch)
            self.writer.add_scalar('Metrics/val_hamming', val_hamming, epoch)
            self.writer.add_scalar('Metrics/val_accuracy', val_accuracy, epoch)

            # Save history
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['val_auc'].append(val_auc)
            training_history['val_hamming'].append(val_hamming)
            training_history['val_accuracy'].append(val_accuracy)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                checkpoint_path = os.path.join(
                    self.config.CHECKPOINT_DIR,
                    f"best_model_{self.config.MODEL_TYPE}.pt"
                )
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"✓ Best model saved (loss: {val_loss:.4f})")

        # Close TensorBoard
        self.writer.close()

        # Save training history
        history_path = os.path.join(self.config.CHECKPOINT_DIR, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=4)
        print(f"\n✓ Training history saved to {history_path}")

        print(f"\n{'=' * 80}")
        print(f"Training Complete!")
        print(f"Best model from epoch {self.best_epoch + 1} (loss: {self.best_val_loss:.4f})")
        print(f"{'=' * 80}")


def main():
    # Set seed
    set_seed(TrainingConfig.SEED)

    print(f"\nDevice: {TrainingConfig.DEVICE}")
    if TrainingConfig.DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"⚠️  Running on CPU - Training will be slow!")
        print(f"   Consider running on GPU for better performance")

    # Create data loaders
    print(f"\n{'=' * 80}")
    print(f"Loading Data")
    print(f"{'=' * 80}")
    train_loader, val_loader = create_data_loaders(
        dataset_dir=TrainingConfig.IMG_DIR,
        csv_file=TrainingConfig.CSV_FILE,
        batch_size=TrainingConfig.BATCH_SIZE,
        num_workers=TrainingConfig.NUM_WORKERS,
        task='classification'
    )

    # Initialize trainer and train
    trainer = ClassificationTrainerWithMetrics(TrainingConfig)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
import torch
from tqdm import tqdm
import os

from config import (
    OUTPUT_ROOT, GRAD_CLIP_VALUE,
    LR_SCHEDULER_FACTOR, LR_SCHEDULER_PATIENCE, LR_SCHEDULER_MIN_LR
)
from .metrics.metrics_logger import MetricsLogger
from .metrics.detection_metrics import DetectionMetricsCalculator


class Trainer:
    def __init__(self, model, criterion, optimizer, train_loader, val_loader, 
                 device, num_epochs, visualization_logger, checkpoints_dir):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=LR_SCHEDULER_FACTOR,
            patience=LR_SCHEDULER_PATIENCE,
            verbose=True,
            min_lr=LR_SCHEDULER_MIN_LR
        )
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.visualization_logger = visualization_logger
        self.checkpoints_dir = checkpoints_dir
        self.metrics_logger = MetricsLogger(OUTPUT_ROOT)
        self.metrics_calculator = DetectionMetricsCalculator()

    def train(self):
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(self.num_epochs):
            # Train for one epoch
            train_loss, val_loss = self.train_epoch(epoch)
            
            # Save best model and implement early stopping
            if val_loss < best_val_loss:
                epochs_without_improvement = 0
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, os.path.join(self.checkpoints_dir, 'best_model.pth'))
                print(f"Saved new best model with val_loss: {val_loss:.4f}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= 10:
                    print("Early stopping triggered")
                    break
        
        self.visualization_logger.close()

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        epoch_predictions = []
        epoch_targets = []
        
        train_loader_bar = tqdm(self.train_loader, desc=f"Training epoch {epoch + 1}/{self.num_epochs}")
        
        for step, (images, _, boxes) in enumerate(train_loader_bar):
            # Prepare data
            images = images.to(self.device)
            targets = [{'boxes': box.to(self.device)} for box in boxes]
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images)
            loss_dict = self.criterion(predictions, targets)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_VALUE)
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            epoch_predictions.extend(predictions)
            epoch_targets.extend(targets)
            
            # Update progress bar
            train_loader_bar.set_postfix({
                'train_loss': total_loss / (step + 1)
            })

        # Calculate average loss and metrics
        avg_loss = total_loss / len(self.train_loader)
        metrics = self.metrics_calculator.calculate_metrics(epoch_predictions, epoch_targets)
        metrics['loss'] = avg_loss
        
        # Log metrics and sample visualizations
        self.visualization_logger.log_epoch_metrics('train', metrics, epoch)
        self.visualization_logger.log_images('train', images[:8], epoch_predictions[:8], epoch_targets[:8], epoch)
        self.metrics_logger.log_epoch_metrics(epoch, 'train', metrics)
        
        # Run validation
        val_loss, val_metrics = self.validate(epoch)
        
        # Update learning rate scheduler
        self.scheduler.step(val_loss)
        
        return avg_loss, val_loss

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        epoch_predictions = []
        epoch_targets = []
        
        with torch.no_grad():
            for images, _, boxes in tqdm(self.val_loader, desc='Validating'):
                # Prepare data
                images = images.to(self.device)
                targets = [{'boxes': box.to(self.device)} for box in boxes]
                
                # Forward pass
                predictions = self.model(images)
                loss_dict = self.criterion(predictions, targets)
                loss = loss_dict['total_loss']
                
                # Update metrics
                total_loss += loss.item()
                epoch_predictions.extend(predictions)
                epoch_targets.extend(targets)
        
        # Calculate average loss and metrics
        avg_loss = total_loss / len(self.val_loader)
        metrics = self.metrics_calculator.calculate_metrics(epoch_predictions, epoch_targets)
        metrics['loss'] = avg_loss
        
        # Log metrics and sample visualizations
        self.visualization_logger.log_epoch_metrics('val', metrics, epoch)
        self.visualization_logger.log_images('val', images[:8], epoch_predictions[:8], epoch_targets[:8], epoch)
        self.metrics_logger.log_epoch_metrics(epoch, 'val', metrics)
        
        return avg_loss, metrics

    def _flatten_metrics(self, metrics, parent_key='', sep='.'):
        """Flatten nested dictionaries of metrics into a single level dictionary."""
        items = []
        for k, v in metrics.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_metrics(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
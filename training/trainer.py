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
        best_f1_score = 0
        epochs_without_improvement = 0

        for epoch in range(self.num_epochs):
            # Train for one epoch
            train_loss, val_metrics = self.train_epoch(epoch)
            
            val_loss = val_metrics['total_loss']
            val_f1 = val_metrics['performance']['f1_score']
            
            # Save best model based on F1 score instead of just validation loss
            improved = False
            if val_f1 > best_f1_score:
                best_f1_score = val_f1
                improved = True
                print(f"New best F1 score: {val_f1:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if not improved:
                    improved = True
                    print(f"New best validation loss: {val_loss:.4f}")
                
            if improved:
                epochs_without_improvement = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_f1': val_f1,
                }, os.path.join(self.checkpoints_dir, 'best_model.pth'))
                print(f"Saved new best model - val_loss: {val_loss:.4f}, val_f1: {val_f1:.4f}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= 10:
                    print("Early stopping triggered")
                    break
        
        self.visualization_logger.close()

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        train_loader_bar = tqdm(self.train_loader, desc=f"Training epoch {epoch + 1}/{self.num_epochs}")
        
        for step, (images, _, boxes) in enumerate(train_loader_bar):
            # Prepare data
            images = torch.stack([image.to(self.device) for image in images])
            targets = []
            for boxes_per_image in boxes:
                target = {
                    'boxes': boxes_per_image.to(self.device),
                    'labels': torch.ones((len(boxes_per_image),), dtype=torch.int64, device=self.device)
                }
                targets.append(target)
            
            # Forward pass and loss calculation
            self.optimizer.zero_grad()
            predictions = self.model(images, targets)
            loss_dict = self.criterion(predictions, targets)
            loss = loss_dict['total_loss']  # This is now a tensor
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_VALUE)
            self.optimizer.step()
            
            # Update progress with float value
            total_loss += loss_dict['loss_values']['total_loss']
            train_loader_bar.set_postfix({'train_loss': total_loss / (step + 1)})
            
            # Get inference predictions for metrics - limit the frequency to reduce memory usage
            if step % 5 == 0:  # Only collect metrics every 5 steps
                with torch.no_grad():
                    inference_preds = self.model(images, None)
                    all_predictions.extend(inference_preds)
                    all_targets.extend(targets)
            
            # Log sample images periodically 
            if step % 50 == 0:
                with torch.no_grad():
                    inference_preds = self.model(images, None)
                    vis_images = images[-min(16, len(images)):]
                    vis_preds = inference_preds[-min(16, len(inference_preds)):]
                    vis_targets = targets[-min(16, len(targets)):]
                    self.visualization_logger.log_images('Train', vis_images, vis_preds, vis_targets, epoch)

        # Calculate average loss
        avg_loss = total_loss / len(self.train_loader)
        
        # Calculate comprehensive metrics if we have predictions collected
        if all_predictions:
            metrics = DetectionMetricsCalculator.calculate_metrics(all_predictions, all_targets)
            metrics.update(loss_dict['loss_values'])  # Use float values for logging
            
            # Log metrics
            self.visualization_logger.log_epoch_metrics('train', metrics, epoch)
            self.metrics_logger.log_epoch_metrics(epoch, 'train', metrics)
        
        # Run validation
        val_metrics = None
        if self.val_loader:
            val_metrics = self.validate(epoch)
            
            # Update learning rate scheduler using F1 score (higher is better, so negate)
            # This helps balance both precision and recall in model optimization
            if val_metrics and 'performance' in val_metrics:
                f1_score = val_metrics['performance']['f1_score']
                self.scheduler.step(-f1_score)  # Negate since scheduler is in 'min' mode
        
        return avg_loss, val_metrics

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, _, boxes in tqdm(self.val_loader, desc='Validating'):
                # Prepare data
                images = torch.stack([image.to(self.device) for image in images])
                targets = []
                for boxes_per_image in boxes:
                    target = {
                        'boxes': boxes_per_image.to(self.device),
                        'labels': torch.ones((len(boxes_per_image),), dtype=torch.int64, device=self.device)
                    }
                    targets.append(target)
                
                # Get predictions
                predictions = self.model(images, targets)
                loss_dict = self.criterion(predictions, targets)
                total_loss += loss_dict['loss_values']['total_loss']  # Use float value
                
                # Get inference predictions for metrics
                inference_preds = self.model(images, None)
                all_predictions.extend(inference_preds)
                all_targets.extend(targets)
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        metrics = DetectionMetricsCalculator.calculate_metrics(all_predictions, all_targets)
        metrics.update(loss_dict['loss_values'])  # Use float values for logging
        metrics['total_loss'] = avg_loss  # Include the average loss in metrics
        
        # Log metrics and sample images
        self.visualization_logger.log_epoch_metrics('val', metrics, epoch)
        self.metrics_logger.log_epoch_metrics(epoch, 'val', metrics)
        
        # Log sample validation images
        if images.size(0) > 0:
            vis_images = images[-min(16, len(images)):]
            vis_preds = inference_preds[-min(16, len(inference_preds)):]
            vis_targets = targets[-min(16, len(targets)):]
            self.visualization_logger.log_images('Val', vis_images, vis_preds, vis_targets, epoch)
        
        return metrics

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
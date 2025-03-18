import torch
from tqdm import tqdm
import os
import numpy as np
from config import (
    DEVICE, LEARNING_RATE, NUM_EPOCHS,
    OUTPUT_ROOT
)
from dataset import get_data_loaders
from model import get_model
from losses import DetectionLoss
from visualization import VisualizationLogger
import math
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

class LRWarmupCosineAnnealing:
    """Custom learning rate scheduler with warmup followed by cosine annealing"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

class WeightDecayScheduler:
    """Dynamically adjust weight decay during training"""
    def __init__(self, optimizer, initial_wd=0.01, final_wd=0.001, total_epochs=100):
        self.optimizer = optimizer
        self.initial_wd = initial_wd
        self.final_wd = final_wd
        self.total_epochs = total_epochs
        
    def step(self, epoch):
        # Linearly decrease weight decay over time
        progress = min(1.0, epoch / self.total_epochs)
        current_wd = self.initial_wd - progress * (self.initial_wd - self.final_wd)
        
        for param_group in self.optimizer.param_groups:
            param_group['weight_decay'] = current_wd
        
        return current_wd

def train():
    # Create save directories
    checkpoints_dir = os.path.join(OUTPUT_ROOT, 'checkpoints')
    tensorboard_dir = os.path.join(OUTPUT_ROOT, 'tensorboard')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    # Initialize visualization logger
    vis_logger = VisualizationLogger(tensorboard_dir)

    # Get model and criterion
    model = get_model(DEVICE)
    criterion = DetectionLoss().to(DEVICE)

    # Setup optimizer with weight decay
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=0.01, 
        betas=(0.9, 0.999)  # Default beta values
    )

    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=4, verbose=True, min_lr=1e-6
    )

    # Get data loaders
    train_loader, val_loader = get_data_loaders()
    
    # Create trainer instance
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        visualization_logger=vis_logger,
        checkpoints_dir=checkpoints_dir
    )
    
    # Train the model
    trainer.train()

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, train_loader, val_loader, 
                 device, num_epochs, visualization_logger, checkpoints_dir):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.visualization_logger = visualization_logger
        self.checkpoints_dir = checkpoints_dir
        
        # Add cosine annealing LR scheduler with warmup
        self.lr_scheduler = LRWarmupCosineAnnealing(
            optimizer=optimizer,
            warmup_epochs=5,  # Increased from 3 to 5 for more stable initial training
            total_epochs=num_epochs,
            min_lr=LEARNING_RATE * 0.01
        )
        
        # Add weight decay scheduler
        self.wd_scheduler = WeightDecayScheduler(
            optimizer=optimizer,
            initial_wd=0.01,
            final_wd=0.0005,
            total_epochs=num_epochs
        )
        
        # For validation loss smoothing
        self.val_loss_history = []
        self.smoothing_window = 3
        
        # For exponential moving average (EMA) loss tracking
        self.ema_val_loss = None
        self.ema_alpha = 0.85  # Changed from 0.9 to 0.85 (more reactive to recent losses)
        
        # For model EMA (parameter averaging)
        self.model_ema = None
        self.ema_update_ratio = 0.999
        
    def train(self):
        best_val_loss = float('inf')
        best_ema_val_loss = float('inf')
        epochs_without_improvement = 0
        
        # Track loss history for analysis
        train_losses = []
        val_losses = []
        ema_val_losses = []
        train_conf_losses = []
        train_bbox_losses = []
        val_conf_losses = []
        val_bbox_losses = []
        learning_rates = []
        weight_decays = []

        # Initialize model EMA
        self.model_ema = self._create_ema_model()

        for epoch in range(self.num_epochs):
            # Update learning rate according to schedule
            current_lr = self.lr_scheduler.step(epoch)
            learning_rates.append(current_lr)
            
            # Update weight decay according to schedule
            current_wd = self.wd_scheduler.step(epoch)
            weight_decays.append(current_wd)
            
            # Train for one epoch
            train_metrics, val_metrics = self.train_epoch(epoch)
            train_loss = train_metrics['total_loss']
            val_loss = val_metrics['total_loss'] if val_metrics else 0
            
            # Store loss values for tracking
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_conf_losses.append(train_metrics['conf_loss'])
            train_bbox_losses.append(train_metrics['bbox_loss'])
            val_conf_losses.append(val_metrics['conf_loss'])
            val_bbox_losses.append(val_metrics['bbox_loss'])
            
            # Calculate smoothed validation loss for more stable early stopping
            self.val_loss_history.append(val_loss)
            smoothed_val_loss = self._get_smoothed_val_loss()
            
            # Update EMA validation loss
            ema_val_loss = self._update_ema_val_loss(val_loss)
            ema_val_losses.append(ema_val_loss)
            
            # Print detailed loss information
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{self.num_epochs} Results:")
            print(f"Learning Rate: {current_lr:.6f}, Weight Decay: {current_wd:.6f}")
            print(f"Training   - Total Loss: {train_loss:.6f}, Conf Loss: {train_metrics['conf_loss']:.6f}, Bbox Loss: {train_metrics['bbox_loss']:.6f}")
            print(f"Validation - Total Loss: {val_loss:.6f}, Conf Loss: {val_metrics['conf_loss']:.6f}, Bbox Loss: {val_metrics['bbox_loss']:.6f}")
            print(f"Smoothed Validation Loss: {smoothed_val_loss:.6f}, EMA Validation Loss: {ema_val_loss:.6f}")
            
            # Print validation metrics
            print(f"Validation Metrics - Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1_score']:.4f}")
            print(f"                  - Mean IoU: {val_metrics['mean_iou']:.4f}, Mean Confidence: {val_metrics['mean_confidence']:.4f}")
            
            # Print loss changes from previous epoch
            if epoch > 0:
                train_loss_change = train_losses[-1] - train_losses[-2]
                val_loss_change = val_losses[-1] - val_losses[-2]
                ema_val_loss_change = ema_val_losses[-1] - ema_val_losses[-2]
                print(f"Loss Changes - Training: {train_loss_change:.6f} ({'↓' if train_loss_change < 0 else '↑'}), "
                      f"Validation: {val_loss_change:.6f} ({'↓' if val_loss_change < 0 else '↑'}), "
                      f"EMA Val: {ema_val_loss_change:.6f} ({'↓' if ema_val_loss_change < 0 else '↑'})")
                
                # Analyze loss components changes
                train_conf_change = train_conf_losses[-1] - train_conf_losses[-2]
                train_bbox_change = train_bbox_losses[-1] - train_bbox_losses[-2]
                val_conf_change = val_conf_losses[-1] - val_conf_losses[-2]
                val_bbox_change = val_bbox_losses[-1] - val_bbox_losses[-2]
                
                print(f"Training Components - Conf: {train_conf_change:.6f} ({'↓' if train_conf_change < 0 else '↑'}), "
                      f"Bbox: {train_bbox_change:.6f} ({'↓' if train_bbox_change < 0 else '↑'})")
                print(f"Validation Components - Conf: {val_conf_change:.6f} ({'↓' if val_conf_change < 0 else '↑'}), "
                      f"Bbox: {val_bbox_change:.6f} ({'↓' if val_bbox_change < 0 else '↑'})")
            print(f"{'='*80}\n")
            
            # Update learning rate based on EMA validation loss
            self.scheduler.step(ema_val_loss)
            
            # Save best model and implement early stopping
            improved = False
            
            if ema_val_loss < best_ema_val_loss:
                epochs_without_improvement = 0
                best_ema_val_loss = ema_val_loss
                improved = True
                
            if val_loss < best_val_loss:  # Still track the absolute best for saving
                best_val_loss = val_loss
                improved = True
                
            if improved:
                # Save both regular model and EMA model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'ema_model_state_dict': self.model_ema.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'ema_val_loss': ema_val_loss
                }, os.path.join(self.checkpoints_dir, 'best_model.pth'))
                print(f"Saved new best model with val_loss: {val_loss:.4f} (EMA: {ema_val_loss:.4f})")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= 15:  # Increased patience
                    print("Early stopping triggered")
                    break
        
        self.visualization_logger.close()
    
    def _create_ema_model(self):
        """Create a copy of the model for EMA tracking"""
        ema_model = get_model(self.device)
        for param_ema, param_model in zip(ema_model.parameters(), self.model.parameters()):
            param_ema.data.copy_(param_model.data)
            param_ema.requires_grad = False
        return ema_model
    
    def _update_ema_model(self):
        """Update EMA model parameters"""
        with torch.no_grad():
            for param_ema, param_model in zip(self.model_ema.parameters(), self.model.parameters()):
                param_ema.data = self.ema_update_ratio * param_ema.data + (1 - self.ema_update_ratio) * param_model.data
    
    def _get_smoothed_val_loss(self):
        """Calculate smoothed validation loss for more stable early stopping"""
        if len(self.val_loss_history) < self.smoothing_window:
            return self.val_loss_history[-1]  # Not enough data points for smoothing
        return sum(self.val_loss_history[-self.smoothing_window:]) / self.smoothing_window
    
    def _update_ema_val_loss(self, val_loss):
        """Update and return exponential moving average of validation loss"""
        if self.ema_val_loss is None:
            self.ema_val_loss = val_loss
        else:
            self.ema_val_loss = self.ema_alpha * self.ema_val_loss + (1 - self.ema_alpha) * val_loss
        return self.ema_val_loss

    def calculate_epoch_metrics(self, predictions, targets):
        """Calculate comprehensive epoch-level metrics"""
        total_images = len(predictions)
        correct_count = 0
        over_detections = 0
        under_detections = 0
        all_ious = []
        all_confidences = []
        total_detections = 0
        total_ground_truth = 0
        true_positives = 0
        
        detections_per_image = []
        
        for pred, target in zip(predictions, targets):
            pred_boxes = pred['boxes']
            pred_scores = pred['scores']
            gt_boxes = target['boxes']
            
            # Count statistics
            num_pred = len(pred_boxes)
            num_gt = len(gt_boxes)
            detections_per_image.append(num_pred)
            total_detections += num_pred
            total_ground_truth += num_gt
            
            # Detection count analysis
            if num_pred == num_gt:
                correct_count += 1
            elif num_pred > num_gt:
                over_detections += 1
            else:
                under_detections += 1
            
            # Collect confidence scores
            if len(pred_scores) > 0:
                all_confidences.extend(pred_scores.cpu().tolist())
            
            # Calculate IoUs for matched predictions
            if num_pred > 0 and num_gt > 0:
                ious = self._calculate_box_iou(pred_boxes, gt_boxes)
                if len(ious) > 0:
                    max_ious, _ = ious.max(dim=0)
                    all_ious.extend(max_ious.cpu().tolist())
                    # Count true positives (IoU > 0.5)
                    true_positives += (max_ious > 0.5).sum().item()
        
        # Convert lists to tensors for histogram logging
        iou_distribution = torch.tensor(all_ious) if all_ious else torch.zeros(0)
        confidence_distribution = torch.tensor(all_confidences) if all_confidences else torch.zeros(0)
        detections_per_image = torch.tensor(detections_per_image)
        
        # Calculate final metrics
        correct_count_percent = (correct_count / total_images) * 100
        avg_detections = total_detections / total_images
        avg_ground_truth = total_ground_truth / total_images
        
        # Calculate mean and median IoU
        mean_iou = np.mean(all_ious) if all_ious else 0
        median_iou = np.median(all_ious) if all_ious else 0
        
        # Calculate confidence score statistics
        mean_confidence = np.mean(all_confidences) if all_confidences else 0
        median_confidence = np.median(all_confidences) if all_confidences else 0
        
        # Calculate precision, recall, and F1 score
        precision = true_positives / total_detections if total_detections > 0 else 0
        recall = true_positives / total_ground_truth if total_ground_truth > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'correct_count_percent': correct_count_percent,
            'over_detections': over_detections,
            'under_detections': under_detections,
            'mean_iou': mean_iou,
            'median_iou': median_iou,
            'mean_confidence': mean_confidence,
            'median_confidence': median_confidence,
            'avg_detections': avg_detections,
            'avg_ground_truth': avg_ground_truth,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'detections_per_image': detections_per_image,
            'iou_distribution': iou_distribution,
            'confidence_distribution': confidence_distribution
        }

    def _calculate_box_iou(self, boxes1, boxes2):
        """Calculate IoU between two sets of boxes"""
        # Convert to x1y1x2y2 format if normalized
        boxes1 = boxes1.clone()
        boxes2 = boxes2.clone()
        
        # Calculate intersection areas
        x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Calculate union areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2 - intersection
        
        return intersection / (union + 1e-6)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        # Track component losses
        total_conf_loss = 0
        total_bbox_loss = 0
        
        train_loader_bar = tqdm(self.train_loader, desc=f"Training epoch {epoch + 1}/{self.num_epochs}")
        
        for step, (images, _, boxes) in enumerate(train_loader_bar):
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
            loss = loss_dict['total_loss']
            
            # Backward pass with gradient clipping
            loss.backward()
            # Increase max_norm from 1.0 to 2.0 for potentially faster convergence
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()
            
            # Update EMA model
            self._update_ema_model()
            
            # Update progress
            total_loss += loss.item()
            total_conf_loss += loss_dict['conf_loss']
            total_bbox_loss += loss_dict['bbox_loss']
            train_loader_bar.set_postfix({
                'train_loss': total_loss / (step + 1),
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Collect predictions and targets for epoch-level metrics
            with torch.no_grad():
                inference_preds = self.model(images, None)
                all_predictions.extend(inference_preds)
                all_targets.extend(targets)
            
            # Log sample images periodically
            if step % 50 == 0:
                self.visualization_logger.log_images('Train', images, inference_preds, targets, epoch)

        # Calculate average loss and metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_conf_loss = total_conf_loss / len(self.train_loader)
        avg_bbox_loss = total_bbox_loss / len(self.train_loader)
        
        # Calculate epoch-level metrics
        metrics = self.calculate_epoch_metrics(all_predictions, all_targets)
        metrics.update({
            'total_loss': avg_loss,
            'conf_loss': avg_conf_loss,
            'bbox_loss': avg_bbox_loss
        })
        
        # Log epoch metrics
        self.visualization_logger.log_epoch_metrics('train', metrics, epoch)
        
        # Run validation
        val_metrics = None
        if self.val_loader:
            # Run validation on both regular model and EMA model
            val_metrics = self.validate(epoch)
        
        return metrics, val_metrics
    
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        # Initialize component loss tracking
        total_conf_loss = 0
        total_bbox_loss = 0
        
        # Store EMA model predictions for later evaluation
        ema_all_predictions = []
        ema_metrics = None
        
        with torch.no_grad():
            for images, _, boxes in tqdm(self.val_loader, desc='Validating'):
                images = torch.stack([image.to(self.device) for image in images])
                targets = []
                for boxes_per_image in boxes:
                    target = {
                        'boxes': boxes_per_image.to(self.device),
                        'labels': torch.ones((len(boxes_per_image),), dtype=torch.int64, device=self.device)
                    }
                    targets.append(target)
                
                # IMPORTANT: Force the model to return predictions in training format for loss calculation
                # This ensures consistency between training and validation loss calculation
                self.model.train()  # Temporarily set to train mode
                predictions = self.model(images, targets)  # Get predictions in training format 
                self.model.eval()   # Set back to eval mode
                
                # Calculate loss using the same format as training
                loss_dict = self.criterion(predictions, targets)
                total_loss += loss_dict['total_loss'].item()
                
                # Track component losses properly
                total_conf_loss += loss_dict['conf_loss']
                total_bbox_loss += loss_dict['bbox_loss']
                
                # Get inference predictions for metrics (now in eval mode)
                inference_preds = self.model(images, None)
                all_predictions.extend(inference_preds)
                all_targets.extend(targets)
                
                # Also get EMA model predictions
                self.model_ema.eval()
                ema_preds = self.model_ema(images, None)
                ema_all_predictions.extend(ema_preds)
        
        # Calculate metrics for regular model
        avg_loss = total_loss / len(self.val_loader)
        avg_conf_loss = total_conf_loss / len(self.val_loader)
        avg_bbox_loss = total_bbox_loss / len(self.val_loader)
        
        metrics = self.calculate_epoch_metrics(all_predictions, all_targets)
        metrics.update({
            'total_loss': avg_loss,
            'conf_loss': avg_conf_loss,
            'bbox_loss': avg_bbox_loss
        })
        
        # Calculate metrics for EMA model
        ema_metrics = self.calculate_epoch_metrics(ema_all_predictions, all_targets)
        ema_metrics.update({
            'total_loss': avg_loss,  # Use the same loss values as regular model since we can't calculate loss for EMA model directly
            'conf_loss': avg_conf_loss,
            'bbox_loss': avg_bbox_loss
        })
        
        # Log validation metrics for regular model
        self.visualization_logger.log_epoch_metrics('val', metrics, epoch)
        
        # Log validation metrics for EMA model
        self.visualization_logger.log_epoch_metrics('val_ema', ema_metrics, epoch)
        
        # Log sample validation images
        self.visualization_logger.log_images('Val', images[-16:], inference_preds[-16:], targets[-16:], epoch)
        
        # Log EMA model predictions too
        self.visualization_logger.log_images('Val_EMA', images[-16:], ema_preds[-16:], targets[-16:], epoch)
        
        # Print EMA model metrics
        print(f"EMA Model Metrics - Precision: {ema_metrics['precision']:.4f}, Recall: {ema_metrics['recall']:.4f}, F1: {ema_metrics['f1_score']:.4f}")
        print(f"                  - Mean IoU: {ema_metrics['mean_iou']:.4f}")
        
        return metrics

    def calculate_epoch_metrics(self, predictions, targets):
        """Calculate comprehensive epoch-level metrics"""
        total_images = len(predictions)
        correct_count = 0
        over_detections = 0
        under_detections = 0
        all_ious = []
        all_confidences = []
        total_detections = 0
        total_ground_truth = 0
        true_positives = 0
        
        detections_per_image = []
        
        for pred, target in zip(predictions, targets):
            pred_boxes = pred['boxes']
            pred_scores = pred['scores']
            gt_boxes = target['boxes']
            
            # Count statistics
            num_pred = len(pred_boxes)
            num_gt = len(gt_boxes)
            detections_per_image.append(num_pred)
            total_detections += num_pred
            total_ground_truth += num_gt
            
            # Detection count analysis
            if num_pred == num_gt:
                correct_count += 1
            elif num_pred > num_gt:
                over_detections += 1
            else:
                under_detections += 1
            
            # Collect confidence scores
            if len(pred_scores) > 0:
                all_confidences.extend(pred_scores.cpu().tolist())
            
            # Calculate IoUs for matched predictions
            if num_pred > 0 and num_gt > 0:
                ious = self._calculate_box_iou(pred_boxes, gt_boxes)
                if len(ious) > 0:
                    max_ious, _ = ious.max(dim=0)
                    all_ious.extend(max_ious.cpu().tolist())
                    # Count true positives (IoU > 0.5)
                    true_positives += (max_ious > 0.5).sum().item()
        
        # Convert lists to tensors for histogram logging
        iou_distribution = torch.tensor(all_ious) if all_ious else torch.zeros(0)
        confidence_distribution = torch.tensor(all_confidences) if all_confidences else torch.zeros(0)
        detections_per_image = torch.tensor(detections_per_image)
        
        # Calculate final metrics
        correct_count_percent = (correct_count / total_images) * 100
        avg_detections = total_detections / total_images
        avg_ground_truth = total_ground_truth / total_images
        
        # Calculate mean and median IoU
        mean_iou = np.mean(all_ious) if all_ious else 0
        median_iou = np.median(all_ious) if all_ious else 0
        
        # Calculate confidence score statistics
        mean_confidence = np.mean(all_confidences) if all_confidences else 0
        median_confidence = np.median(all_confidences) if all_confidences else 0
        
        # Calculate precision, recall, and F1 score
        precision = true_positives / total_detections if total_detections > 0 else 0
        recall = true_positives / total_ground_truth if total_ground_truth > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'correct_count_percent': correct_count_percent,
            'over_detections': over_detections,
            'under_detections': under_detections,
            'mean_iou': mean_iou,
            'median_iou': median_iou,
            'mean_confidence': mean_confidence,
            'median_confidence': median_confidence,
            'avg_detections': avg_detections,
            'avg_ground_truth': avg_ground_truth,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'detections_per_image': detections_per_image,
            'iou_distribution': iou_distribution,
            'confidence_distribution': confidence_distribution
        }

    def _calculate_box_iou(self, boxes1, boxes2):
        """Calculate IoU between two sets of boxes"""
        # Convert to x1y1x2y2 format if normalized
        boxes1 = boxes1.clone()
        boxes2 = boxes2.clone()
        
        # Calculate intersection areas
        x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Calculate union areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2 - intersection
        
        return intersection / (union + 1e-6)

if __name__ == "__main__":
    train()
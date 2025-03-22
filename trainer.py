import torch
import os
from torchvision.ops import box_iou
from config import (
    DEVICE, LEARNING_RATE, NUM_EPOCHS,
    OUTPUT_ROOT
)
from dataset_package import create_dataloaders
from model import get_model
from losses import DetectionLoss
from visualization import VisualizationLogger
import math
import logging

# Use the same logger as in dataset.py
logger = logging.getLogger('dog_detector')

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
    """Main training function"""
    logger.info("Initializing training...")
    
    # Create output directories
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    checkpoints_dir = os.path.join(OUTPUT_ROOT, 'checkpoints')
    tensorboard_dir = os.path.join(OUTPUT_ROOT, 'tensorboard')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # Initialize visualization
    vis_logger = VisualizationLogger(tensorboard_dir)
    
    # Get data loaders
    train_loader, val_loader = create_dataloaders()
    
    # Initialize model, criterion, optimizer
    model = get_model(DEVICE)
    criterion = DetectionLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.02
    )
    
    # Learning rate scheduler with warmup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% warmup
        div_factor=25,  # LR starts at LR/25
        final_div_factor=1000  # Final LR is LR/1000
    )
    
    # Training loop
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    logger.info(f"Starting training for {NUM_EPOCHS} epochs")
    
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0
        train_conf_loss = 0
        train_bbox_loss = 0
        
        for batch_idx, (images, boxes, labels) in enumerate(train_loader):
            # Prepare batch
            images = images.to(DEVICE)
            targets = [{
                'boxes': box.to(DEVICE),
                'labels': label.to(DEVICE)
            } for box, label in zip(boxes, labels)]
            
            # Forward pass
            predictions = model(images, targets)
            loss_dict = criterion(predictions, targets)
            loss = loss_dict['total_loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            train_loss += loss.item()
            train_conf_loss += loss_dict['conf_loss']
            train_bbox_loss += loss_dict['bbox_loss']
            
            # Log progress
            if batch_idx % 20 == 0:
                logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS} [{batch_idx}/{len(train_loader)}] "
                          f"Loss: {loss.item():.4f}")
        
        # Calculate average losses
        train_loss /= len(train_loader)
        train_conf_loss /= len(train_loader)
        train_bbox_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_conf_loss = 0
        val_bbox_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, boxes, labels in val_loader:
                images = images.to(DEVICE)
                targets = [{
                    'boxes': box.to(DEVICE),
                    'labels': label.to(DEVICE)
                } for box, label in zip(boxes, labels)]
                
                predictions = model(images, targets)
                loss_dict = criterion(predictions, targets)
                
                val_loss += loss_dict['total_loss'].item()
                val_conf_loss += loss_dict['conf_loss']
                val_bbox_loss += loss_dict['bbox_loss']
                
                # Get inference predictions for metrics
                inference_preds = model(images, None)
                all_predictions.extend(inference_preds)
                all_targets.extend(targets)
        
        # Calculate average validation losses
        val_loss /= len(val_loader)
        val_conf_loss /= len(val_loader)
        val_bbox_loss /= len(val_loader)
        
        # Calculate detection metrics
        metrics = calculate_metrics(all_predictions, all_targets)
        
        # Log metrics
        logger.info(f"\nEpoch {epoch+1} Summary:")
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        
        # Log to tensorboard
        vis_logger.log_epoch_metrics('train', {
            'total_loss': train_loss,
            'conf_loss': train_conf_loss,
            'bbox_loss': train_bbox_loss
        }, epoch)
        
        vis_logger.log_epoch_metrics('val', {
            'total_loss': val_loss,
            'conf_loss': val_conf_loss,
            'bbox_loss': val_bbox_loss,
            **metrics
        }, epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'metrics': metrics
            }, os.path.join(checkpoints_dir, 'best_model.pth'))
            logger.info(f"Saved new best model with val_loss: {val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            
        # Early stopping
        if epochs_without_improvement >= 15:
            logger.info(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement")
            break
    
    vis_logger.close()
    logger.info("Training completed")

def calculate_metrics(predictions, targets):
    """Calculate precision, recall, and F1 score for object detection"""
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        gt_boxes = target['boxes']
        gt_labels = target['labels']
        
        # Skip if no predictions or ground truth
        if len(pred_boxes) == 0:
            total_false_negatives += len(gt_boxes)
            continue
        if len(gt_boxes) == 0:
            total_false_positives += len(pred_boxes)
            continue
        
        # Calculate IoU between all predictions and ground truth
        ious = box_iou(pred_boxes, gt_boxes)
        
        # For each ground truth box, find the best matching prediction
        matched_preds = set()
        for gt_idx, gt_label in enumerate(gt_labels):
            # Only consider predictions with matching class
            matching_class_mask = pred_labels == gt_label
            if not matching_class_mask.any():
                total_false_negatives += 1
                continue
                
            ious_for_gt = ious[matching_class_mask, gt_idx]
            if len(ious_for_gt) == 0:
                total_false_negatives += 1
                continue
                
            # Find best matching prediction
            best_pred_idx = torch.where(matching_class_mask)[0][ious_for_gt.argmax()]
            best_iou = ious_for_gt.max()
            
            # If IoU is good enough and prediction hasn't been matched yet
            if best_iou >= 0.5 and best_pred_idx.item() not in matched_preds:
                total_true_positives += 1
                matched_preds.add(best_pred_idx.item())
            else:
                total_false_negatives += 1
        
        # Count unmatched predictions as false positives
        total_false_positives += len(pred_boxes) - len(matched_preds)
    
    # Calculate metrics
    precision = total_true_positives / (total_true_positives + total_false_positives) if total_true_positives > 0 else 0
    recall = total_true_positives / (total_true_positives + total_false_negatives) if total_true_positives > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': total_true_positives,
        'false_positives': total_false_positives,
        'false_negatives': total_false_negatives
    }

if __name__ == "__main__":
    train()
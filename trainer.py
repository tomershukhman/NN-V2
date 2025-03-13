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
from metrics_logger import MetricsCSVLogger


def train():
    # Create save directories
    checkpoints_dir = os.path.join(OUTPUT_ROOT, 'checkpoints')
    tensorboard_dir = os.path.join(OUTPUT_ROOT, 'tensorboard')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    # Initialize visualization and metrics loggers
    vis_logger = VisualizationLogger(tensorboard_dir)
    csv_logger = MetricsCSVLogger(OUTPUT_ROOT)

    # Get model and criterion
    model = get_model(DEVICE)
    criterion = DetectionLoss(use_focal_loss=True).to(DEVICE)

    # Setup optimizer with weight decay and parameter groups for different learning rates
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if 'backbone' in n], 'lr': LEARNING_RATE * 0.1},
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n]}
    ]
    
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=LEARNING_RATE,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )

    # Get total training steps for learning rate scheduler
    train_loader, val_loader = get_data_loaders()
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = min(1000, total_steps // 5)
    
    # Use a cosine annealing scheduler with warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
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
        metrics_csv_logger=csv_logger,
        checkpoints_dir=checkpoints_dir
    )
    
    # Train the model
    trainer.train()

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """
    Create a cosine learning rate scheduler with warmup
    """
    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine annealing
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, train_loader, val_loader, 
                 device, num_epochs, visualization_logger, metrics_csv_logger, checkpoints_dir):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.visualization_logger = visualization_logger
        self.metrics_csv_logger = metrics_csv_logger
        self.checkpoints_dir = checkpoints_dir
        self.gradient_clip_val = 1.0  # Limit gradient magnitude to prevent exploding gradients
        
    def train(self):
        best_val_loss = float('inf')
        best_f1_score = 0
        epochs_without_improvement = 0

        for epoch in range(self.num_epochs):
            # Train for one epoch
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Run validation
            val_metrics = self.validate(epoch)
            val_loss = val_metrics['total_loss'] if val_metrics else float('inf')
            val_f1 = val_metrics['f1_score'] if val_metrics else 0
            
            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.visualization_logger.writer.add_scalar('learning_rate', current_lr, epoch)
            
            # Save best model by validation loss
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
                    'val_f1': val_f1,
                }, os.path.join(self.checkpoints_dir, 'best_loss_model.pth'))
                print(f"Saved new best model with val_loss: {val_loss:.4f}")
                
            # Also save best model by F1 score
            if val_f1 > best_f1_score:
                best_f1_score = val_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_f1': val_f1,
                }, os.path.join(self.checkpoints_dir, 'best_f1_model.pth'))
                print(f"Saved new best model with val_F1: {val_f1:.4f}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= 15:  # More patient early stopping
                    print("Early stopping triggered after 15 epochs without improvement")
                    break
        
        self.visualization_logger.close()

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
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            self.optimizer.step()
            self.scheduler.step()  # Step the scheduler every batch
            
            # Update progress
            total_loss += loss.item()
            train_loader_bar.set_postfix({
                'train_loss': total_loss / (step + 1),
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Collect predictions and targets for epoch-level metrics
            with torch.no_grad():
                inference_preds = self.model(images, None)
                all_predictions.extend(inference_preds)
                all_targets.extend(targets)
            
            # Log sample images periodically
            if step % 50 == 0:
                self.visualization_logger.log_images('train', images, inference_preds, targets, epoch)

        # Calculate average loss and metrics
        avg_loss = total_loss / len(self.train_loader)
        
        # Calculate epoch-level metrics
        metrics = self.calculate_epoch_metrics(all_predictions, all_targets)
        metrics.update({
            'total_loss': avg_loss,
            'conf_loss': loss_dict['conf_loss'],
            'bbox_loss': loss_dict['bbox_loss']
        })
        
        # Log epoch metrics to TensorBoard and CSV
        self.visualization_logger.log_epoch_metrics('train', metrics, epoch)
        self.metrics_csv_logger.log_metrics('train', metrics, epoch)
        
        return avg_loss, metrics

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        total_conf_loss = 0
        total_bbox_loss = 0
        all_predictions = []
        all_targets = []
        
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
                
                # Get predictions
                predictions = self.model(images, targets)  # For loss calculation
                loss_dict = self.criterion(predictions, targets)
                total_loss += loss_dict['total_loss'].item()
                total_conf_loss += loss_dict['conf_loss']
                total_bbox_loss += loss_dict['bbox_loss']
                
                # Get inference predictions for metrics
                inference_preds = self.model(images, None)
                all_predictions.extend(inference_preds)
                all_targets.extend(targets)
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        avg_conf_loss = total_conf_loss / len(self.val_loader)
        avg_bbox_loss = total_bbox_loss / len(self.val_loader)
        
        metrics = self.calculate_epoch_metrics(all_predictions, all_targets)
        metrics.update({
            'total_loss': avg_loss,
            'conf_loss': avg_conf_loss,
            'bbox_loss': avg_bbox_loss
        })
        
        # Log validation metrics to TensorBoard and CSV
        self.visualization_logger.log_epoch_metrics('val', metrics, epoch)
        self.metrics_csv_logger.log_metrics('val', metrics, epoch)
        
        # Log sample validation images
        self.visualization_logger.log_images('Val', images[-16:], inference_preds[-16:], targets[-16:], epoch)
        
        return metrics


if __name__ == "__main__":
    train()
import torch
from tqdm import tqdm
import os
import numpy as np

from config import OUTPUT_ROOT
from .metrics.metrics_logger import MetricsLogger
from .metrics.detection_metrics import DetectionMetricsCalculator


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
        self.metrics_logger = MetricsLogger(OUTPUT_ROOT)
        self.metrics_calculator = DetectionMetricsCalculator()

    def train(self):
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(self.num_epochs):
            # Train for one epoch
            train_loss, val_loss = self.train_epoch(epoch)
            
            # Update learning rate based on validation loss
            self.scheduler.step(val_loss)
            
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
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update progress
            total_loss += loss.item()
            train_loader_bar.set_postfix({'train_loss': total_loss / (step + 1)})
            
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
        
        # Calculate epoch-level metrics
        metrics = self.calculate_epoch_metrics(all_predictions, all_targets)
        metrics['total_loss'] = avg_loss
        metrics.update(loss_dict)
        
        # Log epoch metrics
        self.visualization_logger.log_epoch_metrics('train', metrics, epoch)
        self.metrics_logger.log_epoch_metrics(epoch, 'train', metrics)
        
        # Run validation
        val_metrics = None
        if self.val_loader:
            val_metrics = self.validate(epoch)
        
        return avg_loss, val_metrics['total_loss'] if val_metrics else 0

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
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
                
                # Get inference predictions for metrics
                inference_preds = self.model(images, None)
                all_predictions.extend(inference_preds)
                all_targets.extend(targets)
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        metrics = self.calculate_epoch_metrics(all_predictions, all_targets)
        metrics['total_loss'] = avg_loss
        metrics.update(loss_dict)
        
        # Log metrics
        self.visualization_logger.log_epoch_metrics('val', metrics, epoch)
        self.metrics_logger.log_epoch_metrics(epoch, 'val', metrics)
        
        # Log sample validation images
        self.visualization_logger.log_images('Val', images[-16:], inference_preds[-16:], targets[-16:], epoch)
        
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
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
        
        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
        
        union = area1[:, None] + area2 - inter
        iou = inter / union
        
        return iou
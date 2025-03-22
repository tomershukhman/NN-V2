import torch
from tqdm import tqdm
import os
import numpy as np
from config import (
    DEVICE, LEARNING_RATE, NUM_EPOCHS,
    OUTPUT_ROOT
)
from dataset import get_data_loaders, get_multi_dog_transform
from model import get_model
from losses import DetectionLoss
from visualization import VisualizationLogger
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import logging
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast

# Enable cuDNN autotuner
cudnn.benchmark = True

# Use the same logger as in dataset.py
logger = logging.getLogger('dog_detector')

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
    
    logger.info("Starting training process...")
    
    # Initialize visualization logger
    vis_logger = VisualizationLogger(tensorboard_dir)
    
    # Get model and criterion
    model = get_model(DEVICE)
    criterion = DetectionLoss().to(DEVICE)
    
    # Setup optimizer with weight decay
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=0.02,  # Increased weight decay for more regularization
        betas=(0.9, 0.999)
    )
    
    # Add learning rate scheduler with more gradual decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.6, patience=8, verbose=False, min_lr=5e-7
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
        
        # Add weight decay scheduler with more gentle decay
        self.wd_scheduler = WeightDecayScheduler(
            optimizer=optimizer,
            initial_wd=0.02,
            final_wd=0.005,
            total_epochs=num_epochs
        )
        
        # For validation loss smoothing
        self.val_loss_history = []
        self.smoothing_window = 5  # Increased window size
        
        # Add dropout scheduling
        self.initial_dropout = 0.1
        self.final_dropout = 0.3
        
        # Create validation mini-batching for more stable evaluation
        self.val_batch_size = 16
        
        # Gradient accumulation to stabilize training
        self.gradient_accumulation_steps = 4  # Reduced from 8
        
        # Increased validation frequency for better monitoring
        self.validation_cycle = 3  # Reduced from 5 to catch overfitting earlier
        
        # Refined loss weights for better balance
        self.loss_conf_weight_start = 1.0  # Back to original
        self.loss_conf_weight_end = 1.0    # Keep constant
        self.loss_bbox_weight_start = 1.0  # Start equal
        self.loss_bbox_weight_end = 1.2    # Slight increase for bbox accuracy
        
        # Early stopping with more patience
        self.early_stopping_patience = 15  # Match config
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        # Add tracking for multi-dog performance
        self.multi_dog_metrics = {
            'single_dog_precision': [],
            'multi_dog_precision': [],
            'single_dog_recall': [],
            'multi_dog_recall': []
        }
        
        # Additional loss weights specifically for multi-dog cases
        self.multi_dog_conf_weight = 1.2  # Slightly higher confidence weight for multi-dog cases
        
        logger.info("Trainer initialized with weight decay scheduler")
        
        # Initialize mixed precision training
        self.scaler = GradScaler()
        
    def train(self):
        # Initialize tracking variables
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        # Track loss history
        train_losses = []
        val_losses = []
        learning_rates = []
        weight_decays = []
        
        logger.info(f"Starting training for {self.num_epochs} epochs")
        
        for epoch in range(self.num_epochs):
            # Update weight decay
            current_wd = self.wd_scheduler.step(epoch)
            weight_decays.append(current_wd)
            
            # Apply scheduled dropout
            self._update_dropout(epoch)
            
            # Update loss weights
            conf_weight, bbox_weight = self._update_loss_weights(epoch)
            
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            
            # Train for one epoch
            train_metrics, val_metrics = self.train_epoch(epoch, conf_weight, bbox_weight)
            train_loss = train_metrics['total_loss']
            val_loss = val_metrics['total_loss'] if val_metrics else float('inf')
            
            # Store loss values
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Calculate smoothed validation loss
            self.val_loss_history.append(val_loss)
            smoothed_val_loss = self._get_smoothed_val_loss()
            
            # Log basic epoch results
            logger.info(f"Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}")
            
            # Log validation metrics in more detail
            logger.info(f"Val metrics - Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1_score']:.4f}")
            logger.info(f"Val metrics - Mean IoU: {val_metrics['mean_iou']:.4f}, Median IoU: {val_metrics['median_iou']:.4f}")
            logger.info(f"Val metrics - Over/Under detections: {val_metrics['over_detections']}/{val_metrics['under_detections']}, Count Match: {val_metrics['count_match_percentage']:.1f}%")
            
            # Update learning rate based on smoothed validation loss
            self.scheduler.step(smoothed_val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            
            # Save best model logic - only based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                
                # Save model state
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss,
                }
                
                torch.save(save_dict, os.path.join(self.checkpoints_dir, 'best_model.pth'))
                logger.info(f"Saved new best model with val_loss: {val_loss:.4f}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement")
                    break
        
        self.visualization_logger.close()

    def _update_loss_weights(self, epoch):
        """Update loss component weights according to schedule"""
        progress = min(1.0, epoch / self.num_epochs)
        conf_weight = self.loss_conf_weight_start + progress * (self.loss_conf_weight_end - self.loss_conf_weight_start)
        bbox_weight = self.loss_bbox_weight_start + progress * (self.loss_bbox_weight_end - self.loss_bbox_weight_start)
        return conf_weight, bbox_weight
        
    def _update_dropout(self, epoch):
        """Update dropout rate according to schedule"""
        progress = min(1.0, epoch / self.num_epochs)
        current_dropout = self.initial_dropout + progress * (self.final_dropout - self.initial_dropout)
        
        # Apply dropout to applicable layers (implementation depends on your model structure)
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = current_dropout
    
    def _get_smoothed_val_loss(self):
        """Calculate smoothed validation loss for more stable early stopping"""
        if len(self.val_loss_history) < self.smoothing_window:
            return self.val_loss_history[-1]
        
        # Use median instead of mean for better robustness against outliers
        recent_losses = self.val_loss_history[-self.smoothing_window:]
        return np.median(recent_losses)
        
    def train_epoch(self, epoch, conf_weight, bbox_weight):
        """Train for one epoch and return metrics"""
        logger.info("Starting training epoch")
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        # Track component losses
        total_conf_loss = 0
        total_bbox_loss = 0
        
        try:
            train_loader_bar = tqdm(self.train_loader, desc=f"Training epoch {epoch + 1}/{self.num_epochs}", 
                                    ncols=100, leave=False)  # Make tqdm less verbose
            
            # Implement gradient accumulation
            self.optimizer.zero_grad()
            accumulated_steps = 0
            
            for step, (images, num_dogs, boxes) in enumerate(train_loader_bar):
                images = torch.stack([image.to(self.device) for image in images])
                targets = []
                
                # Create targets with dog count information included
                for i, boxes_per_image in enumerate(boxes):
                    target = {
                        'boxes': boxes_per_image.to(self.device),
                        'labels': torch.ones((len(boxes_per_image),), dtype=torch.int64, device=self.device),
                        'dog_count': num_dogs[i].item()  # Include dog count
                    }
                    targets.append(target)
                
                # Forward pass with dynamic loss weights based on dog count
                with autocast():
                    predictions = self.model(images, targets)
                
                    # Apply different weights for multi-dog vs single-dog images
                    batch_conf_weights = []
                    batch_bbox_weights = []
                    
                    for target in targets:
                        if target['dog_count'] > 1:
                            # For multi-dog images, use slightly higher confidence weight
                            batch_conf_weights.append(conf_weight * self.multi_dog_conf_weight)
                            batch_bbox_weights.append(bbox_weight)
                        else:
                            # For single-dog images, use standard weights
                            batch_conf_weights.append(conf_weight)
                            batch_bbox_weights.append(bbox_weight)
                    
                    # Calculate average weights for the batch
                    avg_conf_weight = sum(batch_conf_weights) / len(batch_conf_weights) if batch_conf_weights else conf_weight
                    avg_bbox_weight = sum(batch_bbox_weights) / len(batch_bbox_weights) if batch_bbox_weights else bbox_weight
                    
                    # Calculate loss with adjusted weights
                    loss_dict = self.criterion(predictions, targets, conf_weight=avg_conf_weight, bbox_weight=avg_bbox_weight)
                    loss = loss_dict['total_loss']
                
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                
                # Backward pass with gradient clipping
                self.scaler.scale(loss).backward()
                
                accumulated_steps += 1
                
                # Only perform optimization step after accumulating gradients
                if accumulated_steps == self.gradient_accumulation_steps or step == len(self.train_loader) - 1:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=3.0)
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    accumulated_steps = 0
                
                # Update progress
                total_loss += loss.item() * self.gradient_accumulation_steps  # Scale back for tracking
                total_conf_loss += loss_dict['conf_loss']
                total_bbox_loss += loss_dict['bbox_loss']
                
                # Only update tqdm bar with basic info to reduce log verbosity
                train_loader_bar.set_postfix({
                    'loss': f"{total_loss / (step + 1):.4f}"
                })
                
                # Collect predictions and targets for epoch-level metrics
                with torch.no_grad():
                    inference_preds = self.model(images, None)
                    all_predictions.extend(inference_preds)
                    all_targets.extend(targets)
                
                # Log sample images periodically (less frequently)
                if step % 100 == 0:
                    try:
                        self.visualization_logger.log_images('Train', images[-4:], inference_preds[-4:], targets[-4:], epoch)
                    except Exception as e:
                        logger.warning(f"Error logging images: {e}")
                    
            logger.info("Finished training loop, calculating metrics")
            
            # Calculate average loss and metrics
            avg_loss = total_loss / len(self.train_loader)
            avg_conf_loss = total_conf_loss / len(self.train_loader)
            avg_bbox_loss = total_bbox_loss / len(self.train_loader)
            
            # Calculate epoch-level metrics with separate metrics for multi-dog cases
            logger.info("Calculating epoch metrics")
            metrics = self.calculate_epoch_metrics(all_predictions, all_targets)
            metrics.update({
                'total_loss': avg_loss,
                'conf_loss': avg_conf_loss,
                'bbox_loss': avg_bbox_loss
            })
            
            # Log epoch metrics
            logger.info("Logging epoch metrics")
            self.visualization_logger.log_epoch_metrics('train', metrics, epoch)
            
            # Run validation
            val_metrics = None
            if self.val_loader:
                logger.info("Starting validation")
                # Run validation using cycle strategy to reduce zigzag pattern
                val_metrics_list = []
                for i in range(self.validation_cycle):
                    logger.info(f"Validation cycle {i+1}/{self.validation_cycle}")
                    # Run validation with different random seeds
                    seed = random.randint(0, 10000) 
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    random.seed(seed)
                    
                    curr_val_metrics = self.validate(epoch, conf_weight, bbox_weight)
                    val_metrics_list.append(curr_val_metrics)
                
                # Average the validation metrics to reduce variance
                logger.info("Averaging validation metrics")
                val_metrics = self._average_validation_metrics(val_metrics_list)
                
                # Log the averaged validation metrics
                logger.info("Logging averaged validation metrics")
                self.visualization_logger.log_epoch_metrics('val_avg', val_metrics, epoch)
            
            logger.info("Completed epoch successfully")
            return metrics, val_metrics
        except Exception as e:
            logger.error(f"Error in train_epoch: {e}", exc_info=True)
            # Return default metrics to prevent further crashes
            empty_metrics = {'total_loss': 999.0, 'conf_loss': 0.0, 'bbox_loss': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
            return empty_metrics, empty_metrics
    
    def _average_validation_metrics(self, metrics_list):
        """Average multiple validation runs to reduce variance"""
        if not metrics_list:
            return None
            
        avg_metrics = {}
        for key in metrics_list[0].keys():
            if key in ['iou_distribution', 'confidence_distribution', 'detections_per_image']:
                # For tensor metrics, concatenate them
                tensors = [metrics[key] for metrics in metrics_list]
                if all(tensor.numel() > 0 for tensor in tensors):
                    avg_metrics[key] = torch.cat(tensors)
                else:
                    avg_metrics[key] = torch.zeros(0)
            elif key in ['single_dog_metrics', 'multi_dog_metrics']:
                # Handle nested dictionaries specially
                avg_metrics[key] = {}
                # Get all the nested keys
                nested_keys = metrics_list[0][key].keys()
                for nested_key in nested_keys:
                    avg_metrics[key][nested_key] = sum(metrics[key][nested_key] for metrics in metrics_list) / len(metrics_list)
            else:
                # For scalar metrics, average them
                try:
                    avg_metrics[key] = sum(metrics[key] for metrics in metrics_list) / len(metrics_list)
                except TypeError:
                    # If we get here, it means the metric couldn't be summed - log it for debugging
                    logger.warning(f"Could not average metric '{key}', using first value instead")
                    avg_metrics[key] = metrics_list[0][key]
                    
        return avg_metrics
    
    def validate(self, epoch, conf_weight, bbox_weight):
        """Validate the regular model"""
        # Validate regular model
        val_metrics = self.validate_model(self.model, epoch, prefix="val", conf_weight=conf_weight, bbox_weight=bbox_weight)
        
        return val_metrics
        
    def validate_model(self, model, epoch, prefix="val", conf_weight=1.0, bbox_weight=1.0):
        """Run validation on a specific model instance"""
        model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        num_samples = 0
        
        # Initialize component loss tracking
        total_conf_loss = 0
        total_bbox_loss = 0
        
        with torch.no_grad():
            # Process in batches directly
            for images, _, boxes in self.val_loader:
                # Move to device and create targets
                images = torch.stack([image.to(self.device) for image in images])
                targets = []
                for boxes_per_image in boxes:
                    target = {
                        'boxes': boxes_per_image.to(self.device),
                        'labels': torch.ones((len(boxes_per_image),), dtype=torch.int64, device=self.device)
                    }
                    targets.append(target)
                
                num_samples += len(images)
                
                # Get predictions in training format for loss calculation
                if model == self.model:
                    model.train()  # Temporarily set to train mode
                    with autocast():
                        predictions = model(images, targets)
                    model.eval()  # Set back to eval mode
                    
                    # Calculate loss
                    loss_dict = self.criterion(predictions, targets, conf_weight=conf_weight, bbox_weight=bbox_weight)
                    total_loss += loss_dict['total_loss'].item() * len(targets)
                    total_conf_loss += loss_dict['conf_loss'] * len(targets)
                    total_bbox_loss += loss_dict['bbox_loss'] * len(targets)
                
                # Get inference predictions for metrics
                inference_preds = model(images, None)
                all_predictions.extend(inference_preds)
                all_targets.extend(targets)
                
                # Free memory explicitly
                del images
                del targets
                if 'predictions' in locals():
                    del predictions
                torch.cuda.empty_cache()
        
        # Calculate metrics
        avg_loss = total_loss / num_samples
        avg_conf_loss = total_conf_loss / num_samples
        avg_bbox_loss = total_bbox_loss / num_samples
        
        metrics = self.calculate_epoch_metrics(all_predictions, all_targets)
        metrics.update({
            'total_loss': avg_loss,
            'conf_loss': avg_conf_loss,
            'bbox_loss': avg_bbox_loss
        })
        
        # Log validation metrics
        self.visualization_logger.log_epoch_metrics(prefix, metrics, epoch)
            
        # Log sample validation images (only for regular validation)
        if prefix == "val":
            # Get a small batch for visualization
            vis_images = []
            vis_preds = []
            vis_targets = []
            with torch.no_grad():
                for images, _, boxes in self.val_loader:
                    if len(vis_images) >= 8:
                        break
                    vis_images.extend(images[:8-len(vis_images)])
                    batch_images = torch.stack([img.to(self.device) for img in images[:8-len(vis_preds)]])
                    batch_preds = model(batch_images, None)
                    vis_preds.extend(batch_preds)
                    for boxes_per_image in boxes[:8-len(vis_targets)]:
                        vis_targets.append({
                            'boxes': boxes_per_image.to(self.device),
                            'labels': torch.ones((len(boxes_per_image),), dtype=torch.int64, device=self.device)
                        })
            
            self.visualization_logger.log_images(prefix, vis_images, vis_preds, vis_targets, epoch)
        
        return metrics
        
    def calculate_epoch_metrics(self, predictions, targets):
        """Calculate comprehensive epoch-level metrics with multi-dog specific metrics"""
        total_images = len(predictions)
        correct_count = 0
        over_detections = 0
        under_detections = 0
        all_ious = []
        all_confidences = []
        total_detections = 0
        total_ground_truth = 0
        true_positives = 0
        
        # Multi-dog specific tracking
        single_dog_metrics = {
            'true_positives': 0, 'false_positives': 0, 'false_negatives': 0,
            'total_gt': 0, 'total_pred': 0
        }
        
        multi_dog_metrics = {
            'true_positives': 0, 'false_positives': 0, 'false_negatives': 0,
            'total_gt': 0, 'total_pred': 0
        }
        
        count_match_percentage = 0
        count_diff_sum = 0
        detections_per_image = []
        
        # Process batches to reduce memory usage
        batch_size = 32
        for batch_start in range(0, total_images, batch_size):
            batch_end = min(batch_start + batch_size, total_images)
            batch_preds = predictions[batch_start:batch_end]
            batch_targets = targets[batch_start:batch_end]
            
            for pred, target in zip(batch_preds, batch_targets):
                pred_boxes = pred['boxes']
                pred_scores = pred['scores']
                gt_boxes = target['boxes']
                
                # Get dog count and update multi-dog tracking
                is_multi_dog = False
                if 'dog_count' in target:
                    dog_count = target['dog_count']
                    is_multi_dog = dog_count > 1
                else:
                    dog_count = len(gt_boxes)
                    is_multi_dog = dog_count > 1
                
                num_pred = len(pred_boxes)
                num_gt = len(gt_boxes)
                
                metrics_tracker = multi_dog_metrics if is_multi_dog else single_dog_metrics
                metrics_tracker['total_gt'] += num_gt
                metrics_tracker['total_pred'] += num_pred
                
                detections_per_image.append(num_pred)
                total_detections += num_pred
                total_ground_truth += num_gt
                
                # Track count differences
                count_diff_sum += abs(num_pred - num_gt)
                if num_pred == num_gt:
                    correct_count += 1
                    count_match_percentage += 1
                elif num_pred > num_gt:
                    over_detections += 1
                else:
                    under_detections += 1
                
                # Collect confidence scores
                if len(pred_scores) > 0:
                    all_confidences.extend(pred_scores.cpu().tolist())
                
                # Calculate IoUs efficiently using vectorized operations
                if num_gt > 0 and num_pred > 0:
                    # Move tensors to the same device if needed
                    if pred_boxes.device != gt_boxes.device:
                        gt_boxes = gt_boxes.to(pred_boxes.device)
                    
                    # Vectorized IoU calculation
                    ious = self._calculate_box_iou(pred_boxes, gt_boxes)  # Shape: (num_pred, num_gt)
                    
                    # Get max IoU for each GT box
                    max_ious_per_gt, matched_pred_indices = ious.max(dim=0)
                    all_ious.extend(max_ious_per_gt.cpu().tolist())
                    
                    # Count true positives
                    image_true_positives = (max_ious_per_gt > 0.5).sum().item()
                    true_positives += image_true_positives
                    metrics_tracker['true_positives'] += image_true_positives
                    
                    # Count false positives efficiently
                    matched_with_high_iou = (max_ious_per_gt > 0.5).sum().item()
                    metrics_tracker['false_positives'] += (num_pred - matched_with_high_iou)
                    
                    # Count false negatives
                    metrics_tracker['false_negatives'] += (num_gt - image_true_positives)
                elif num_gt > 0:
                    # If no predictions but we have GT boxes
                    all_ious.extend([0.0] * num_gt)
                    metrics_tracker['false_negatives'] += num_gt

        # Convert metrics to tensors and calculate final values
        iou_distribution = torch.tensor(all_ious) if all_ious else torch.zeros(0)
        confidence_distribution = torch.tensor(all_confidences) if all_confidences else torch.zeros(0)
        detections_per_image = torch.tensor(detections_per_image)
        
        # Calculate final metrics using numpy for efficiency
        all_ious_np = np.array(all_ious) if all_ious else np.array([0])
        all_confidences_np = np.array(all_confidences) if all_confidences else np.array([0])
        
        metrics = {
            'correct_count_percent': (correct_count / total_images) * 100 if total_images > 0 else 0,
            'count_match_percentage': (count_match_percentage / total_images) * 100 if total_images > 0 else 0,
            'avg_count_diff': count_diff_sum / total_images if total_images > 0 else 0,
            'over_detections': over_detections,
            'under_detections': under_detections,
            'mean_iou': float(all_ious_np.mean()) if len(all_ious_np) > 0 else 0,
            'median_iou': float(np.median(all_ious_np)) if len(all_ious_np) > 0 else 0,
            'mean_confidence': float(all_confidences_np.mean()) if len(all_confidences_np) > 0 else 0,
            'median_confidence': float(np.median(all_confidences_np)) if len(all_confidences_np) > 0 else 0,
            'avg_detections': total_detections / total_images if total_images > 0 else 0,
            'avg_ground_truth': total_ground_truth / total_images if total_images > 0 else 0,
            'precision': true_positives / total_detections if total_detections > 0 else 0,
            'recall': true_positives / total_ground_truth if total_ground_truth > 0 else 0,
            'detections_per_image': detections_per_image,
            'iou_distribution': iou_distribution,
            'confidence_distribution': confidence_distribution,
        }
        
        # Calculate F1 after precision and recall
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
        
        # Calculate single-dog and multi-dog metrics
        for prefix, tracker in [('single_dog', single_dog_metrics), ('multi_dog', multi_dog_metrics)]:
            total_pred = tracker['total_pred']
            total_gt = tracker['total_gt']
            metrics[f'{prefix}_precision'] = tracker['true_positives'] / total_pred if total_pred > 0 else 0
            metrics[f'{prefix}_recall'] = tracker['true_positives'] / total_gt if total_gt > 0 else 0
        
        # Update tracking metrics
        for metric_type in ['precision', 'recall']:
            self.multi_dog_metrics[f'single_dog_{metric_type}'].append(metrics[f'single_dog_{metric_type}'])
            self.multi_dog_metrics[f'multi_dog_{metric_type}'].append(metrics[f'multi_dog_{metric_type}'])
        
        # Add the detailed trackers to the metrics
        metrics['single_dog_metrics'] = single_dog_metrics
        metrics['multi_dog_metrics'] = multi_dog_metrics
        
        return metrics
        
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
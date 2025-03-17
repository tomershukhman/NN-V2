"""
Tensorboard logging utilities for tracking training metrics, losses, and visualizations.
"""
import os
import torch
import numpy as np
import cv2
from torch.utils.tensorboard import SummaryWriter
import csv
from datetime import datetime

class VisualizationLogger:
    def __init__(self, log_dir):
        """Initialize the visualization logger

        Args:
            log_dir (str): Directory to save tensorboard logs
        """
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        
        # Set up CSV logging
        self.csv_path = os.path.join(log_dir, "../metrics_log.csv")
        self.csv_file = None
        self.csv_writer = None
        self.header_written = False
        
        # Store metrics history
        self.history = {}

    def log_loss(self, loss, step, prefix='train'):
        """Log loss values

        Args:
            loss (float): Loss value to log
            step (int): Current training step
            prefix (str): Prefix for the loss tag (e.g., 'train' or 'val')
        """
        self.writer.add_scalar(f'{prefix}/loss', loss, step)
        
        # Also log to CSV
        self._log_to_csv({f'{prefix}_loss': loss}, step)
        
        # Store in history
        key = f"{prefix}_loss"
        if key not in self.history:
            self.history[key] = []
        self.history[key].append((step, loss))

    def log_metrics(self, metrics, step, prefix='train'):
        """Log evaluation metrics

        Args:
            metrics (dict): Dictionary of metric names and values
            step (int): Current training step
            prefix (str): Prefix for the metric tags
        """
        # Log to TensorBoard
        for name, value in metrics.items():
            self.writer.add_scalar(f'{prefix}/{name}', value, step)
            
            # Store in history
            key = f"{prefix}_{name}"
            if key not in self.history:
                self.history[key] = []
            self.history[key].append((step, value))
        
        # Log to CSV with prefixed keys
        prefixed_metrics = {f'{prefix}_{name}': value for name, value in metrics.items()}
        self._log_to_csv(prefixed_metrics, step)

    def log_images(self, images, boxes, scores, step, prefix='train'):
        """Log images with detected boxes

        Args:
            images (torch.Tensor): Batch of images
            boxes (list): List of detected boxes for each image
            scores (list): List of confidence scores for each image
            step (int): Current training step
            prefix (str): Prefix for the image tags
        """
        for i, (image, image_boxes, image_scores) in enumerate(zip(images, boxes, scores)):
            # Convert image from tensor to numpy
            img = image.cpu().numpy().transpose(1, 2, 0)

            # Draw boxes on image
            img_with_boxes = self._draw_boxes(img, image_boxes, image_scores)

            # Log to tensorboard
            self.writer.add_image(f'{prefix}/detection_{i}',
                                  torch.from_numpy(
                                      img_with_boxes).permute(2, 0, 1),
                                  step)

    def _draw_boxes(self, image, boxes, scores):
        """Draw bounding boxes on image

        Args:
            image (numpy.ndarray): Image to draw on
            boxes (torch.Tensor): Detected boxes
            scores (torch.Tensor): Confidence scores

        Returns:
            numpy.ndarray: Image with drawn boxes
        """
        image = (image * 255).astype(np.uint8).copy()

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{score:.2f}', (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image
    
    def _log_to_csv(self, metrics_dict, step):
        """Log metrics to CSV file
        
        Args:
            metrics_dict (dict): Dictionary of metric names and values
            step (int): Current step or epoch
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        
        # Add step/epoch to metrics
        metrics_dict['step'] = step
        
        # Open file if not already open
        if self.csv_file is None:
            is_new_file = not os.path.exists(self.csv_path)
            self.csv_file = open(self.csv_path, 'a', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=['step'] + sorted(list(set(metrics_dict.keys()) - {'step'})))
            
            # Write header if it's a new file
            if is_new_file:
                self.csv_writer.writeheader()
        
        # Write row
        self.csv_writer.writerow(metrics_dict)
        self.csv_file.flush()
        
    def display_metrics_summary(self, train_metrics, val_metrics, epoch, epoch_time=None):
        """Display a formatted summary of key metrics at the end of each train/val cycle
        
        Args:
            train_metrics (dict): Dictionary of training metrics
            val_metrics (dict): Dictionary of validation metrics
            epoch (int): Current epoch number
            epoch_time (float, optional): Time taken for the epoch in seconds
        """
        border_line = "=" * 80
        print(f"\n{border_line}")
        print(f"📊 MODEL PERFORMANCE SUMMARY - EPOCH {epoch+1} 📊")
        print(border_line)
        
        if epoch_time is not None:
            mins, secs = divmod(epoch_time, 60)
            print(f"⏱️  Epoch duration: {int(mins)}m {int(secs)}s")
            
        # Create columns for train and validation
        print("\n📈 LOSS METRICS:")
        print(f"  {'Metric':<20} {'Training':<15} {'Validation':<15} {'Δ (Change)':<15}")
        print("  " + "-" * 65)
        
        # Compare key metrics
        for metric in ['total_loss', 'cls_loss', 'reg_loss']:
            train_val = train_metrics.get(metric, 0)
            val_val = val_metrics.get(metric, 0)
            
            diff = val_val - train_val
            diff_str = f"{diff:+.4f}"
            
            # Add color indicators (using Unicode box-drawing chars)
            if metric == 'total_loss' and diff < 0:
                indicator = "✓"  # Good: validation loss lower than training (might be overfitting though)
            elif metric == 'total_loss' and diff > 0:
                indicator = "!"  # Warning: validation loss higher than training
            else:
                indicator = " "
                
            print(f"  {metric:<20} {train_val:<15.4f} {val_val:<15.4f} {diff_str:<15} {indicator}")
        
        # If we have more advanced metrics in validation
        if any(k in val_metrics for k in ['precision', 'recall', 'f1_score', 'mean_iou']):
            print("\n🎯 DETECTION METRICS:")
            for metric in ['precision', 'recall', 'f1_score', 'mean_iou']:
                if metric in val_metrics:
                    val = val_metrics[metric]
                    
                    # Add symbols for important metrics based on thresholds
                    symbol = ""
                    if metric == 'precision' or metric == 'recall' or metric == 'f1_score':
                        if val >= 0.9:
                            symbol = "🔥"  # Excellent
                        elif val >= 0.8:
                            symbol = "✓"   # Good
                        elif val >= 0.6:
                            symbol = "⚠️"  # Moderate
                        else:
                            symbol = "⚠️"  # Needs improvement
                            
                    metric_name = metric.replace('_', ' ').title()
                    print(f"  {metric_name:<20} {val:.4f}  {symbol}")
                    
        # Show statistics about predictions
        if 'mean_pred_count' in val_metrics:
            print("\n📏 MODEL STATISTICS:")
            print(f"  - Average predictions per image: {val_metrics['mean_pred_count']:.2f}")
            if 'mean_confidence' in val_metrics:
                print(f"  - Average confidence score: {val_metrics['mean_confidence']:.4f}")
            if 'true_positives' in val_metrics and 'false_positives' in val_metrics:
                tp = val_metrics['true_positives']
                fp = val_metrics['false_positives']
                print(f"  - True positives: {tp}, False positives: {fp}")
            
        # Print current time
        print(f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(border_line)
        
    def close(self):
        """Close the tensorboard writer and CSV file"""
        self.writer.close()
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
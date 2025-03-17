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
from dog_detector.visualization.image_utils import visualize_predictions
import matplotlib.pyplot as plt

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

    def log_images(self, images, boxes, scores, step, prefix='train', targets=None):
        """Log images with detected boxes using the better visualization function.

        Args:
            images (torch.Tensor): Batch of images
            boxes (list): List of detected boxes for each image
            scores (list): List of confidence scores for each image
            step (int): Current training step
            prefix (str): Prefix for the image tags
            targets (list, optional): List of target dictionaries with ground truth boxes
        """
        if targets is None:
            # Create empty targets if none provided
            targets = [{"boxes": torch.zeros((0, 4), device=images[0].device)} for _ in range(len(images))]

        for i, (image, image_boxes, image_scores, target) in enumerate(zip(images, boxes, scores, targets)):
            # Use the visualize_predictions function to get properly colored boxes
            fig = visualize_predictions(image, target, image_boxes, image_scores)
            
            # Log the figure to tensorboard
            self.writer.add_figure(f'{prefix}/detection_{i}', fig, step)
            
            # Close figure to free memory
            plt.close(fig)

    def log_best_model_images(self, model, val_loader, device, step, name="best_model"):
        """
        Generate and log validation images with the current (best) model
        
        Args:
            model (torch.nn.Module): The best model
            val_loader (DataLoader): Validation data loader
            device (torch.device): Device to run inference on
            step (int): Current training step
            name (str): Name suffix for the images
        """
        model.eval()
        
        # Get a batch of validation images
        with torch.no_grad():
            try:
                images, targets = next(iter(val_loader))
                images = torch.stack([img.to(device) for img in images])
                targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                          for k, v in t.items()} for t in targets]
                
                # Forward pass
                cls_output, reg_output, anchors = model(images)
                
                # Post-process outputs
                boxes, scores = model.post_process(cls_output, reg_output, anchors)
                
                # Log the images with properly colored boxes
                self.log_images(images, boxes, scores, step, f"best_model/{name}", targets)
                
            except Exception as e:
                print(f"Warning: Failed to log best model images: {e}")
    
    def _draw_boxes(self, image, boxes, scores):
        """
        Legacy method - Use log_images with visualize_predictions instead
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
        
        # Read existing rows and get all column names if file exists
        existing_rows = []
        all_columns = {'step'} | set(metrics_dict.keys())
        
        if os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    all_columns |= set(reader.fieldnames or [])
                    existing_rows = list(reader)
            except Exception:
                # If there's any error reading the file, we'll start fresh
                pass
        
        # Write to temporary file
        temp_path = self.csv_path + '.tmp'
        with open(temp_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(list(all_columns)))
            writer.writeheader()
            
            # Write existing rows
            for row in existing_rows:
                writer.writerow(row)
            
            # Write new row
            writer.writerow(metrics_dict)
        
        # Atomic replace
        os.replace(temp_path, self.csv_path)
        
        # Close existing file handle if open
        if self.csv_file:
            try:
                self.csv_file.close()
            except Exception:
                pass
            self.csv_file = None
            self.csv_writer = None
        
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
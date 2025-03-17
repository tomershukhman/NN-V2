"""
Tensorboard logging utilities for tracking training metrics, losses, and visualizations.
"""
import os
import torch
import numpy as np
import cv2
from torch.utils.tensorboard import SummaryWriter
import csv

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

    def close(self):
        """Close the tensorboard writer and CSV file"""
        self.writer.close()
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
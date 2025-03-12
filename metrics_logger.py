import os
import csv
import time
from datetime import datetime

class MetricsCSVLogger:
    """
    Logger that saves training and validation metrics to a single CSV file.
    This allows for easy analysis of metrics outside of TensorBoard.
    """
    def __init__(self, output_dir):
        """
        Initialize CSV logger.
        
        Args:
            output_dir (str): Directory to save the CSV file to
        """
        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, 'metrics'), exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(output_dir, 'metrics', f'metrics_{timestamp}.csv')
        
        # Define headers for our CSV file
        self.headers = [
            'phase', 'epoch', 'timestamp', 'total_loss', 'conf_loss', 'bbox_loss', 
            'correct_count_percent', 'over_detections', 'under_detections',
            'mean_iou', 'median_iou', 'mean_confidence', 'median_confidence',
            'avg_detections', 'avg_ground_truth', 'precision', 'recall', 'f1_score'
        ]
        
        # Initialize the CSV file with headers
        self._init_csv_file(self.csv_path)
        
        print(f"CSV metrics will be saved to: {self.csv_path}")

    def _init_csv_file(self, filepath):
        """Initialize a CSV file with headers"""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def log_metrics(self, phase, metrics, epoch):
        """
        Log metrics to the CSV file.
        
        Args:
            phase (str): Either 'train' or 'val'
            metrics (dict): Dictionary containing metrics
            epoch (int): Current epoch number
        """
        # Extract values from metrics dictionary
        row = [
            phase,                           # Phase (train or val)
            epoch,                           # Epoch number
            time.time(),                     # Timestamp
            metrics['total_loss'],           # Total loss
            metrics['conf_loss'],            # Confidence loss
            metrics['bbox_loss'],            # Bounding box loss
            metrics['correct_count_percent'], # Correct count percentage
            metrics['over_detections'],      # Over detections
            metrics['under_detections'],     # Under detections
            metrics['mean_iou'],             # Mean IoU
            metrics['median_iou'],           # Median IoU
            metrics['mean_confidence'],      # Mean confidence score
            metrics['median_confidence'],    # Median confidence score
            metrics['avg_detections'],       # Average detections per image
            metrics['avg_ground_truth'],     # Average ground truth boxes per image
            metrics['precision'],            # Precision
            metrics['recall'],               # Recall
            metrics['f1_score']              # F1 score
        ]
        
        # Append row to the CSV file
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
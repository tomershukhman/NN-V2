import torch
import torchvision.transforms.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import ImageDraw
from config import (
    TENSORBOARD_TRAIN_IMAGES, TENSORBOARD_VAL_IMAGES,
    CLASS_NAMES
)

class VisualizationLogger:
    def __init__(self, tensorboard_dir):
        self.writer = SummaryWriter(tensorboard_dir, max_queue=100)  # Increased buffer size
        self.denormalize = Denormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # Class-specific colors for visualization
        self.class_colors = {
            1: 'green',  # dog
            2: 'blue',   # person
        }

    def log_images(self, prefix, images, predictions, targets, epoch):
        """Log a sample of images with predictions for visual inspection"""
        num_images = TENSORBOARD_TRAIN_IMAGES if prefix == 'train' else TENSORBOARD_VAL_IMAGES
        # Process images in smaller batches to avoid memory issues
        batch_size = 10
        for start_idx in range(0, min(num_images, len(images)), batch_size):
            end_idx = min(start_idx + batch_size, num_images, len(images))
            batch_images = images[start_idx:end_idx]
            batch_preds = predictions[start_idx:end_idx]
            batch_targets = targets[start_idx:end_idx]
            
            for img_idx, (image, preds, target) in enumerate(zip(batch_images, batch_preds, batch_targets)):
                image = image.cpu()
                pred_boxes = preds['boxes'].cpu()
                pred_scores = preds['scores'].cpu()
                pred_labels = preds['labels'].cpu()
                gt_boxes = target['boxes'].cpu()
                gt_labels = target['labels'].cpu()
                
                vis_img = self.draw_boxes(image, pred_boxes, pred_scores, gt_boxes, gt_labels, pred_labels)
                self.writer.add_image(
                    f'{prefix}/Image_{start_idx + img_idx}',
                    torch.tensor(np.array(vis_img)).permute(2, 0, 1),
                    epoch
                )
            # Force a flush after each batch
            self.writer.flush()

    def log_epoch_metrics(self, phase, metrics, epoch):
        """Log comprehensive epoch-level metrics"""
        prefix = 'Train' if phase == 'train' else 'Val'
        
        # Safely log metrics with error handling
        def safe_log(name, value):
            try:
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'{prefix}/{name}', value, epoch)
            except Exception as e:
                print(f"Warning: Could not log metric {name}: {e}")

        # Loss metrics
        for loss_type in ['total_loss', 'conf_loss', 'bbox_loss']:
            if loss_type in metrics:
                safe_log(f'Loss/{loss_type.replace("_loss", "")}', metrics[loss_type])

        # Per-class metrics
        for class_idx in range(1, len(CLASS_NAMES)):
            class_name = CLASS_NAMES[class_idx]
            # Log standard metrics
            for metric in ['precision', 'recall', 'f1', 'mean_iou']:
                key = f'{class_name}_{metric}'
                if key in metrics:
                    safe_log(f'Classes/{class_name}/{metric}', metrics[key])
            
            # Log detection counts
            if f'{class_name}_total_gt' in metrics:
                safe_log(f'Classes/{class_name}/ground_truth_count', metrics[f'{class_name}_total_gt'])
            if f'{class_name}_total_pred' in metrics:
                safe_log(f'Classes/{class_name}/prediction_count', metrics[f'{class_name}_total_pred'])

        # Detection count metrics
        detection_count_metrics = {
            'OverDetections': 'over_detections',
            'UnderDetections': 'under_detections',
            'CorrectCount': 'correct_count_percent',
            'CountMatchPercentage': 'count_match_percentage',
            'AvgCountDiff': 'avg_count_diff'
        }
        for display_name, metric_key in detection_count_metrics.items():
            if metric_key in metrics:
                safe_log(f'DetectionCounts/{display_name}', metrics[metric_key])

        # Overall detection metrics
        detection_metrics = {
            'MeanIoU': 'mean_iou',
            'MedianIoU': 'median_iou',
            'MeanConfidence': 'mean_confidence',
            'MedianConfidence': 'median_confidence'
        }
        for display_name, metric_key in detection_metrics.items():
            if metric_key in metrics:
                safe_log(f'Detection/{display_name}', metrics[metric_key])
        
        # Per-image statistics
        image_stats = {
            'AvgDetectionsPerImage': 'avg_detections',
            'AvgGroundTruthPerImage': 'avg_ground_truth'
        }
        for display_name, metric_key in image_stats.items():
            if metric_key in metrics:
                safe_log(f'Statistics/{display_name}', metrics[metric_key])
        
        # Overall performance metrics
        performance_metrics = ['precision', 'recall', 'f1_score']
        for metric in performance_metrics:
            if metric in metrics:
                safe_log(f'Performance/{metric}', metrics[metric])

        # Log distributions as histograms
        distribution_metrics = {
            'DetectionsPerImage': 'detections_per_image',
            'IoUScores': 'iou_distribution',
            'ConfidenceScores': 'confidence_distribution'
        }
        for display_name, metric_key in distribution_metrics.items():
            if metric_key in metrics and isinstance(metrics[metric_key], torch.Tensor):
                if metrics[metric_key].nelement() > 0:
                    try:
                        self.writer.add_histogram(f'{prefix}/Distributions/{display_name}', 
                                               metrics[metric_key], epoch)
                    except Exception as e:
                        print(f"Warning: Could not log distribution {display_name}: {e}")

    def draw_boxes(self, image, boxes, scores=None, gt_boxes=None, gt_labels=None, pred_labels=None):
        # Denormalize the image first
        if isinstance(image, torch.Tensor):
            image = self.denormalize(image)
            image = F.to_pil_image(image.clip(0, 1))
        
        draw = ImageDraw.Draw(image)
        
        # Draw ground truth boxes with class-specific colors
        if gt_boxes is not None:
            for box, label in zip(gt_boxes, gt_labels if gt_labels is not None else [1] * len(gt_boxes)):
                box_coords = box.tolist()
                x1, y1, x2, y2 = box_coords
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                # Convert normalized coordinates to pixel coordinates
                x1, x2 = x1 * image.width, x2 * image.width
                y1, y2 = y1 * image.height, y2 * image.height
                color = self.class_colors.get(int(label), 'gray')
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                # Add class label
                draw.text([x1, y1-15], CLASS_NAMES[int(label)], fill=color)
        
        # Draw predicted boxes with different colors and confidence scores
        if boxes is not None and scores is not None:
            for idx, (box, score, label) in enumerate(zip(boxes, scores, 
                    pred_labels if pred_labels is not None else [1] * len(boxes))):
                color = self.class_colors.get(int(label), 'red')
                box_coords = box.tolist()
                x1, y1, x2, y2 = box_coords
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                # Convert normalized coordinates to pixel coordinates
                x1, x2 = x1 * image.width, x2 * image.width
                y1, y2 = y1 * image.height, y2 * image.height
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                if score is not None:
                    text = f'{CLASS_NAMES[int(label)]} {score:.2f}'
                    draw.text([x1, y1-15], text, fill=color)
        
        return image

    def close(self):
        self.writer.close()

class Denormalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, tensor):
        return tensor * self.std + self.mean
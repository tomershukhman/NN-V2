import torch
import torchvision.transforms.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import ImageDraw, ImageFont
from config import (
    TENSORBOARD_TRAIN_IMAGES, TENSORBOARD_VAL_IMAGES,
    CLASS_NAMES
)

class VisualizationLogger:
    def __init__(self, tensorboard_dir):
        self.writer = SummaryWriter(tensorboard_dir)
        self.denormalize = Denormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # Class-specific colors for visualization
        self.class_colors = {
            1: 'green',  # dog
            2: 'blue',   # person
        }

    def log_images(self, prefix, images, predictions, targets, epoch):
        """Log a sample of images with predictions for visual inspection"""
        num_images = TENSORBOARD_TRAIN_IMAGES if prefix == 'train' else TENSORBOARD_VAL_IMAGES
        for img_idx in range(min(num_images, len(images))):
            image = images[img_idx].cpu()
            pred_boxes = predictions[img_idx]['boxes'].cpu()
            pred_scores = predictions[img_idx]['scores'].cpu()
            pred_labels = predictions[img_idx]['labels'].cpu()
            gt_boxes = targets[img_idx]['boxes'].cpu()
            gt_labels = targets[img_idx]['labels'].cpu()
            
            vis_img = self.draw_boxes(image, pred_boxes, pred_scores, gt_boxes, gt_labels, pred_labels)
            self.writer.add_image(
                f'{prefix}/Image_{img_idx}',
                torch.tensor(np.array(vis_img)).permute(2, 0, 1),
                epoch
            )

    def log_epoch_metrics(self, phase, metrics, epoch):
        """Log comprehensive epoch-level metrics"""
        prefix = 'Train' if phase == 'train' else 'Val'
        
        # Loss metrics
        self.writer.add_scalar(f'{prefix}/Loss/Total', metrics['total_loss'], epoch)
        self.writer.add_scalar(f'{prefix}/Loss/Confidence', metrics['conf_loss'], epoch)
        self.writer.add_scalar(f'{prefix}/Loss/BBox', metrics['bbox_loss'], epoch)

        # Detection quality metrics per class
        for class_idx in range(1, len(CLASS_NAMES)):  # Skip background class
            class_name = CLASS_NAMES[class_idx]
            if f'{class_name}_precision' in metrics:
                self.writer.add_scalar(f'{prefix}/{class_name}/Precision', 
                                    metrics[f'{class_name}_precision'], epoch)
                self.writer.add_scalar(f'{prefix}/{class_name}/Recall', 
                                    metrics[f'{class_name}_recall'], epoch)
                self.writer.add_scalar(f'{prefix}/{class_name}/F1', 
                                    metrics[f'{class_name}_f1'], epoch)

        # Overall detection metrics
        self.writer.add_scalar(f'{prefix}/Detection/MeanIoU', metrics['mean_iou'], epoch)
        self.writer.add_scalar(f'{prefix}/Detection/MedianIoU', metrics['median_iou'], epoch)
        self.writer.add_scalar(f'{prefix}/Detection/MeanConfidence', metrics['mean_confidence'], epoch)
        self.writer.add_scalar(f'{prefix}/Detection/MedianConfidence', metrics['median_confidence'], epoch)
        
        # Per-image statistics
        self.writer.add_scalar(f'{prefix}/Statistics/AvgDetectionsPerImage', metrics['avg_detections'], epoch)
        self.writer.add_scalar(f'{prefix}/Statistics/AvgGroundTruthPerImage', metrics['avg_ground_truth'], epoch)
        
        # Overall performance metrics
        self.writer.add_scalar(f'{prefix}/Performance/Precision', metrics['precision'], epoch)
        self.writer.add_scalar(f'{prefix}/Performance/Recall', metrics['recall'], epoch)
        self.writer.add_scalar(f'{prefix}/Performance/F1Score', metrics['f1_score'], epoch)

        # Log distributions
        if metrics['detections_per_image'].nelement() > 0:
            self.writer.add_histogram(f'{prefix}/Distributions/DetectionsPerImage', 
                                    metrics['detections_per_image'], epoch)
        
        if metrics['iou_distribution'].nelement() > 0:
            self.writer.add_histogram(f'{prefix}/Distributions/IoUScores',
                                    metrics['iou_distribution'], epoch)
        
        if metrics['confidence_distribution'].nelement() > 0:
            self.writer.add_histogram(f'{prefix}/Distributions/ConfidenceScores',
                                    metrics['confidence_distribution'], epoch)

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
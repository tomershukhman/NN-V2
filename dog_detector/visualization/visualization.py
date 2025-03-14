import torch
import torchvision.transforms.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import ImageDraw
from config import (
    TENSORBOARD_TRAIN_IMAGES, TENSORBOARD_VAL_IMAGES,
    NORMALIZE_MEAN, NORMALIZE_STD
)
from ..model.utils.box_utils import coco_to_xyxy

class VisualizationLogger:
    def __init__(self, tensorboard_dir):
        self.writer = SummaryWriter(tensorboard_dir)
        self.denormalize = Denormalize(NORMALIZE_MEAN, NORMALIZE_STD)

    def log_images(self, prefix, images, predictions, targets, epoch):
        """Log a sample of images with predictions for visual inspection"""
        num_images = TENSORBOARD_TRAIN_IMAGES if 'train' in prefix else TENSORBOARD_VAL_IMAGES
        
        # Convert predictions to list format if it's a dictionary
        if isinstance(predictions, dict):
            batch_size = len(images)
            boxes = predictions.get('bbox_pred', predictions.get('boxes'))
            scores = predictions.get('conf_pred', predictions.get('scores'))
            # Apply sigmoid to logits before converting to list format
            if isinstance(scores, torch.Tensor):
                scores = torch.sigmoid(scores)
            predictions = [
                {'boxes': boxes[i], 'scores': scores[i]}
                for i in range(batch_size)
            ]
        
        # Process each image
        for img_idx in range(min(num_images, len(images))):
            image = images[img_idx].cpu()
            target = targets[img_idx]
            
            # Handle case where predictions[img_idx] might be empty or None
            if img_idx < len(predictions) and predictions[img_idx] is not None:
                pred_boxes = predictions[img_idx].get('boxes', torch.empty(0, 4))
                pred_scores = predictions[img_idx].get('scores', torch.empty(0))
                # Scores should already be probabilities at this point
            else:
                pred_boxes = torch.empty(0, 4)
                pred_scores = torch.empty(0)
            
            # Move tensors to CPU for visualization
            pred_boxes = pred_boxes.cpu()
            pred_scores = pred_scores.cpu()
            gt_boxes = target['boxes'].cpu()
            
            # Convert boxes from COCO to XYXY format if needed
            if 'format' in target and target['format'] == 'coco':
                gt_boxes = coco_to_xyxy(gt_boxes)
            if (img_idx < len(predictions) and predictions[img_idx] is not None and 
                'format' in predictions[img_idx] and predictions[img_idx]['format'] == 'coco'):
                pred_boxes = coco_to_xyxy(pred_boxes)
            
            # Draw boxes on image
            vis_img = self.draw_boxes(image, pred_boxes, pred_scores, gt_boxes)
            
            # Convert to tensor and log with consistent prefix 
            img_tensor = torch.tensor(np.array(vis_img)).permute(2, 0, 1)
            image_tag = f'{prefix}/Image_{img_idx}'
            self.writer.add_image(image_tag, img_tensor, epoch)

    def log_epoch_metrics(self, phase, metrics, epoch):
        """Log comprehensive epoch-level metrics"""
        prefix = 'Train' if phase == 'train' else 'Val'
        
        # Loss metrics
        self.writer.add_scalar(f'{prefix}/Loss/Total', metrics['total_loss'], epoch)
        self.writer.add_scalar(f'{prefix}/Loss/Confidence', metrics['conf_loss'], epoch)
        self.writer.add_scalar(f'{prefix}/Loss/BBox', metrics['bbox_loss'], epoch)
        
        # Detection quality metrics
        self.writer.add_scalar(f'{prefix}/Detection/CorrectCountPercent', metrics['correct_count_percent'], epoch)
        self.writer.add_scalar(f'{prefix}/Detection/OverDetections', metrics['over_detections'], epoch)
        self.writer.add_scalar(f'{prefix}/Detection/UnderDetections', metrics['under_detections'], epoch)
        self.writer.add_scalar(f'{prefix}/Detection/MeanIoU', metrics['mean_iou'], epoch)
        self.writer.add_scalar(f'{prefix}/Detection/MedianIoU', metrics['median_iou'], epoch)
        
        # Confidence score distribution
        self.writer.add_scalar(f'{prefix}/Confidence/MeanScore', metrics['mean_confidence'], epoch)
        self.writer.add_scalar(f'{prefix}/Confidence/MedianScore', metrics['median_confidence'], epoch)
        
        # Per-image statistics
        self.writer.add_scalar(f'{prefix}/Statistics/AvgDetectionsPerImage', metrics['avg_detections'], epoch)
        self.writer.add_scalar(f'{prefix}/Statistics/AvgGroundTruthPerImage', metrics['avg_ground_truth'], epoch)
        
        # Performance metrics
        self.writer.add_scalar(f'{prefix}/Performance/Precision', metrics['precision'], epoch)
        self.writer.add_scalar(f'{prefix}/Performance/Recall', metrics['recall'], epoch)
        self.writer.add_scalar(f'{prefix}/Performance/F1Score', metrics['f1_score'], epoch)
        
        # Log detection count distribution - only log if we have data
        detections = metrics['detections_per_image']
        if detections.numel() > 0:
            self.writer.add_histogram(f'{prefix}/Distributions/DetectionsPerImage', detections, epoch)
        
        # Log IoU distribution - only if we have IoU values
        iou_dist = metrics['iou_distribution']
        if iou_dist.numel() > 0:
            self.writer.add_histogram(f'{prefix}/Distributions/IoUScores', iou_dist, epoch)
        
        # Log confidence score distribution - only if we have confidence values
        conf_dist = metrics['confidence_distribution']
        if conf_dist.numel() > 0:
            self.writer.add_histogram(f'{prefix}/Distributions/ConfidenceScores', conf_dist, epoch)

    def draw_boxes(self, image, boxes, scores=None, gt_boxes=None):
        # Denormalize the image first
        if isinstance(image, torch.Tensor):
            image = self.denormalize(image)
            image = F.to_pil_image(image.clip(0, 1))
        
        draw = ImageDraw.Draw(image)
        
        # Draw ground truth boxes in green
        if gt_boxes is not None and len(gt_boxes) > 0:
            for box in gt_boxes:
                box_coords = box.tolist()
                x1, y1, x2, y2 = box_coords
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                # Convert normalized coordinates to pixel coordinates
                x1, x2 = x1 * image.width, x2 * image.width
                y1, y2 = y1 * image.height, y2 * image.height
                draw.rectangle([x1, y1, x2, y2], outline='green', width=2)
        
        # Draw predicted boxes with different colors and confidence scores
        colors = ['red', 'blue', 'purple', 'orange', 'brown', 'pink']
        if boxes is not None and len(boxes) > 0 and scores is not None and len(scores) > 0:
            for idx, (box, score) in enumerate(zip(boxes, scores)):
                color = colors[idx % len(colors)]
                box_coords = box.tolist()
                x1, y1, x2, y2 = box_coords
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                # Convert normalized coordinates to pixel coordinates
                x1, x2 = x1 * image.width, x2 * image.width
                y1, y2 = y1 * image.height, y2 * image.height
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                text = f'{score:.2f}'
                draw.text((x1, y1 - 10), text, fill=color)
        
        return image

    def close(self):
        self.writer.close()

class Denormalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, tensor):
        return tensor * self.std + self.mean
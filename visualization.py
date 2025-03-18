import torch
import torchvision.transforms.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import ImageDraw
from config import TENSORBOARD_TRAIN_IMAGES, TENSORBOARD_VAL_IMAGES

class VisualizationLogger:
    def __init__(self, tensorboard_dir):
        self.writer = SummaryWriter(tensorboard_dir)
        self.denormalize = Denormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def log_images(self, prefix, images, predictions, targets, epoch):
        """Log a sample of images with predictions for visual inspection"""
        num_images = TENSORBOARD_TRAIN_IMAGES if prefix == 'train' else TENSORBOARD_VAL_IMAGES
        for img_idx in range(min(num_images, len(images))):
            image = images[img_idx].cpu()
            pred_boxes = predictions[img_idx]['boxes'].cpu()
            pred_scores = predictions[img_idx]['scores'].cpu()
            gt_boxes = targets[img_idx]['boxes'].cpu()
            
            vis_img = self.draw_boxes(image, pred_boxes, pred_scores, gt_boxes)
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

        # Safely log histograms only if there's data to log
        # Log detection count distribution
        if len(metrics['detections_per_image']) > 0:
            self.writer.add_histogram(f'{prefix}/Distributions/DetectionsPerImage', 
                                    metrics['detections_per_image'], epoch)
        
        # Log IoU scores distribution
        if metrics['iou_distribution'].nelement() > 0:
            self.writer.add_histogram(f'{prefix}/Distributions/IoUScores',
                                    metrics['iou_distribution'], epoch)
        
        # Log confidence scores distribution
        if metrics['confidence_distribution'].nelement() > 0:
            self.writer.add_histogram(f'{prefix}/Distributions/ConfidenceScores',
                                    metrics['confidence_distribution'], epoch)

    def draw_boxes(self, image, boxes, scores=None, gt_boxes=None):
        # Denormalize the image first
        if isinstance(image, torch.Tensor):
            image = self.denormalize(image)
            image = F.to_pil_image(image.clip(0, 1))
        
        draw = ImageDraw.Draw(image)
        
        # Draw ground truth boxes in green
        if gt_boxes is not None:
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
        if boxes is not None and scores is not None:
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
                if score is not None:
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
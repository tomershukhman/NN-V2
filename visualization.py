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
        
        # Log basic metrics
        if 'total_loss' in metrics:
            self.writer.add_scalar(f'{prefix}/Loss/Total', metrics['total_loss'], epoch)
        if 'conf_loss' in metrics:
            self.writer.add_scalar(f'{prefix}/Loss/Confidence', metrics['conf_loss'], epoch)
        if 'bbox_loss' in metrics:
            self.writer.add_scalar(f'{prefix}/Loss/BBox', metrics['bbox_loss'], epoch)

        # Log detection statistics
        if 'detection_stats' in metrics:
            det_stats = metrics['detection_stats']
            self.writer.add_scalar(f'{prefix}/Detection/CorrectCountPercent', det_stats['correct_count_percent'], epoch)
            self.writer.add_scalar(f'{prefix}/Detection/OverDetections', det_stats['over_detections'], epoch)
            self.writer.add_scalar(f'{prefix}/Detection/UnderDetections', det_stats['under_detections'], epoch)
            self.writer.add_scalar(f'{prefix}/Statistics/AvgDetectionsPerImage', det_stats['avg_detections'], epoch)
            self.writer.add_scalar(f'{prefix}/Statistics/AvgGroundTruthPerImage', det_stats['avg_ground_truth'], epoch)

        # Log IoU statistics
        if 'iou_stats' in metrics:
            iou_stats = metrics['iou_stats']
            self.writer.add_scalar(f'{prefix}/Detection/MeanIoU', iou_stats['mean'], epoch)
            self.writer.add_scalar(f'{prefix}/Detection/MedianIoU', iou_stats['median'], epoch)
        
        # Log confidence score statistics
        if 'confidence_stats' in metrics:
            conf_stats = metrics['confidence_stats']
            self.writer.add_scalar(f'{prefix}/Confidence/MeanScore', conf_stats['mean'], epoch)
            self.writer.add_scalar(f'{prefix}/Confidence/MedianScore', conf_stats['median'], epoch)
        
        # Log performance metrics
        if 'performance' in metrics:
            perf = metrics['performance']
            self.writer.add_scalar(f'{prefix}/Performance/Precision', perf['precision'], epoch)
            self.writer.add_scalar(f'{prefix}/Performance/Recall', perf['recall'], epoch)
            self.writer.add_scalar(f'{prefix}/Performance/F1Score', perf['f1_score'], epoch)

        # Log distributions with empty tensor checks
        if 'distributions' in metrics:
            dist = metrics['distributions']
            
            # Detections per image histogram
            if 'detections_per_image' in dist:
                dets = dist['detections_per_image']
                if isinstance(dets, torch.Tensor) and dets.numel() > 0:
                    self.writer.add_histogram(f'{prefix}/Distributions/DetectionsPerImage', dets, epoch)
            
            # IoU scores histogram
            if 'iou_scores' in dist:
                ious = dist['iou_scores']
                if isinstance(ious, torch.Tensor) and ious.numel() > 0:
                    self.writer.add_histogram(f'{prefix}/Distributions/IoUScores', ious, epoch)
            
            # Confidence scores histogram
            if 'confidence_scores' in dist:
                confs = dist['confidence_scores']
                if isinstance(confs, torch.Tensor) and confs.numel() > 0:
                    self.writer.add_histogram(f'{prefix}/Distributions/ConfidenceScores', confs, epoch)

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
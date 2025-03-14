import os
import torch
from tqdm import tqdm
import numpy as np
from config import (
    DATA_ROOT, BATCH_SIZE, DEVICE,
    CONFIDENCE_THRESHOLD
)
from dog_detector.data import get_data_loaders
from dog_detector.model.model import get_model
from dog_detector.model.losses import DetectionLoss
from dog_detector.visualization.visualization import VisualizationLogger
from dog_detector.utils.metrics_logger import MetricsCSVLogger




def calculate_metrics(predictions, targets):
    """Calculate comprehensive evaluation metrics with improved error handling"""
    total_images = len(predictions)
    if total_images == 0:
        return {
            'correct_count_percent': 0,
            'over_detections': 0,
            'under_detections': 0,
            'mean_iou': 0,
            'median_iou': 0,
            'mean_confidence': 0,
            'median_confidence': 0,
            'avg_detections': 0,
            'avg_ground_truth': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'detections_per_image': torch.zeros(0),
            'iou_distribution': torch.zeros(0),
            'confidence_distribution': torch.zeros(0)
        }

    correct_count = 0
    over_detections = 0
    under_detections = 0
    all_ious = []
    all_confidences = []
    total_detections = 0
    total_ground_truth = 0
    true_positives = 0

    detections_per_image = []

    for pred, target in zip(predictions, targets):
        try:
            pred_boxes = pred['boxes']
            pred_scores = pred['scores']
            gt_boxes = target['boxes']

            # Ensure tensors are on CPU for numpy operations
            pred_boxes = pred_boxes.cpu()
            pred_scores = pred_scores.cpu()
            gt_boxes = gt_boxes.cpu()

            # Count statistics
            num_pred = len(pred_boxes)
            num_gt = len(gt_boxes)
            detections_per_image.append(num_pred)
            total_detections += num_pred
            total_ground_truth += num_gt

            # Detection count analysis
            if num_pred == num_gt:
                correct_count += 1
            elif num_pred > num_gt:
                over_detections += 1
            else:
                under_detections += 1

            # Collect confidence scores
            if len(pred_scores) > 0:
                all_confidences.extend(pred_scores.tolist())

            # Calculate IoUs and handle confidence thresholds
            if num_pred > 0 and num_gt > 0:
                # Filter predictions by confidence threshold
                conf_mask = pred_scores >= CONFIDENCE_THRESHOLD
                filtered_pred_boxes = pred_boxes[conf_mask]
                filtered_pred_scores = pred_scores[conf_mask]

                if len(filtered_pred_boxes) > 0:
                    ious = calculate_box_iou(filtered_pred_boxes, gt_boxes)
                    if len(ious) > 0:
                        # For each ground truth box, find the best matching prediction
                        max_ious, max_idx = ious.max(dim=0)
                        # Count true positives (IoU > 0.5 and confidence > threshold)
                        true_positives += (max_ious > 0.5).sum().item()
                        # Collect IoUs and confidence scores for matched predictions
                        matched_ious = max_ious[max_ious > 0.5]
                        matched_scores = filtered_pred_scores[max_idx[max_ious > 0.5]]
                        all_ious.extend(matched_ious.tolist())
                        all_confidences.extend(matched_scores.tolist())
        except Exception as e:
            print(f"Error calculating metrics for an image: {e}")
            continue

    # Convert lists to tensors for histogram logging
    iou_distribution = torch.tensor(all_ious) if all_ious else torch.zeros(0)
    confidence_distribution = torch.tensor(
        all_confidences) if all_confidences else torch.zeros(0)
    detections_per_image = torch.tensor(detections_per_image)

    # Calculate final metrics with proper error handling
    try:
        correct_count_percent = (correct_count / total_images) * 100
        avg_detections = total_detections / total_images
        avg_ground_truth = total_ground_truth / total_images

        mean_iou = np.mean(all_ious) if all_ious else 0
        median_iou = np.median(all_ious) if all_ious else 0

        mean_confidence = np.mean(all_confidences) if all_confidences else 0
        median_confidence = np.median(
            all_confidences) if all_confidences else 0

        precision = true_positives / total_detections if total_detections > 0 else 0
        recall = true_positives / total_ground_truth if total_ground_truth > 0 else 0
        f1_score = 2 * (precision * recall) / (precision +
                                               recall) if (precision + recall) > 0 else 0
    except Exception as e:
        print(f"Error calculating final metrics: {e}")
        return {
            'correct_count_percent': 0,
            'over_detections': 0,
            'under_detections': 0,
            'mean_iou': 0,
            'median_iou': 0,
            'mean_confidence': 0,
            'median_confidence': 0,
            'avg_detections': 0,
            'avg_ground_truth': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'detections_per_image': torch.zeros(0),
            'iou_distribution': torch.zeros(0),
            'confidence_distribution': torch.zeros(0)
        }

    return {
        'correct_count_percent': correct_count_percent,
        'over_detections': over_detections,
        'under_detections': under_detections,
        'mean_iou': mean_iou,
        'median_iou': median_iou,
        'mean_confidence': mean_confidence,
        'median_confidence': median_confidence,
        'avg_detections': avg_detections,
        'avg_ground_truth': avg_ground_truth,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'detections_per_image': detections_per_image,
        'iou_distribution': iou_distribution,
        'confidence_distribution': confidence_distribution
    }


def calculate_box_iou(boxes1, boxes2):
    """Calculate IoU between two sets of boxes with error handling"""
    try:
        # Convert to x1y1x2y2 format if normalized
        boxes1 = boxes1.clone()
        boxes2 = boxes2.clone()

        # Calculate intersection areas
        x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])

        intersection = torch.clamp(x2 - x1, min=0) * \
            torch.clamp(y2 - y1, min=0)

        # Calculate union areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2 - intersection

        # Add small epsilon to prevent division by zero
        return intersection / (union + 1e-6)
    except Exception as e:
        print(f"Error calculating IoU: {e}")
        return torch.zeros((len(boxes1), len(boxes2)))

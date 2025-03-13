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

def evaluate(checkpoint_path, data_root=None, batch_size=None):
    """
    Evaluate a trained model on the validation set
    
    Args:
        checkpoint_path (str): Path to the model checkpoint
        data_root (str, optional): Path to the data directory (overrides config)
        batch_size (int, optional): Batch size for evaluation (overrides config)
    
    Returns:
        dict: Evaluation metrics
    """
    # Use provided values or fallback to config
    actual_data_root = data_root if data_root is not None else DATA_ROOT
    actual_batch_size = batch_size if batch_size is not None else BATCH_SIZE
    
    print(f"Evaluating model checkpoint: {checkpoint_path}")
    
    # Load the model
    model = get_model(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load validation data
    _, val_loader = get_data_loaders(
        root=actual_data_root,
        batch_size=actual_batch_size,
        download=False  # No need to download for evaluation
    )
    
    # Setup criterion
    criterion = DetectionLoss(use_focal_loss=True).to(DEVICE)
    
    # Performance tracking
    total_loss = 0
    total_conf_loss = 0
    total_bbox_loss = 0
    all_predictions = []
    all_targets = []
    
    print(f"Running evaluation on validation dataset: {len(val_loader)} batches")
    
    # Main evaluation loop
    with torch.no_grad():
        for images, _, boxes in tqdm(val_loader):
            # Prepare inputs
            images = torch.stack([image.to(DEVICE) for image in images])
            targets = []
            for boxes_per_image in boxes:
                target = {
                    'boxes': boxes_per_image.to(DEVICE),
                    'labels': torch.ones((len(boxes_per_image),), dtype=torch.int64, device=DEVICE)
                }
                targets.append(target)
            
            # Forward pass for loss calculation
            predictions = model(images, targets)
            loss_dict = criterion(predictions, targets)
            total_loss += loss_dict['total_loss'].item()
            total_conf_loss += loss_dict['conf_loss']
            total_bbox_loss += loss_dict['bbox_loss']
            
            # Forward pass for evaluation
            inference_preds = model(images, None)
            all_predictions.extend(inference_preds)
            all_targets.extend(targets)
    
    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_targets)
    
    # Add loss metrics
    metrics.update({
        'total_loss': total_loss / len(val_loader),
        'conf_loss': total_conf_loss / len(val_loader),
        'bbox_loss': total_bbox_loss / len(val_loader)
    })
    
    # Print summary metrics
    print("\nEvaluation Results:")
    print(f"Validation Loss: {metrics['total_loss']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"Average detections per image: {metrics['avg_detections']:.2f}")
    
    # Write metrics to CSV
    results_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    csv_logger = MetricsCSVLogger(results_dir)
    csv_logger.log_metrics('evaluation', metrics, checkpoint['epoch'])
    
    # Visualize a few examples
    if len(images) > 0:
        tensorboard_dir = os.path.join(results_dir, 'evaluation')
        os.makedirs(tensorboard_dir, exist_ok=True)
        vis_logger = VisualizationLogger(tensorboard_dir)
        vis_logger.log_images('eval', images, inference_preds, targets, checkpoint['epoch'])
        vis_logger.close()
    
    return metrics

def calculate_metrics(predictions, targets):
    """Calculate comprehensive evaluation metrics"""
    total_images = len(predictions)
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
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        gt_boxes = target['boxes']
        
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
            all_confidences.extend(pred_scores.cpu().tolist())
        
        # Calculate IoUs for matched predictions
        if num_pred > 0 and num_gt > 0:
            ious = calculate_box_iou(pred_boxes, gt_boxes)
            if len(ious) > 0:
                max_ious, _ = ious.max(dim=0)
                all_ious.extend(max_ious.cpu().tolist())
                # Count true positives (IoU > 0.5)
                true_positives += (max_ious > 0.5).sum().item()
    
    # Convert lists to tensors for histogram logging
    iou_distribution = torch.tensor(all_ious) if all_ious else torch.zeros(0)
    confidence_distribution = torch.tensor(all_confidences) if all_confidences else torch.zeros(0)
    detections_per_image = torch.tensor(detections_per_image)
    
    # Calculate final metrics
    correct_count_percent = (correct_count / total_images) * 100
    avg_detections = total_detections / total_images
    avg_ground_truth = total_ground_truth / total_images
    
    # Calculate mean and median IoU
    mean_iou = np.mean(all_ious) if all_ious else 0
    median_iou = np.median(all_ious) if all_ious else 0
    
    # Calculate confidence score statistics
    mean_confidence = np.mean(all_confidences) if all_confidences else 0
    median_confidence = np.median(all_confidences) if all_confidences else 0
    
    # Calculate precision, recall, and F1 score
    precision = true_positives / total_detections if total_detections > 0 else 0
    recall = true_positives / total_ground_truth if total_ground_truth > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
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
    """Calculate IoU between two sets of boxes"""
    # Convert to x1y1x2y2 format if normalized
    boxes1 = boxes1.clone()
    boxes2 = boxes2.clone()
    
    # Calculate intersection areas
    x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Calculate union areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2 - intersection
    
    return intersection / (union + 1e-6)
import torch
import numpy as np
from utils import box_iou

class DetectionMetricsCalculator:
    @staticmethod
    def calculate_metrics(predictions, targets):
        """Calculate comprehensive epoch-level metrics"""
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
            
            num_pred = len(pred_boxes)
            num_gt = len(gt_boxes)
            detections_per_image.append(num_pred)
            total_detections += num_pred
            total_ground_truth += num_gt
            
            # Only count over/under detections if ground truth exists
            if num_gt > 0:
                if num_pred == num_gt:
                    correct_count += 1
                elif num_pred > num_gt:
                    over_detections += 1
                else:
                    under_detections += 1
            
            # Calculate IoUs and true positives with better matching
            if num_pred > 0 and num_gt > 0:
                # Calculate IoU matrix between all predictions and ground truths
                ious = box_iou(pred_boxes, gt_boxes)
                
                # For each ground truth, find best matching prediction
                max_ious, _ = ious.max(dim=0)
                all_ious.extend(max_ious.cpu().tolist())
                
                # Count true positives with stricter criteria
                matched_gt = set()
                sorted_preds = torch.argsort(pred_scores, descending=True)
                
                for pred_idx in sorted_preds:
                    gt_ious = ious[pred_idx]
                    best_gt_idx = torch.argmax(gt_ious)
                    max_iou = gt_ious[best_gt_idx]
                    
                    if max_iou >= 0.5 and best_gt_idx.item() not in matched_gt:
                        true_positives += 1
                        matched_gt.add(best_gt_idx.item())
            
            # Collect confidence scores for valid predictions
            if len(pred_scores) > 0:
                all_confidences.extend(pred_scores.cpu().tolist())

        return DetectionMetricsCalculator._compute_final_metrics(
            total_images, correct_count, over_detections, under_detections,
            all_ious, all_confidences, total_detections, total_ground_truth,
            true_positives, detections_per_image
        )

    @staticmethod
    def _compute_final_metrics(total_images, correct_count, over_detections, under_detections,
                             all_ious, all_confidences, total_detections, total_ground_truth,
                             true_positives, detections_per_image):
        """Compute final metrics from collected statistics"""
        # Convert lists to tensors for histogram logging
        iou_distribution = torch.tensor(all_ious) if all_ious else torch.zeros(0)
        confidence_distribution = torch.tensor(all_confidences) if all_confidences else torch.zeros(0)
        detections_per_image = torch.tensor(detections_per_image)
        
        metrics = {
            'detection_stats': {
                'correct_count_percent': (correct_count / total_images) * 100,
                'over_detections': over_detections,
                'under_detections': under_detections,
                'avg_detections': total_detections / total_images,
                'avg_ground_truth': total_ground_truth / total_images
            },
            'iou_stats': {
                'mean': np.mean(all_ious) if all_ious else 0,
                'median': np.median(all_ious) if all_ious else 0
            },
            'confidence_stats': {
                'mean': np.mean(all_confidences) if all_confidences else 0,
                'median': np.median(all_confidences) if all_confidences else 0
            },
            'performance': {
                'precision': true_positives / total_detections if total_detections > 0 else 0,
                'recall': true_positives / total_ground_truth if total_ground_truth > 0 else 0
            },
            'distributions': {
                'detections_per_image': detections_per_image,
                'iou_scores': iou_distribution,
                'confidence_scores': confidence_distribution
            }
        }
        
        # Calculate F1 score
        precision = metrics['performance']['precision']
        recall = metrics['performance']['recall']
        metrics['performance']['f1_score'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return metrics
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
            
            # Since we know there's at least 1 dog per image
            if num_pred == 0:
                under_detections += 1
            else:
                detections_per_image.append(num_pred)
                total_detections += num_pred
                total_ground_truth += num_gt
                
                if num_pred == num_gt:
                    correct_count += 1
                elif num_pred > num_gt:
                    over_detections += 1
                else:
                    under_detections += 1
                
                # Calculate IoUs and true positives with better matching
                ious = box_iou(pred_boxes, gt_boxes)
                
                # For each ground truth, find best matching prediction
                max_ious, _ = ious.max(dim=0)
                all_ious.extend(max_ious.cpu().tolist())
                
                # Count true positives with confidence-aware matching
                matched_gt = set()
                sorted_preds = torch.argsort(pred_scores, descending=True)
                
                for pred_idx in sorted_preds:
                    gt_ious = ious[pred_idx]
                    best_gt_idx = torch.argmax(gt_ious)
                    max_iou = gt_ious[best_gt_idx]
                    
                    # Only count high-confidence predictions
                    if max_iou >= 0.5 and best_gt_idx.item() not in matched_gt and pred_scores[pred_idx] >= 0.4:
                        true_positives += 1
                        matched_gt.add(best_gt_idx.item())
            
            # Collect confidence scores for valid predictions
            if len(pred_scores) > 0:
                all_confidences.extend(pred_scores.cpu().tolist())

        # Ensure we never divide by zero since we know total_ground_truth > 0
        total_ground_truth = max(total_ground_truth, total_images)  # At least 1 per image
        total_detections = max(total_detections, 1)  # Avoid division by zero

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
import torch
import numpy as np

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
                ious = DetectionMetricsCalculator._calculate_box_iou(pred_boxes, gt_boxes)
                if len(ious) > 0:
                    max_ious, _ = ious.max(dim=0)
                    all_ious.extend(max_ious.cpu().tolist())
                    # Count true positives (IoU > 0.5)
                    true_positives += (max_ious > 0.5).sum().item()

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

    @staticmethod
    def _calculate_box_iou(boxes1, boxes2):
        """Calculate IoU between two sets of boxes"""
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
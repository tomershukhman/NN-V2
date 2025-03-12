import torch
import torchvision.transforms.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import ImageDraw


class VisualizationLogger:
    def __init__(self, tensorboard_dir):
        self.writer = SummaryWriter(tensorboard_dir)
        self.denormalize = Denormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

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

    def log_images(self, prefix, images, predictions, targets, epoch, step):
        """Log images with bounding boxes to tensorboard"""
        # Show up to 16 images
        for img_idx in range(min(16, len(images))):
            image = images[img_idx].cpu()
            pred_boxes = predictions[img_idx]['boxes'].cpu()
            pred_scores = predictions[img_idx]['scores'].cpu()
            gt_boxes = targets[img_idx]['boxes'].cpu()
            
            vis_img = self.draw_boxes(image, pred_boxes, pred_scores, gt_boxes)
            self.writer.add_image(
                f'{prefix}/Image_{img_idx}',
                torch.tensor(np.array(vis_img)).permute(2, 0, 1),
                epoch * step
            )

            # Log per-image metrics
            if len(pred_scores) > 0:
                self.writer.add_scalar(f'{prefix}/Avg_Confidence_{img_idx}', pred_scores.mean(), epoch * step)
                self.writer.add_scalar(f'{prefix}/Num_Detections_{img_idx}', len(pred_scores), epoch * step)
            
            # Calculate and log IoU between predictions and ground truth
            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                ious = self._calculate_ious(pred_boxes, gt_boxes)
                if len(ious) > 0:
                    self.writer.add_scalar(f'{prefix}/Max_IoU_{img_idx}', ious.max(), epoch * step)
                    self.writer.add_scalar(f'{prefix}/Mean_IoU_{img_idx}', ious.mean(), epoch * step)

    def _calculate_ious(self, boxes1, boxes2):
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

    def log_train_metrics(self, loss_dict, epoch, step):
        """Log training metrics to tensorboard"""
        self.writer.add_scalar('Train/Total_Loss', loss_dict['total_loss'], epoch * step)
        self.writer.add_scalar('Train/Confidence_Loss', loss_dict['conf_loss'], epoch * step)
        self.writer.add_scalar('Train/BBox_Loss', loss_dict['bbox_loss'], epoch * step)
        
        # Add histogram of total loss if available
        if isinstance(loss_dict['total_loss'], torch.Tensor):
            self.writer.add_histogram('Train/Loss_Distribution', loss_dict['total_loss'], epoch * step)

    def log_epoch_metrics(self, train_loss, val_loss, lr, epoch):
        """Log epoch-level metrics"""
        self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
        self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
        self.writer.add_scalar('Epoch/Learning_Rate', lr, epoch)

    def log_model_stats(self, model, epoch):
        """Log model statistics"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f'Parameters/{name}', param.data, epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f'Gradients/{name}', param.grad.data, epoch)

    def close(self):
        self.writer.close()

class Denormalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, tensor):
        return tensor * self.std + self.mean
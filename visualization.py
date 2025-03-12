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

    def log_train_metrics(self, loss_dict, epoch, step):
        """Log training metrics to tensorboard"""
        self.writer.add_scalar('Train/Total_Loss', loss_dict['total_loss'], epoch * step)
        self.writer.add_scalar('Train/Confidence_Loss', loss_dict['conf_loss'], epoch * step)
        self.writer.add_scalar('Train/BBox_Loss', loss_dict['bbox_loss'], epoch * step)

    def log_images(self, prefix, images, predictions, targets, epoch, step):
        """Log images with bounding boxes to tensorboard"""
        for img_idx in range(min(3, len(images))):
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

    def log_epoch_metrics(self, train_loss, val_loss, lr, epoch):
        """Log epoch-level metrics"""
        self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
        self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
        self.writer.add_scalar('Epoch/Learning_Rate', lr, epoch)

    def close(self):
        self.writer.close()

class Denormalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, tensor):
        return tensor * self.std + self.mean
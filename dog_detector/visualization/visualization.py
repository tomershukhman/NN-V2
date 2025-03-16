import os
import torch
from torch.utils.tensorboard import SummaryWriter


class VisualizationLogger:
    def __init__(self, log_dir):
        """Initialize the visualization logger

        Args:
            log_dir (str): Directory to save tensorboard logs
        """
        self.writer = SummaryWriter(log_dir)

    def log_loss(self, loss, step, prefix='train'):
        """Log loss values

        Args:
            loss (float): Loss value to log
            step (int): Current training step
            prefix (str): Prefix for the loss tag (e.g., 'train' or 'val')
        """
        self.writer.add_scalar(f'{prefix}/loss', loss, step)

    def log_metrics(self, metrics, step, prefix='train'):
        """Log evaluation metrics

        Args:
            metrics (dict): Dictionary of metric names and values
            step (int): Current training step
            prefix (str): Prefix for the metric tags
        """
        for name, value in metrics.items():
            self.writer.add_scalar(f'{prefix}/{name}', value, step)

    def log_images(self, images, boxes, scores, step, prefix='train'):
        """Log images with detected boxes

        Args:
            images (torch.Tensor): Batch of images
            boxes (list): List of detected boxes for each image
            scores (list): List of confidence scores for each image
            step (int): Current training step
            prefix (str): Prefix for the image tags
        """
        for i, (image, image_boxes, image_scores) in enumerate(zip(images, boxes, scores)):
            # Convert image from tensor to numpy
            img = image.cpu().numpy().transpose(1, 2, 0)

            # Draw boxes on image
            img_with_boxes = self._draw_boxes(img, image_boxes, image_scores)

            # Log to tensorboard
            self.writer.add_image(f'{prefix}/detection_{i}',
                                  torch.from_numpy(
                                      img_with_boxes).permute(2, 0, 1),
                                  step)

    def _draw_boxes(self, image, boxes, scores):
        """Draw bounding boxes on image

        Args:
            image (numpy.ndarray): Image to draw on
            boxes (torch.Tensor): Detected boxes
            scores (torch.Tensor): Confidence scores

        Returns:
            numpy.ndarray: Image with drawn boxes
        """
        import cv2
        import numpy as np

        image = (image * 255).astype(np.uint8).copy()

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{score:.2f}', (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image

    def close(self):
        """Close the tensorboard writer"""
        self.writer.close()

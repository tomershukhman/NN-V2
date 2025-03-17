"""
Image visualization utilities for detection results and groundtruth annotations.
"""
import matplotlib.pyplot as plt
import torch
import numpy as np
from config import MEAN, STD, CONFIDENCE_THRESHOLD

def visualize_predictions(image, target, boxes_list, scores_list, max_boxes=10):
    """
    Creates a visualization of ground truth boxes and predictions on a single image.
    
    Args:
        image (torch.Tensor): The input image tensor [C, H, W]
        target (dict): Dictionary containing ground truth boxes
        boxes_list (torch.Tensor): Predicted bounding boxes [N, 4]
        scores_list (torch.Tensor): Confidence scores [N]
        max_boxes (int): Maximum number of boxes to display (highest confidence first)
    
    Returns:
        matplotlib.figure.Figure: The created figure with visualizations
    """
    # Make a copy of image tensor to avoid modifying the original
    img_tensor = image.detach().cpu().clone()
    
    # Ensure image is in [C, H, W] format
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0)
    
    # Convert to numpy for visualization
    img_np = img_tensor.numpy()
    
    # Properly denormalize the image
    for i in range(3):
        img_np[i] = img_np[i] * STD[i] + MEAN[i]
    
    # Transpose from [C, H, W] to [H, W, C] for matplotlib
    img_np = np.transpose(img_np, (1, 2, 0))
    
    # Ensure values are in valid range [0, 1]
    img_np = np.clip(img_np, 0, 1)
    
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(10, 10))
    
    # Display the image
    ax.imshow(img_np)
    
    # Get ground truth boxes
    if target and "boxes" in target:
        gt_boxes = target["boxes"].cpu().numpy()
        # Draw ground truth boxes in green
        for box in gt_boxes:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Draw rectangle
            rect = plt.Rectangle((x1, y1), width, height,
                               fill=False, edgecolor='lime', linewidth=2, alpha=0.8)
            ax.add_patch(rect)
    
    # Process predictions
    if isinstance(boxes_list, torch.Tensor) and boxes_list.numel() > 0:
        # Convert tensors to numpy
        boxes = boxes_list.cpu().numpy()
        scores = scores_list.cpu().numpy()
        
        # Filter by confidence threshold
        mask = scores > CONFIDENCE_THRESHOLD
        filtered_boxes = boxes[mask] if mask.any() else []
        filtered_scores = scores[mask] if mask.any() else []
        
        if len(filtered_boxes) > 0:
            # Sort by confidence (highest first)
            sort_idx = np.argsort(-filtered_scores)
            filtered_boxes = filtered_boxes[sort_idx]
            filtered_scores = filtered_scores[sort_idx]
            
            # Limit number of boxes
            filtered_boxes = filtered_boxes[:max_boxes]
            filtered_scores = filtered_scores[:max_boxes]
            
            # Draw prediction boxes with different colors based on confidence
            for i, (box, score) in enumerate(zip(filtered_boxes, filtered_scores)):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                
                # Color based on confidence
                if score > 0.8:
                    color = 'blue'
                elif score > 0.5:
                    color = 'cyan'
                else:
                    color = 'magenta'
                
                # Draw box
                rect = plt.Rectangle((x1, y1), width, height,
                                   fill=False, edgecolor=color, linewidth=2)
                ax.add_patch(rect)
                
                # Add score text with white background for readability
                text = f"{score:.2f}"
                ax.text(x1, y1-5, text, color=color, fontsize=10, 
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor=color, pad=0))
    
    # Remove axes and padding
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    return fig

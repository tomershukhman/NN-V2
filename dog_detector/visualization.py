#dog_detector/visualization.py
import matplotlib.pyplot as plt
import torch
from dog_detector.config import config


def visualize_predictions(image, target, boxes_list, scores_list):
    """
    Creates a visualization of ground truth boxes and predictions on a single image.
    Args:
        image (torch.Tensor): The input image tensor [C, H, W]
        target (dict): Dictionary containing ground truth boxes
        boxes_list (torch.Tensor): Predicted bounding boxes [N, 4]
        scores_list (torch.Tensor): Confidence scores [N]
    Returns:
        matplotlib.figure.Figure: The created figure with visualizations
    """
    import matplotlib.pyplot as plt
    
    # Convert and denormalize the image
    img_tensor = image.cpu().clone()
    # Ensure image is in [C, H, W] format
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0)
    
    # Denormalize
    for t, m, s in zip(img_tensor, config.MEAN, config.STD):
        t.mul_(s).add_(m)
    img_tensor = img_tensor.mul(255).byte()
    
    # Convert to numpy and correct format for matplotlib
    img_np = img_tensor.permute(1, 2, 0).numpy()
    
    # Create figure without margins
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0, 0, 1, 1])
    
    # Display the image
    ax.imshow(img_np)
    
    # Plot ground truth boxes in green
    gt_boxes = target["boxes"].cpu().numpy()
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        ax.add_patch(plt.Rectangle((x1, y1),
                                 x2 - x1,
                                 y2 - y1,
                                 fill=False,
                                 color="green",
                                 linewidth=2,
                                 label='Ground Truth'))
    
    # Plot predictions with color coding based on confidence
    if isinstance(boxes_list, torch.Tensor):
        boxes_np = boxes_list.cpu().numpy()
        scores_np = scores_list.cpu().numpy()
    else:
        boxes_np = boxes_list
        scores_np = scores_list
    
    # Ensure we have valid boxes
    if len(boxes_np) > 0:
        for box, score in zip(boxes_np, scores_np):
            if score > config.CONF_THRESHOLD:
                x1, y1, x2, y2 = box
                # Color code based on confidence
                if score > 0.8:
                    color = 'red'
                elif score > 0.6:
                    color = 'orange'
                else:
                    color = 'yellow'
                
                # Add box
                ax.add_patch(plt.Rectangle((x1, y1),
                                         x2 - x1,
                                         y2 - y1,
                                         fill=False,
                                         color=color,
                                         linewidth=2))
                # Add score text
                ax.text(x1, y1 - 5, f"{score:.2f}",
                       color=color,
                       fontsize=10,
                       bbox=dict(facecolor="white", alpha=0.7))
    
    # Remove axes for clean visualization
    ax.set_axis_off()
    
    return fig

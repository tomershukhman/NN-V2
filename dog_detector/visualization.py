#dog_detector/visualization.py
import matplotlib.pyplot as plt
from dog_detector.config import config


def visualize_predictions(image, targets, boxes_list, scores_list):
    """
    Creates a visualization of ground truth boxes and predictions on a single image.
    
    Args:
        image (torch.Tensor): The input image tensor.
        targets (dict): Dictionary containing ground truth boxes.
        boxes_list (torch.Tensor): Predicted bounding boxes.
        scores_list (torch.Tensor): Confidence scores for predictions.
    
    Returns:
        matplotlib.figure.Figure: The created figure with visualizations.
    """
    # Convert and denormalize the image
    img_tensor = image.cpu().clone()
    for t, m, s in zip(img_tensor, config.MEAN, config.STD):
        t.mul_(s).add_(m)
    img_tensor = img_tensor.mul(255).byte()
    
    # Create figure without any margins
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0, 0, 1, 1])
    
    # Display the image
    ax.imshow(img_tensor.permute(1, 2, 0))
    
    # (Optional) If you wish to include a title, consider adding it as an overlay
    # on the image to avoid adding extra whitespace. For example:
    # ax.text(0.5, 0.05, "Ground Truth (Green) and Predictions", 
    #         transform=ax.transAxes, fontsize=12, color="white", 
    #         ha="center", va="bottom", bbox=dict(facecolor="black", alpha=0.5))
    
    # Plot ground truth boxes in green
    for box in targets["boxes"].cpu().numpy():
        ax.add_patch(plt.Rectangle((box[0], box[1]),
                                   box[2] - box[0],
                                   box[3] - box[1],
                                   fill=False,
                                   color="green",
                                   linewidth=2,
                                   label='Ground Truth'))
    
    # Plot predictions with color coding based on confidence
    for box, score in zip(boxes_list.cpu().numpy(), scores_list.cpu().numpy()):
        if score > config.CONF_THRESHOLD:
            if score > 0.8:
                color = 'red'
            elif score > 0.6:
                color = 'orange'
            else:
                color = 'yellow'
            ax.add_patch(plt.Rectangle((box[0], box[1]),
                                       box[2] - box[0],
                                       box[3] - box[1],
                                       fill=False,
                                       color=color,
                                       linewidth=2))
            ax.text(box[0], box[1], f"{score:.2f}",
                    color=color, fontsize=10,
                    bbox=dict(facecolor="white", alpha=0.7))
    
    # Remove axes for a clean visualization without white borders
    ax.set_axis_off()
    
    return fig

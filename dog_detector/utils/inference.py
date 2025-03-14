import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from config import (
    DEVICE, NORMALIZE_MEAN, NORMALIZE_STD, IMAGE_SIZE, CONFIDENCE_THRESHOLD,
    DETECTION_COLOR_THRESHOLD, DETECTION_HIGH_CONF, DETECTION_MED_CONF,
    DETECTION_LINE_THICKNESS_FACTOR, DETECTION_FONT_SIZE, 
    DETECTION_BG_OPACITY_BASE, DETECTION_BG_OPACITY_FACTOR
)
from dog_detector.model.model import get_model
from dog_detector.model.utils.box_utils import coco_to_xyxy, xyxy_to_coco
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_model_from_checkpoint(checkpoint_path, device=None):
    if device is None:
        device = DEVICE
        
    # Get fresh model instance
    model = get_model(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Checkpoint from epoch {checkpoint['epoch']} with validation loss {checkpoint['val_loss']:.4f}")
    
    return model


def get_color(score):
    """Get color based on confidence score"""
    if score >= DETECTION_HIGH_CONF:
        return 'green'
    elif score >= DETECTION_MED_CONF:
        return 'orange'
    return 'red'

def draw_predictions(image, boxes, scores):
    """
    Draw bounding boxes and scores on the image with improved visual clarity
    
    Args:
        image (PIL.Image): Input image
        boxes (numpy.ndarray): Box coordinates in format [x1, y1, x2, y2] (normalized)
        scores (numpy.ndarray): Confidence scores
        
    Returns:
        PIL.Image: Image with drawn predictions
    """
    # Make a copy of the image to avoid modifying the original
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # Get image dimensions
    width, height = image.size
    
    # Try to use a nicer font if available
    try:
        font = ImageFont.truetype("Arial", DETECTION_FONT_SIZE)
    except:
        font = ImageFont.load_default()
    
    # Sort by confidence score to draw highest confidence boxes last (on top)
    indices = np.argsort(scores)
    boxes = boxes[indices]
    scores = scores[indices]
    
    # Draw each box
    for box, score in zip(boxes, scores):
        # Get color based on confidence
        box_color = get_color(score)
        text_color = "white"
        
        # Convert coordinates to pixels
        x1, y1, x2, y2 = box
        
        # Ensure coordinates are within image boundaries
        x1 = max(0, min(width - 1, int(x1)))
        y1 = max(0, min(height - 1, int(y1)))
        x2 = max(0, min(width - 1, int(x2)))
        y2 = max(0, min(height - 1, int(y2)))
        
        # Draw rectangle with thickness based on confidence
        thickness = max(1, min(3, int(score * DETECTION_LINE_THICKNESS_FACTOR)))
        draw.rectangle([(x1, y1), (x2, y2)], outline=box_color, width=thickness)
        
        # Draw score
        conf_percentage = int(score * 100)
        score_text = f"Dog: {conf_percentage}%"
        
        # Get text size
        text_bbox = draw.textbbox((0, 0), score_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Draw text background
        bg_opacity = int(min(230, DETECTION_BG_OPACITY_BASE + score * DETECTION_BG_OPACITY_FACTOR))
        bg_color_tuple = (0, 0, 0, bg_opacity)
        
        rect_y1 = max(0, y1 - text_height - 8)
        rect_y2 = max(text_height, y1)
        
        draw.rectangle([(x1, rect_y1), (x1 + text_width + 8, rect_y2)], 
                      fill=bg_color_tuple)
        
        # Draw confidence score text
        draw.text((x1 + 4, rect_y1 + 2), score_text, fill=text_color, font=font)
    
    return draw_image


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

def predict_on_image(image_path, checkpoint_path, device=None, confidence_threshold=None, save_output=True):
    """
    Run detection on a single image with improved confidence filtering
    
    Args:
        image_path (str): Path to the input image
        checkpoint_path (str): Path to model checkpoint
        device (torch.device, optional): Device to run inference on
        confidence_threshold (float, optional): Override default confidence threshold 
        save_output (bool): Whether to save the detection image
        
    Returns:
        tuple: (predictions, output_image)
    """
    if device is None:
        device = DEVICE
    
    # Use provided threshold or default from config
    if confidence_threshold is None:
        confidence_threshold = CONFIDENCE_THRESHOLD
    
    # Load the model
    model = load_model_from_checkpoint(checkpoint_path, device)
    
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Apply transforms using config values
    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
    ])
    
    # Transform the image and add batch dimension
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        predictions = model(img_tensor)
    
    # Get the predictions for the image
    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    
    # Convert boxes to COCO format for consistency
    boxes = xyxy_to_coco(boxes)
    
    # Filter by confidence threshold
    mask = scores >= confidence_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    
    # Convert back to XYXY for visualization
    vis_boxes = coco_to_xyxy(boxes)
    
    # Draw predictions on the image
    output_image = draw_predictions(image, vis_boxes, scores)
    
    # Print detection statistics
    print(f"Found {len(boxes)} dogs with confidence >= {confidence_threshold}")
    
    # Give user feedback on confidence score distribution
    if len(scores) > 0:
        print(f"Confidence score range: {scores.min():.3f} to {scores.max():.3f}, mean: {scores.mean():.3f}")
        
        # Count boxes in different confidence intervals using config parameters
        high_conf = (scores >= DETECTION_HIGH_CONF).sum()
        mid_conf = ((scores >= DETECTION_MED_CONF) & (scores < DETECTION_HIGH_CONF)).sum()
        low_conf = (scores < DETECTION_MED_CONF).sum()
        
        print(f"High confidence (≥{DETECTION_HIGH_CONF}): {high_conf} detections")
        print(f"Medium confidence ({DETECTION_MED_CONF}-{DETECTION_HIGH_CONF}): {mid_conf} detections")
        print(f"Low confidence (<{DETECTION_MED_CONF}): {low_conf} detections")
    
    # Save the output image if requested
    if save_output:
        output_filename = os.path.splitext(os.path.basename(image_path))[0] + "_detection.jpg"
        output_path = os.path.join(os.path.dirname(image_path), output_filename)
        output_image.save(output_path)
        print(f"Detection result saved to {output_path}")
        
        # Also display the image if in a notebook environment
        try:
            plt.figure(figsize=(12, 10))
            plt.imshow(np.array(output_image))
            plt.axis('off')
            plt.title(f"Detection Results: {len(boxes)} dogs found")
            
            # Add a subplot with confidence distribution if we have detections
            if len(scores) > 0:
                plt.figure(figsize=(8, 3))
                plt.hist(scores, bins=10, range=(0, 1.0), color='blue', alpha=0.7)
                plt.title('Confidence Score Distribution')
                plt.xlabel('Confidence')
                plt.ylabel('Count')
                plt.tight_layout()
                
            plt.show()
        except Exception as e:
            print(f"Could not display visualization: {e}")
    
    # Return both the raw predictions and the annotated image
    return pred, output_image

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

def batch_predict(image_paths, checkpoint_path, output_dir=None, device=None, confidence_threshold=None):
    """
    Run detection on a batch of images
    
    Args:
        image_paths (list): List of paths to the input images
        checkpoint_path (str): Path to the checkpoint file
        output_dir (str, optional): Directory to save output images
        device (torch.device, optional): Device to run inference on
        confidence_threshold (float, optional): Override default confidence threshold
        
    Returns:
        list: List of tuples containing (image_path, predictions, output_image)
    """
    if device is None:
        device = DEVICE
    
    # Use provided threshold or default from config
    if confidence_threshold is None:
        confidence_threshold = CONFIDENCE_THRESHOLD
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    model = load_model_from_checkpoint(checkpoint_path, device)
    
    # Create transform using config values
    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
    ])
    
    results = []
    
    # Process each image
    for image_path in image_paths:
        # Load and transform the image
        try:
            image = Image.open(image_path).convert('RGB')
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            # Run inference
            with torch.no_grad():
                predictions = model(img_tensor)
            
            # Get the predictions for the image
            pred = predictions[0]
            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            
            # Convert boxes to COCO format
            boxes = xyxy_to_coco(boxes)
            
            # Filter by confidence threshold
            mask = scores >= confidence_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            
            # Convert back to XYXY for visualization
            vis_boxes = coco_to_xyxy(boxes)
            
            # Draw predictions on the image
            output_image = draw_predictions(image, vis_boxes, scores)
            
            # Save the output image if output directory is specified
            if output_dir:
                output_filename = os.path.splitext(os.path.basename(image_path))[0] + "_detection.jpg"
                output_path = os.path.join(output_dir, output_filename)
                output_image.save(output_path)
                print(f"Detection result for {image_path} saved to {output_path}")
            
            # Append result
            results.append((image_path, pred, output_image))
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append((image_path, None, None))
    
    return results
import os
import torch
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as T
from config import DEVICE
from dog_detector.model.model import get_model
import matplotlib.pyplot as plt

def load_model_from_checkpoint(checkpoint_path, device=None):
    """
    Load a trained model from checkpoint
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        device (torch.device, optional): Device to load model to
    
    Returns:
        model: Loaded model
    """
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

def predict_on_image(image_path, checkpoint_path, device=None, confidence_threshold=0.5, save_output=True):
    """
    Run detection on a single image
    
    Args:
        image_path (str): Path to the input image
        checkpoint_path (str): Path to the checkpoint file
        device (torch.device, optional): Device to run inference on
        confidence_threshold (float): Threshold for displaying detections
        save_output (bool): Whether to save the output image
        
    Returns:
        tuple: (predictions, output_image)
    """
    if device is None:
        device = DEVICE
    
    # Load the model
    model = load_model_from_checkpoint(checkpoint_path, device)
    
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    
    # Apply transforms
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
    
    # Filter by confidence threshold
    mask = scores >= confidence_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    
    # Draw predictions on the image
    output_image = draw_predictions(image, boxes, scores)
    
    # Save the output image if requested
    if save_output:
        output_filename = os.path.splitext(os.path.basename(image_path))[0] + "_detection.jpg"
        output_path = os.path.join(os.path.dirname(image_path), output_filename)
        output_image.save(output_path)
        print(f"Detection result saved to {output_path}")
        
        # Also display the image if in a notebook environment
        try:
            plt.figure(figsize=(10, 10))
            plt.imshow(np.array(output_image))
            plt.axis('off')
            plt.title(f"Detection Results: {len(boxes)} dogs found")
            plt.show()
        except:
            pass
    
    # Return both the raw predictions and the annotated image
    return pred, output_image

def draw_predictions(image, boxes, scores):
    """
    Draw bounding boxes and scores on the image
    
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
    
    # Define colors
    box_color = "green"
    text_color = "white"
    
    # Draw each box
    for i, (box, score) in enumerate(zip(boxes, scores)):
        # Convert normalized coordinates to pixels
        x1, y1, x2, y2 = box
        x1 = int(x1 * width)
        y1 = int(y1 * height)
        x2 = int(x2 * width)
        y2 = int(y2 * height)
        
        # Ensure coordinates are within image boundaries
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        
        # Draw rectangle
        draw.rectangle([(x1, y1), (x2, y2)], outline=box_color, width=3)
        
        # Draw score
        score_text = f"Dog: {score:.2f}"
        text_width, text_height = draw.textsize(score_text)
        
        # Make sure text box doesn't go outside image
        rect_y1 = y1 - text_height - 5 if y1 > text_height + 5 else y1
        rect_y2 = rect_y1 + text_height + 5
        
        # Draw text background
        draw.rectangle([(x1, rect_y1), (x1 + text_width + 5, rect_y2)], fill=box_color)
        
        # Draw text
        draw.text((x1 + 2, rect_y1 + 2), score_text, fill=text_color)
    
    return draw_image

def batch_predict(image_paths, checkpoint_path, output_dir=None, device=None, confidence_threshold=0.5):
    """
    Run detection on a batch of images
    
    Args:
        image_paths (list): List of paths to the input images
        checkpoint_path (str): Path to the checkpoint file
        output_dir (str, optional): Directory to save output images
        device (torch.device, optional): Device to run inference on
        confidence_threshold (float): Threshold for displaying detections
        
    Returns:
        list: List of tuples containing (image_path, predictions, output_image)
    """
    if device is None:
        device = DEVICE
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    model = load_model_from_checkpoint(checkpoint_path, device)
    
    # Create transform
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
            
            # Filter by confidence threshold
            mask = scores >= confidence_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            
            # Draw predictions on the image
            output_image = draw_predictions(image, boxes, scores)
            
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
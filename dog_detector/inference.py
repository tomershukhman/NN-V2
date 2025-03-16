#dog_detector/inference.py

import os
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from dog_detector.model import DogDetector
from dog_detector.visualization import visualize_predictions
from dog_detector.config import config


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    model = DogDetector(num_classes=config.NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def process_image(img_path, transform=None):
    """Load and preprocess an image."""
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=config.MEAN, std=config.STD),
            transforms.Resize(config.IMAGE_SIZE, antialias=True)
        ])
    
    # Load image
    img_pil = Image.open(img_path).convert("RGB")
    orig_size = img_pil.size  # (width, height)
    
    # Apply transform
    img_tensor = transform(img_pil)
    
    return img_tensor, img_pil, orig_size


def detect_dogs(model, img_tensor, device, conf_threshold=None):
    """Run inference on an image."""
    conf_threshold = conf_threshold or config.CONF_THRESHOLD
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Forward pass
        cls_output, reg_output = model(img_tensor)
        
        # Get feature map size for anchor generation
        _, _, fm_height, fm_width = cls_output.shape
        
        # Generate anchors
        anchors = model.generate_anchors((fm_height, fm_width), device)
        
        # Post-process outputs to get bounding boxes and scores
        boxes, scores = model.post_process(
            cls_output, reg_output, anchors, conf_threshold=conf_threshold
        )
        
    return boxes[0], scores[0]  # Return for first (only) image


def main(args):
    # Set up device
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, device)
    print(f"Model loaded from {args.model_path}")
    
    # Process single image
    if args.image_path:
        # Load and preprocess image
        img_tensor, img_pil, orig_size = process_image(args.image_path)
        
        # Detect dogs
        boxes, scores = detect_dogs(model, img_tensor, device, args.conf_threshold)
        
        # Convert PIL image to NumPy array for visualization
        img_np = np.array(img_pil)
        
        # Scale boxes back to original image size
        if boxes.size(0) > 0:
            scale_x = orig_size[0] / img_tensor.shape[2]
            scale_y = orig_size[1] / img_tensor.shape[1]
            boxes[:, 0] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 2] *= scale_x
            boxes[:, 3] *= scale_y
        
        # Create target dict for visualization
        target = {"boxes": torch.zeros((0, 4))}  # Empty boxes for visualization
        
        # Visualize detections
        fig = visualize_predictions(transforms.ToTensor()(img_pil), target, boxes, scores)
        
        # Save or display the result
        if args.output_path:
            plt.savefig(args.output_path)
            print(f"Detection result saved to {args.output_path}")
        else:
            plt.show()
    
    # Process all images in a directory
    elif args.image_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Get all image files
        image_files = [
            f for f in os.listdir(args.image_dir)
            if os.path.isfile(os.path.join(args.image_dir, f)) and
            os.path.splitext(f)[1].lower() in config.IMAGE_EXTENSIONS
        ]
        
        print(f"Found {len(image_files)} images in {args.image_dir}")
        
        for img_file in image_files:
            img_path = os.path.join(args.image_dir, img_file)
            output_path = os.path.join(args.output_dir, img_file)
            
            # Load and preprocess image
            img_tensor, img_pil, orig_size = process_image(img_path)
            
            # Detect dogs
            boxes, scores = detect_dogs(model, img_tensor, device, args.conf_threshold)
            
            # Scale boxes back to original image size
            if boxes.size(0) > 0:
                scale_x = orig_size[0] / img_tensor.shape[2]
                scale_y = orig_size[1] / img_tensor.shape[1]
                boxes[:, 0] *= scale_x
                boxes[:, 1] *= scale_y
                boxes[:, 2] *= scale_x
                boxes[:, 3] *= scale_y
            
            # Create target dict for visualization
            target = {"boxes": torch.zeros((0, 4))}  # Empty boxes for visualization
            
            # Visualize detections
            fig = visualize_predictions(transforms.ToTensor()(img_pil), target, boxes, scores)
            plt.savefig(output_path)
            plt.close(fig)
            print(f"Detection result saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect dogs in images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image_path", type=str, help="Path to input image")
    group.add_argument("--image_dir", type=str, help="Directory containing images to process")
    parser.add_argument("--output_path", type=str, help="Path to save output image (used with --image_path)")
    parser.add_argument("--output_dir", type=str, default=config.OUTPUT_DIR, help="Directory to save output images")
    parser.add_argument("--conf_threshold", type=float, default=config.CONF_THRESHOLD, help="Confidence threshold")
    
    args = parser.parse_args()
    main(args)
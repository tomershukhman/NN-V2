import cv2
import torch
import argparse
from PIL import Image
import torchvision.transforms as transforms
from model import DogDetector
import config
from visualization import VisualizationLogger
import numpy as np
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

def load_model(model_path, device):
    """Load the trained model"""
    model = DogDetector()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])  # Extract only model weights
    model.to(device)
    model.eval()
    return model


def process_frame(frame, model, transform, device, vis_logger):
    """Process a single frame for dog detection"""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    
    # Apply transforms
    if isinstance(image, np.ndarray):
        img_np = image  # Already a NumPy array
    else:
        img_np = np.array(image)  # Convert to NumPy array

    augmented = transform(image=img_np, bboxes=[], labels=[])
    img_tensor = augmented["image"].unsqueeze(0).to(device)

    
    # Get predictions
    with torch.no_grad():
        predictions = model(img_tensor)
    
    # Create dummy target for visualization
    target = {
        'boxes': torch.tensor([]),
        'labels': torch.tensor([]),
    }
    
    # Use visualization logger to draw boxes
    vis_img = vis_logger.draw_boxes(
        Image.fromarray(img_np), 
        predictions[0]['boxes'].cpu(),
        predictions[0]['scores'].cpu(),
        None
    )
    
    # Convert back to BGR for OpenCV
    return cv2.cvtColor(np.array(vis_img), cv2.COLOR_RGB2BGR)

def main():
    parser = argparse.ArgumentParser(description="Test dog detector on video")
    parser.add_argument("--video_path", type=str, help="Path to input video file (use 0 for webcam)")
    parser.add_argument("--model_path", type=str, default="outputs/checkpoints/best_model.pth", help="Path to trained model")
    parser.add_argument("--output_path", type=str, default="output.mp4", help="Path to save output video")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
                        help="Device to run inference on (cuda/mps/cpu)")
    args = parser.parse_args()

    # Set up device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = load_model(args.model_path, device)
    
    # Initialize visualization logger
    vis_logger = VisualizationLogger("./outputs/tensorboard")
    
    # Set up transforms - same as validation
    transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    # Open video capture
    if args.video_path == "0":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create video writer
    #os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (frame_width, frame_height))

    print("Processing video...")
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame and draw detections
            processed_frame = process_frame(frame, model, transform, device, vis_logger)
            
            # Write frame
            out.write(processed_frame)

            # Display frame (press 'q' to quit)
            cv2.imshow('Dog Detection', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
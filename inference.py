import cv2
import torch
import numpy as np
from model import get_model
from config import DEVICE
import argparse
from torchvision import transforms

def load_model(checkpoint_path):
    """Load the trained model from checkpoint"""
    model = get_model(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def preprocess_frame(frame):
    """Preprocess a video frame for model input"""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image for transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transforms and add batch dimension
    img_tensor = transform(frame_rgb).unsqueeze(0).to(DEVICE)
    return img_tensor

def draw_detections(frame, predictions, threshold=0.5):
    """Draw detection boxes on the frame"""
    height, width = frame.shape[:2]
    boxes = predictions[0]['boxes']
    scores = predictions[0]['scores']
    
    for box, score in zip(boxes, scores):
        if score > threshold:
            # Get normalized coordinates
            x1, y1, x2, y2 = box.cpu().numpy()
            
            # Convert normalized coordinates to pixel coordinates properly
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)
            
            # Draw rectangle and confidence score
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{score:.2f}', (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def process_video(input_path, output_path, model, conf_threshold=0.5):
    """Process video file and save output with detections"""
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Preprocess frame
            img_tensor = preprocess_frame(frame)
            
            # Get predictions
            predictions = model(img_tensor, None)
            
            # Draw detections on frame
            processed_frame = draw_detections(frame, predictions, conf_threshold)
            
            # Write frame
            out.write(processed_frame)
            
            # Display frame (optional)
            cv2.imshow('Dog Detection', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Dog detection on video')
    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('output_video', help='Path to save output video')
    parser.add_argument('--checkpoint', default='outputs/checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Confidence threshold for detections')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.checkpoint)
    
    # Process video
    process_video(args.input_video, args.output_video, model, args.threshold)

if __name__ == '__main__':
    main()
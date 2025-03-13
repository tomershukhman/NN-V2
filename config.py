import os
import torch
# Dataset parameters
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/open-images")
OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
DATA_SET_TO_USE = 0.1  # Use 1% of available data for faster iteration
TRAIN_VAL_SPLIT = 0.8  # 80% training, 20% validation
# Training parameters
BATCH_SIZE = 16  # Reduced batch size due to more complex model
NUM_WORKERS = min(8, os.cpu_count() or 1)
LEARNING_RATE = 5e-5  # Reduced learning rate for stability
NUM_EPOCHS = 100  # Increased epochs since we have a more complex model
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# Model parameters
NUM_CLASSES = 2  # Background and Dog
FEATURE_MAP_SIZE = 7  # Size of the feature map for detection
ANCHOR_SCALES = [0.3, 0.5, 0.75, 1.0, 1.5]  # Different scales of anchor boxes
ANCHOR_RATIOS = [0.5, 0.75, 1.0, 1.5, 2.0]  # Different aspect ratios
NUM_ANCHORS_PER_CELL = 25  # 5 scales × 5 ratios
TOTAL_ANCHORS = FEATURE_MAP_SIZE * FEATURE_MAP_SIZE * NUM_ANCHORS_PER_CELL  # 1225 total anchors
# Detection parameters
IOU_THRESHOLD = 0.5  # IoU threshold for positive matches
NEG_POS_RATIO = 3  # Ratio of negative to positive examples
# Use higher thresholds for both training and validation
TRAIN_CONFIDENCE_THRESHOLD = 0.4
TRAIN_NMS_THRESHOLD = 0.45
# Inference thresholds - using slightly higher values for validation
CONFIDENCE_THRESHOLD = 0.45  # Higher threshold to reduce false positives
NMS_THRESHOLD = 0.4  # Slightly stricter NMS to avoid duplicate detections
MAX_DETECTIONS = 4  # Limit maximum detections per image
# Visualization parameters
TENSORBOARD_TRAIN_IMAGES = 20  # Number of training images to show in tensorboard
TENSORBOARD_VAL_IMAGES = 20    # Number of validation images to show in tensorboard
# CSV Metrics logging parameters
METRICS_DIR = os.path.join(OUTPUT_ROOT, 'metrics')  # Directory for storing CSV metrics
# Regularization parameters
DROPOUT_RATE = 0.2  # Dropout rate for regularization
WEIGHT_DECAY = 0.01  # Weight decay for AdamW optimizer
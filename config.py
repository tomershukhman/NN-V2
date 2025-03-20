import os
import torch

# Dataset parameters
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/open-images")
OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
DATA_SET_TO_USE = 0.1  # Use 10% of available data for faster iteration
TRAIN_VAL_SPLIT = 0.8  # 80% training, 20% validation

# Training parameters
BATCH_SIZE = 16
NUM_WORKERS = min(8, os.cpu_count() or 1)
LEARNING_RATE = 5e-5
NUM_EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Model parameters
NUM_CLASSES = 2  # Background and Dog
FEATURE_MAP_SIZE = 9  # Increased from 7 to 9 for better resolution for multiple objects

# Anchor box configuration - enhanced for better multi-scale detection
ANCHOR_SCALES = [0.25, 0.5, 1.0, 2.0]  # Added smaller scale for detecting small dogs
ANCHOR_RATIOS = [0.5, 0.75, 1.0, 1.5, 2.0]  # Added more aspect ratios for varied dog poses
NUM_ANCHORS_PER_CELL = len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)
TOTAL_ANCHORS = FEATURE_MAP_SIZE * FEATURE_MAP_SIZE * NUM_ANCHORS_PER_CELL

# Detection parameters
IOU_THRESHOLD = 0.35  # Slightly lower threshold to increase recall for multiple dogs
NEG_POS_RATIO = 3  # Ratio of negative to positive examples 

# Training thresholds
TRAIN_CONFIDENCE_THRESHOLD = 0.25  # Lower threshold to ensure more objects are detected during training
TRAIN_NMS_THRESHOLD = 0.6  # Higher NMS threshold to prevent over-suppression during training

# Inference thresholds
CONFIDENCE_THRESHOLD = 0.25  # Lower threshold to ensure boxes are detected
NMS_THRESHOLD = 0.4  # Reduced threshold to prevent merging close dogs
MAX_DETECTIONS = 30  # Increased to allow more detections per image

# Loss function parameters
BBOX_LOSS_WEIGHT = 1.0
CONF_LOSS_WEIGHT = 1.0

# Visualization parameters
TENSORBOARD_TRAIN_IMAGES = 20
TENSORBOARD_VAL_IMAGES = 20
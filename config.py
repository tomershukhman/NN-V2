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
FEATURE_MAP_SIZE = 7  # Size of the feature map for detection

# Anchor box configuration - use original settings which work
ANCHOR_SCALES = [0.5, 1.0, 2.0]
ANCHOR_RATIOS = [0.5, 1.0, 2.0]
NUM_ANCHORS_PER_CELL = len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)
TOTAL_ANCHORS = FEATURE_MAP_SIZE * FEATURE_MAP_SIZE * NUM_ANCHORS_PER_CELL

# Detection parameters
IOU_THRESHOLD = 0.4  # Threshold for box matching during training
NEG_POS_RATIO = 3  # Ratio of negative to positive examples 

# Training thresholds
TRAIN_CONFIDENCE_THRESHOLD = 0.3
TRAIN_NMS_THRESHOLD = 0.5

# Inference thresholds
CONFIDENCE_THRESHOLD = 0.3  # Lower threshold to ensure boxes are detected
NMS_THRESHOLD = 0.45  # Balanced NMS threshold
MAX_DETECTIONS = 25  # Allow more detections per image

# Loss function parameters
BBOX_LOSS_WEIGHT = 1.0
CONF_LOSS_WEIGHT = 1.0

# Visualization parameters
TENSORBOARD_TRAIN_IMAGES = 20
TENSORBOARD_VAL_IMAGES = 20
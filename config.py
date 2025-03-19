import os
import torch
from device import get_device

DEVICE = get_device()


# Dataset parameters
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/open-images")
OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
DATA_SET_TO_USE = 0.5  # Use 1% of available data for faster iteration
TRAIN_VAL_SPLIT = 0.8  # 80% training, 20% validation
MAX_DATA_SET_SIZE = 7_000

# Training parameters
BATCH_SIZE = 64  # Increased batch size for better gradient estimates
NUM_WORKERS = min(8, os.cpu_count() or 1)
LEARNING_RATE = 1e-3  # Increased from 5e-5 for faster initial learning
NUM_EPOCHS = 100  # Increased epochs since we have a more complex model
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Learning rate scheduler parameters
LR_SCHEDULER_FACTOR = 0.1  # Reduce LR by factor of 10
LR_SCHEDULER_PATIENCE = 5  # Number of epochs to wait before reducing LR
LR_SCHEDULER_MIN_LR = 1e-6  # Minimum learning rate
GRAD_CLIP_VALUE = 1.0  # Maximum gradient norm for gradient clipping

# Model parameters
NUM_CLASSES = 2  # Background and Dog
FEATURE_MAP_SIZE = 7  # Size of the feature map for detection
ANCHOR_SCALES = [0.5, 1.0, 2.0]  # Different scales of anchor boxes
ANCHOR_RATIOS = [0.5, 1.0, 2.0]  # Different aspect ratios
NUM_ANCHORS_PER_CELL = 9  # 3 scales × 3 ratios
TOTAL_ANCHORS = FEATURE_MAP_SIZE * FEATURE_MAP_SIZE * NUM_ANCHORS_PER_CELL  # 441 total anchors

# Detection parameters
IOU_THRESHOLD = 0.5  # IoU threshold for positive matches
NEG_POS_RATIO = 3  # Ratio of negative to positive examples

# Training thresholds (more permissive)
TRAIN_CONFIDENCE_THRESHOLD = 0.3
TRAIN_NMS_THRESHOLD = 0.45

# Inference thresholds (more strict)
CONFIDENCE_THRESHOLD = 0.5  # Increased to reduce false positives
NMS_THRESHOLD = 0.3  # Reduced to prevent duplicate detections
MAX_DETECTIONS = 20  # Reduced maximum detections per image

# Visualization parameters
TENSORBOARD_TRAIN_IMAGES = 20  # Number of training images to show in tensorboard
TENSORBOARD_VAL_IMAGES = 20    # Number of validation images to show in tensorboard

if DEVICE == "cuda":
    DATA_SET_TO_USE = 1.0  # Use full dataset when training on GPU
    BATCH_SIZE = 64  # Increase batch size for GPU training
import os
import torch
from device import get_device

# Dataset configuration
DATA_ROOT = os.path.join(os.path.dirname(__file__), 'data', 'open-images')
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), 'outputs')
DATA_SET_TO_USE = 1  # Use 10% of available data for faster iteration
TRAIN_VAL_SPLIT = 0.85  # Slightly increased training data proportion

DEVICE = get_device()

# Training configuration
BATCH_SIZE = 32     # Increased to see more examples per batch
NUM_WORKERS = 4
LEARNING_RATE = 5e-5  # Slightly reduced for stability with larger batch
NUM_EPOCHS = 100
WEIGHT_DECAY = 0.02  # Increased slightly for better regularization

# Early stopping
PATIENCE = 20       # Increased to account for class imbalance learning
MIN_DELTA = 1e-4    # Threshold for improvement

# Data augmentation parameters
MIN_SCALE = 0.8  # Increased minimum scale
MAX_SCALE = 1.2  # Reduced maximum scale
ROTATION_MAX = 15  # Reduced rotation range
TRANSLATION_FRAC = 0.1  # Reduced translation

# Model configuration
NUM_CLASSES = 3  # Background (0), Person (1), Dog (2)
CLASS_NAMES = ["background", "person", "dog"]
FEATURE_MAP_SIZE = 7  # Size of the feature map for detection

# Anchor box configuration - optimized based on size statistics
ANCHOR_SCALES = [0.07, 0.14, 0.28, 0.42]  # Covers both person (9.3%) and dog (17.67%) sizes
ANCHOR_RATIOS = [0.5, 0.75, 1.0, 1.5, 2.0]  # Extended range for varied poses

NUM_ANCHORS_PER_CELL = len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)
TOTAL_ANCHORS = FEATURE_MAP_SIZE * FEATURE_MAP_SIZE * NUM_ANCHORS_PER_CELL

# Detection parameters
IOU_THRESHOLD = 0.4  # Slightly reduced to catch more valid detections
NEG_POS_RATIO = 5  # Increased from 3 to handle class imbalance better

# Training thresholds - adjusted for better confidence calibration
TRAIN_CONFIDENCE_THRESHOLD = 0.25  # Lowered further to allow more predictions during training
TRAIN_NMS_THRESHOLD = 0.6  # Increased for better multi-dog detection during training

# Inference thresholds - fine-tuned for production
CONFIDENCE_THRESHOLD = 0.2  # Lowered to catch more valid detections in multi-dog scenarios
NMS_THRESHOLD = 0.5  # Increased to better handle overlapping dogs
MAX_DETECTIONS = 8  # Increased from 5 to allow more detections per image

# Detection parameters for each class
CLASS_CONFIDENCE_THRESHOLDS = {
    "person": 0.4,   # Higher threshold for common class (94.1%)
    "dog": 0.25      # Lower threshold for rare class (19.1%)
}

CLASS_NMS_THRESHOLDS = {
    "person": 0.5,   # Standard NMS for common class
    "dog": 0.4       # More permissive NMS for rare class
}

# Adjusted max detections based on statistics
CLASS_MAX_DETECTIONS = {
    "person": 12,    # ~52.3% have multiple people
    "dog": 3         # ~5% have multiple dogs
}

# Loss function parameters
BBOX_LOSS_WEIGHT = 1.0
CONF_LOSS_WEIGHT = 1.2  # Slightly increased to emphasize confidence accuracy

# Visualization parameters
TENSORBOARD_TRAIN_IMAGES = 8
TENSORBOARD_VAL_IMAGES = 16  # Increased from 20 to show more validation images

if DEVICE == torch.device("cuda"):
    DATA_SET_TO_USE = 1.0
    #BATCH_SIZE = 32  # Reduced from 64 for more stable training
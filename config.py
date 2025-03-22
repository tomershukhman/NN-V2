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
FEATURE_MAP_SIZE = 20  # Adjusted for better detection of smaller objects

# Anchor box configuration - optimized for class statistics
# Person: avg 9.30% of image area -> sqrt(0.093) ≈ 0.305 relative size
# Dog: avg 17.67% of image area -> sqrt(0.1767) ≈ 0.42 relative size
ANCHOR_SCALES = [
    0.2,   # For small people
    0.305, # For average people
    0.42,  # For average dogs
    0.6    # For large dogs/groups
]

# Wider range of aspect ratios to handle varying poses
ANCHOR_RATIOS = [0.5, 0.75, 1.0, 1.5, 2.0]

NUM_ANCHORS_PER_CELL = len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)
TOTAL_ANCHORS = FEATURE_MAP_SIZE * FEATURE_MAP_SIZE * NUM_ANCHORS_PER_CELL

# Detection parameters
IOU_THRESHOLD = 0.4  # Slightly reduced to catch more valid detections
NEG_POS_RATIO = 5  # Increased from 3 to handle class imbalance better

# Training thresholds - adjusted for better confidence calibration
TRAIN_CONFIDENCE_THRESHOLD = 0.2  # Lower to catch more rare class instances
TRAIN_NMS_THRESHOLD = 0.5  # Balance between multi-instance detection and false positives

# Inference thresholds - fine-tuned for production
CONFIDENCE_THRESHOLD = 0.2  # Lowered to catch more valid detections in multi-dog scenarios
NMS_THRESHOLD = 0.5  # Increased to better handle overlapping dogs
MAX_DETECTIONS = 8  # Increased from 5 to allow more detections per image

# Detection parameters for each class
CLASS_CONFIDENCE_THRESHOLDS = {
    "person": 0.3,    # More permissive for common class
    "dog": 0.2        # More permissive for rare class
}

CLASS_NMS_THRESHOLDS = {
    "person": 0.45,   # Stricter for crowded scenes (52.3% have multiple)
    "dog": 0.35       # More permissive for rare multiple dogs (5.0%)
}

# Adjusted max detections based on statistics
CLASS_MAX_DETECTIONS = {
    "person": 12,     # Higher limit for crowded scenes
    "dog": 3          # Lower limit based on statistics
}

# Loss function parameters
BBOX_LOSS_WEIGHT = 1.0
CONF_LOSS_WEIGHT = 1.2  # Slightly higher to focus on classification accuracy

# Visualization parameters
TENSORBOARD_TRAIN_IMAGES = 8
TENSORBOARD_VAL_IMAGES = 16  # Increased from 20 to show more validation images

if DEVICE == torch.device("cuda"):
    DATA_SET_TO_USE = 1.0
    #BATCH_SIZE = 32  # Reduced from 64 for more stable training
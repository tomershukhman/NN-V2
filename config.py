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
BATCH_SIZE = 16
NUM_WORKERS = 4
LEARNING_RATE = 1e-4  # Slightly reduced
NUM_EPOCHS = 100
WEIGHT_DECAY = 0.01  # Added L2 regularization

# Early stopping
PATIENCE = 15  # Increased from 10
MIN_DELTA = 1e-4  # Reduced sensitivity

# Data augmentation parameters
MIN_SCALE = 0.8  # Increased minimum scale
MAX_SCALE = 1.2  # Reduced maximum scale
ROTATION_MAX = 15  # Reduced rotation range
TRANSLATION_FRAC = 0.1  # Reduced translation

# Model configuration
NUM_CLASSES = 3  # Background (0), Dog (1), and Person (2)
CLASS_NAMES = ["background", "dog", "person"]
FEATURE_MAP_SIZE = 7  # Size of the feature map for detection

# Anchor box configuration - expanded to better handle multi-dog cases
ANCHOR_SCALES = [0.1, 0.2, 0.4]  # Small to medium objects
ANCHOR_RATIOS = [0.5, 1.0, 2.0]  # Handle different aspect ratios
NUM_ANCHORS_PER_CELL = len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)
TOTAL_ANCHORS = FEATURE_MAP_SIZE * FEATURE_MAP_SIZE * NUM_ANCHORS_PER_CELL

# Detection parameters
IOU_THRESHOLD = 0.45  # Increased from 0.4 for better box matching
NEG_POS_RATIO = 3  # Keep existing ratio

# Training thresholds - adjusted for better confidence calibration
TRAIN_CONFIDENCE_THRESHOLD = 0.25  # Lowered further to allow more predictions during training
TRAIN_NMS_THRESHOLD = 0.6  # Increased for better multi-dog detection during training

# Inference thresholds - fine-tuned for production
CONFIDENCE_THRESHOLD = 0.2  # Lowered to catch more valid detections in multi-dog scenarios
NMS_THRESHOLD = 0.5  # Increased to better handle overlapping dogs
MAX_DETECTIONS = 8  # Increased from 5 to allow more detections per image

# Detection parameters for each class
CLASS_CONFIDENCE_THRESHOLDS = {
    "dog": 0.3,      # Lower threshold for dogs to catch more instances
    "person": 0.4    # Higher threshold for persons to reduce false positives
}

CLASS_NMS_THRESHOLDS = {
    "dog": 0.45,     # More permissive NMS for dogs
    "person": 0.5    # Stricter NMS for persons
}

# Per-class maximum detections
CLASS_MAX_DETECTIONS = {
    "dog": 10,       # Allow more dog detections per image
    "person": 20     # Allow many person detections for crowd scenes
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
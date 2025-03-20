import os
import torch

# Dataset parameters
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/open-images")
OUTPUT_ROOT = "outputs"
DATA_SET_TO_USE = 0.1  # Use 10% of available data for faster iteration
TRAIN_VAL_SPLIT = 0.85  # Slightly increased training data proportion

# Training parameters
BATCH_SIZE = 16
NUM_WORKERS = min(8, os.cpu_count() or 1)
LEARNING_RATE = 1e-4  # Slightly reduced
NUM_EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
WEIGHT_DECAY = 0.01  # Added L2 regularization

# Early stopping
PATIENCE = 15  # Increased from 10
MIN_DELTA = 1e-4  # Reduced sensitivity

# Data augmentation parameters
MIN_SCALE = 0.8  # Increased minimum scale
MAX_SCALE = 1.2  # Reduced maximum scale
ROTATION_MAX = 15  # Reduced rotation range
TRANSLATION_FRAC = 0.1  # Reduced translation

# Model parameters
NUM_CLASSES = 2  # Background and Dog
FEATURE_MAP_SIZE = 7  # Size of the feature map for detection

# Anchor box configuration - use original settings which work
ANCHOR_SCALES = [0.5, 1.0, 2.0]
ANCHOR_RATIOS = [0.5, 1.0, 2.0]
NUM_ANCHORS_PER_CELL = len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)
TOTAL_ANCHORS = FEATURE_MAP_SIZE * FEATURE_MAP_SIZE * NUM_ANCHORS_PER_CELL

# Detection parameters - carefully adjusted
IOU_THRESHOLD = 0.35  # Lowered to be more lenient with matches
NEG_POS_RATIO = 2.5  # Reduced from 3 to focus more on positive examples

# Training thresholds - adjusted for better balance
TRAIN_CONFIDENCE_THRESHOLD = 0.4  # Lowered from 0.45 to catch more potential detections
TRAIN_NMS_THRESHOLD = 0.5  # Lowered from 0.6 to be more selective during training

# Inference thresholds - carefully tuned
CONFIDENCE_THRESHOLD = 0.4  # Increased from 0.35 to reduce false positives
NMS_THRESHOLD = 0.4  # Lowered from 0.45 to prevent duplicate detections
MAX_DETECTIONS = 12  # Increased from 10 to allow more potential detections

# Loss function parameters - adjusted for balance
BBOX_LOSS_WEIGHT = 1.1  # Slightly increased from 1.0
CONF_LOSS_WEIGHT = 1.0  # Kept as is

# Visualization parameters
TENSORBOARD_TRAIN_IMAGES = 20
TENSORBOARD_VAL_IMAGES = 20
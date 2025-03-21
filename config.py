import os
import torch
from device import get_device

DEVICE = get_device()

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

# Anchor box configuration - improved for multi-scale detection
ANCHOR_SCALES = [0.3, 0.5, 0.7, 1.0, 1.5]  # More scales for better coverage
ANCHOR_RATIOS = [0.5, 0.75, 1.0, 1.5, 2.0]  # More ratios for varied poses
NUM_ANCHORS_PER_CELL = len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)
TOTAL_ANCHORS = FEATURE_MAP_SIZE * FEATURE_MAP_SIZE * NUM_ANCHORS_PER_CELL

# Detection parameters - adjusted for better box matching
IOU_THRESHOLD = 0.4  # Slightly lowered for better recall
NEG_POS_RATIO = 2  # Reduced to allow more positive samples

# Training thresholds - adjusted for better confidence calibration
TRAIN_CONFIDENCE_THRESHOLD = 0.2  # Lowered to allow learning from more samples
TRAIN_NMS_THRESHOLD = 0.4  # Adjusted for training stability

# Inference thresholds - fine-tuned for multi-dog detection
CONFIDENCE_THRESHOLD = 0.25  # Lowered to detect more valid instances
NMS_THRESHOLD = 0.35  # Lowered to allow closer overlapping dogs
MAX_DETECTIONS = 8  # Increased for multi-dog scenarios

# Loss function parameters
BBOX_LOSS_WEIGHT = 1.0
CONF_LOSS_WEIGHT = 1.0

# Visualization parameters
TENSORBOARD_TRAIN_IMAGES = 20
TENSORBOARD_VAL_IMAGES = 20

if DEVICE == "cuda":
    DATA_SET_TO_USE = 1.0
    #BATCH_SIZE = 32  # Reduced from 64 for more stable training
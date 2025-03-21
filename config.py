from typing import Literal
import os
import torch
from device import get_device

# Dataset parameters
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/open-images")
OUTPUT_ROOT = "outputs"
DATA_SET_TO_USE = 0.1  # Use 10% of available data for faster iteration
TRAIN_VAL_SPLIT = 0.85  # Slightly increased training data proportion

DEVICE = get_device()

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

# Anchor box configuration - enhanced with more diverse scales and ratios
ANCHOR_SCALES = [0.25, 0.4, 0.6, 0.8, 1.2]  # Better scaled steps for diverse sizes
ANCHOR_RATIOS = [0.5, 0.75, 1.0, 1.5, 2.0]  # Enhanced ratios for different poses
NUM_ANCHORS_PER_CELL = len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)
TOTAL_ANCHORS = FEATURE_MAP_SIZE * FEATURE_MAP_SIZE * NUM_ANCHORS_PER_CELL

# Detection parameters
IOU_THRESHOLD = 0.4  # Optimal threshold for dog detection
NEG_POS_RATIO = 3  # Keep existing ratio

# Training thresholds - optimized for better confidence calibration
TRAIN_CONFIDENCE_THRESHOLD = 0.3  # Lower threshold to detect more potential dogs during training
TRAIN_NMS_THRESHOLD = 0.5  # Adjusted for better multi-dog detection during training

# Inference thresholds - optimized for better dog detection
CONFIDENCE_THRESHOLD = 0.2  # Lowered to improve dog detection recall
NMS_THRESHOLD = 0.45  # Better balance between duplicate suppression and distinct dog detection
MAX_DETECTIONS = 8  # Increased to allow more dogs in a single frame

# Loss function parameters
BBOX_LOSS_WEIGHT = 1.0
CONF_LOSS_WEIGHT = 1.2  # Slightly increased to improve confidence scores

# Visualization parameters
TENSORBOARD_TRAIN_IMAGES = 20
TENSORBOARD_VAL_IMAGES = 20

if DEVICE == torch.device("cuda"):
    DATA_SET_TO_USE = 1.0
    #BATCH_SIZE = 32  # Reduced from 64 for more stable training
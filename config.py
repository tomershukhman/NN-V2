import os
import torch
from device import get_device

DEVICE = get_device()

# Dataset parameters
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/open-images")
OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
DATA_SET_TO_USE = 0.3
TRAIN_VAL_SPLIT = 0.8

# Training parameters
BATCH_SIZE = 8  # Further reduced for more stable gradients 
NUM_WORKERS = min(8, os.cpu_count() or 1)
LEARNING_RATE = 1e-5  # Further reduced learning rate
NUM_EPOCHS = 100  # Increased to allow proper convergence
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Learning rate scheduler parameters  
LR_SCHEDULER_FACTOR = 0.2  # More aggressive LR reduction
LR_SCHEDULER_PATIENCE = 3  # Reduced patience for faster adaptation
LR_SCHEDULER_MIN_LR = 1e-7
GRAD_CLIP_VALUE = 0.25  # Further reduced for more stable training

# Model parameters
NUM_CLASSES = 2
FEATURE_MAP_SIZE = 7

# Anchor configurations optimized for dog detection
ANCHOR_SCALES = [0.3, 0.6, 0.9]  # Reduced scales to better match dog sizes
ANCHOR_RATIOS = [0.8, 1.0, 1.2]  # Tightened ratios around 1.0 for dog shapes
NUM_ANCHORS_PER_CELL = 9
TOTAL_ANCHORS = FEATURE_MAP_SIZE * FEATURE_MAP_SIZE * NUM_ANCHORS_PER_CELL

# Detection parameters - stricter matching criteria
IOU_THRESHOLD = 0.5  # Increased for more precise matching
NEG_POS_RATIO = 3   # Increased to handle class imbalance better

# Training thresholds - less permissive to reduce false positives
TRAIN_CONFIDENCE_THRESHOLD = 0.4  # Increased to reduce weak predictions
TRAIN_NMS_THRESHOLD = 0.4  # Tightened NMS during training

# Inference thresholds
CONFIDENCE_THRESHOLD = 0.6  # Increased to reduce false positives
NMS_THRESHOLD = 0.35  # Slightly increased but still strict
MAX_DETECTIONS = 2  # Reduced since ground truth averages 1.3 dogs

# Visualization parameters
TENSORBOARD_TRAIN_IMAGES = 20  # Number of training images to show in tensorboard
TENSORBOARD_VAL_IMAGES = 20    # Number of validation images to show in tensorboard

if DEVICE == "cuda":
    DATA_SET_TO_USE = 1.0
    BATCH_SIZE = 32  # Reduced from 64 for more stable training
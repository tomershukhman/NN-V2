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
BATCH_SIZE = 16  # Reduced for more stable gradients
NUM_WORKERS = min(8, os.cpu_count() or 1)
LEARNING_RATE = 5e-5  # Further reduced learning rate
NUM_EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Learning rate scheduler parameters
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_MIN_LR = 1e-6
GRAD_CLIP_VALUE = 0.5  # Reduced for more stable training

# Model parameters
NUM_CLASSES = 2
FEATURE_MAP_SIZE = 7
ANCHOR_SCALES = [0.4, 0.8, 1.2]  # Updated to match model
ANCHOR_RATIOS = [0.7, 1.0, 1.3]  # Updated to match model
NUM_ANCHORS_PER_CELL = 9
TOTAL_ANCHORS = FEATURE_MAP_SIZE * FEATURE_MAP_SIZE * NUM_ANCHORS_PER_CELL

# Detection parameters
IOU_THRESHOLD = 0.4  # Reduced to make positive matching easier
NEG_POS_RATIO = 2

# Training thresholds (more permissive)
TRAIN_CONFIDENCE_THRESHOLD = 0.3  # Reduced to allow more training signals
TRAIN_NMS_THRESHOLD = 0.5  # More permissive during training

# Inference thresholds (more strict)
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3
MAX_DETECTIONS = 3  # Further reduced since images average 1.3 dogs

# Visualization parameters
TENSORBOARD_TRAIN_IMAGES = 20  # Number of training images to show in tensorboard
TENSORBOARD_VAL_IMAGES = 20    # Number of validation images to show in tensorboard

if DEVICE == "cuda":
    DATA_SET_TO_USE = 1.0
    BATCH_SIZE = 32  # Reduced from 64 for more stable training
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

# Dynamic batch size based on available GPU memory
if torch.cuda.is_available():
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
    BATCH_SIZE = max(8, min(32, int(gpu_mem / 1.5)))  # Scale batch size with GPU memory
else:
    BATCH_SIZE = 16  # Default for CPU

# Training parameters - optimized for both GPU and CPU
NUM_WORKERS = min(8, os.cpu_count() or 1)
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
WEIGHT_DECAY = 0.01

# Early stopping
PATIENCE = 15
MIN_DELTA = 1e-4

# Data augmentation parameters
MIN_SCALE = 0.7
MAX_SCALE = 1.2
ROTATION_MAX = 15
TRANSLATION_FRAC = 0.1

# Model parameters
NUM_CLASSES = 2  # Background and Dog
FEATURE_MAP_SIZE = 7

# Anchor box configuration
ANCHOR_SCALES = [0.3, 0.5, 0.8, 1.2, 1.5]
ANCHOR_RATIOS = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
NUM_ANCHORS_PER_CELL = len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)
TOTAL_ANCHORS = FEATURE_MAP_SIZE * FEATURE_MAP_SIZE * NUM_ANCHORS_PER_CELL

# Detection parameters
IOU_THRESHOLD = 0.4
NEG_POS_RATIO = 3

# Training thresholds
TRAIN_CONFIDENCE_THRESHOLD = 0.3
TRAIN_NMS_THRESHOLD = 0.45

# Inference thresholds
CONFIDENCE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.35
MAX_DETECTIONS = 10

# Loss function parameters
BBOX_LOSS_WEIGHT = 1.2
CONF_LOSS_WEIGHT = 1.0

# Visualization parameters
TENSORBOARD_TRAIN_IMAGES = 20
TENSORBOARD_VAL_IMAGES = 20

# GPU optimization settings
if DEVICE == torch.device("cuda"):
    DATA_SET_TO_USE = 1.0
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
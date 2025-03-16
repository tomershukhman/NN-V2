import os
import torch

# Basic paths
DATA_ROOT = 'data'
OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# Dataset parameters
DATA_SET_TO_USE = 1.0  # Use 100% of dataset
TRAIN_VAL_SPLIT = 0.8
COCO_DOG_CATEGORY_ID = 18

# Training parameters
BATCH_SIZE = 16
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-4

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Model parameters
IMAGE_SIZE = 512
CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.5
MAX_DETECTIONS = 100
IOU_THRESHOLD = 0.5  # For matching predictions with ground truth

# Image normalization parameters (ImageNet stats)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Visualization parameters
TENSORBOARD_TRAIN_IMAGES = 4
TENSORBOARD_VAL_IMAGES = 4

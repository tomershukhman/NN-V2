import os
import torch

# Basic paths
DATA_ROOT = 'data'
OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# Dataset parameters
DOG_USAGE_RATIO = 0.25  # Use 10% of total available dog images
TRAIN_VAL_SPLIT = 0.8  # 80% training, 20% validation
COCO_DOG_CATEGORY_ID = 18
VISUALZIE_TOP_K = 5

# Training parameters
BATCH_SIZE = 16
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-4
REG_LOSS_WEIGHT = 1.0  # Weight for regression loss component

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Model parameters
IMAGE_SIZE = (512, 512)  # Fixed input size as width,height tuple
CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.5
MAX_DETECTIONS = 100
IOU_THRESHOLD = 0.25  # Further lowered to 0.25

# Visualization parameters
TENSORBOARD_TRAIN_IMAGES = 4
TENSORBOARD_VAL_IMAGES = 4

NUM_CLASSES = 2  # Dog and person (background is handled separately)
CLASS_NAMES = ['dog', 'person']
TRAIN_SET = "train2017"
VAL_SET = "val2017"
DOG_CATEGORY_ID = 18

MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
STD = [0.229, 0.224, 0.225]

# Model architecture settings - Better anchor scales based on dataset analysis
# GT box sizes range from ~50px to ~400px
ANCHOR_SCALES = [32, 64, 128, 192, 256, 384]  # More granular anchor scales
ANCHOR_RATIOS = [0.5, 1.0, 2.0]  # Width/height ratios to handle different aspect ratios
BACKBONE_FROZEN_LAYERS = 2

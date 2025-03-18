import os
import torch

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
if DEVICE == "cuda":
    DOG_USAGE_RATIO = 1.0
else:
    DOG_USAGE_RATIO = 0.1    

# Basic paths
DATA_ROOT = 'data'
OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "predictions")

# Dataset parameters
TRAIN_VAL_SPLIT = 0.8
COCO_DOG_CATEGORY_ID = 18
VISUALZIE_TOP_K = 5
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

# Training parameters
BATCH_SIZE = 8  # Reduced from 16 to reduce memory usage
NUM_WORKERS = 4
LEARNING_RATE = 5e-5  # Reduced learning rate for better stability
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-4
REG_LOSS_WEIGHT = 1.0  # Balanced weight for regression loss

# Loss function parameters
POS_IOU_THRESHOLD = 0.5   # Threshold for positive matches
NEG_IOU_THRESHOLD = 0.3   # Threshold for negative samples

# Model parameters
IMAGE_SIZE = (384, 384)  # Reduced from 512x512 to reduce memory usage
CONFIDENCE_THRESHOLD = 0.6  # Increased to reduce false positives
NMS_THRESHOLD = 0.3       # Helps remove overlapping boxes
MAX_DETECTIONS = 100
IOU_THRESHOLD = 0.5        # For evaluation
PRETRAINED = True  # Use pretrained backbone

# Visualization parameters
TENSORBOARD_TRAIN_IMAGES = 4
TENSORBOARD_VAL_IMAGES = 4
NUM_VAL_IMAGES_TO_LOG = 5

NUM_CLASSES = 2  # Dog and person (background is handled separately)
CLASS_NAMES = ['dog', 'person']
TRAIN_SET = "train2017"
VAL_SET = "val2017"
DOG_CATEGORY_ID = 18

MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
STD = [0.229, 0.224, 0.225]

# Model architecture settings - Optimized based on anchor analysis
# GT box sizes range from ~30px to ~500px, with median width around 139px and height 96px
# GT aspect ratios range from 0.3 to 4.0
ANCHOR_SCALES = [32, 64, 128, 256, 384]  # Key scales targeting median and extremes
ANCHOR_RATIOS = [0.5, 0.75, 1.0, 1.5, 2.0]  # Focused ratio coverage
BACKBONE_FROZEN_LAYERS = 2

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
BATCH_SIZE = 16
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-4
REG_LOSS_WEIGHT = 1.0  # Reduced since we now have proper normalization

# Loss function parameters
POS_IOU_THRESHOLD = 0.5   # Stricter threshold for positive matches
NEG_IOU_THRESHOLD = 0.3   # Better separation between positive and negative samples

# Model parameters
IMAGE_SIZE = (512, 512)  # Fixed input size as width,height tuple
CONFIDENCE_THRESHOLD = 0.5  # Stricter confidence threshold to reduce false positives
NMS_THRESHOLD = 0.3       # Lower NMS threshold to remove more overlapping boxes
MAX_DETECTIONS = 100
IOU_THRESHOLD = 0.5        # Stricter IoU threshold for evaluation
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

# Model architecture settings - Better anchor scales based on dataset analysis
# GT box sizes range from ~5px to ~400px, with median around 86px
ANCHOR_SCALES = [16, 32, 64, 128, 256]  # Better distribution across observed sizes
ANCHOR_RATIOS = [0.3, 0.7, 1.0, 1.5]    # Better coverage of observed aspect ratios
BACKBONE_FROZEN_LAYERS = 2

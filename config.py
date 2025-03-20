import os
import torch

# Dataset parameters
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/open-images")
OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
DATA_SET_TO_USE = 0.1  # Use 1% of available data for faster iteration
TRAIN_VAL_SPLIT = 0.8  # 80% training, 20% validation

# Training parameters
BATCH_SIZE = 16  # Reduced batch size due to more complex model
NUM_WORKERS = min(8, os.cpu_count() or 1)
LEARNING_RATE = 5e-5  # Reduced learning rate for stability
NUM_EPOCHS = 100  # Increased epochs since we have a more complex model
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Model parameters
NUM_CLASSES = 2  # Background and Dog
FEATURE_MAP_SIZE = 7  # Size of the feature map for detection

# Increasing anchor diversity to better capture objects at different scales
ANCHOR_SCALES = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]  # More diverse scales of anchor boxes
ANCHOR_RATIOS = [0.3, 0.5, 1.0, 2.0, 3.0]  # More diverse aspect ratios
NUM_ANCHORS_PER_CELL = len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)  # Total anchor configurations per cell
TOTAL_ANCHORS = FEATURE_MAP_SIZE * FEATURE_MAP_SIZE * NUM_ANCHORS_PER_CELL

# Detection parameters - adjusted for better count matching
IOU_THRESHOLD = 0.35  # Even lower threshold to get more positive matches during training
NEG_POS_RATIO = 3  # Ratio of negative to positive examples

# Training thresholds (very permissive to encourage detecting more objects)
TRAIN_CONFIDENCE_THRESHOLD = 0.15  # Even lower threshold to capture more potential matches
TRAIN_NMS_THRESHOLD = 0.6  # Further increased to keep more potential overlapping boxes during training

# Inference thresholds (much more permissive to address under-detection)
CONFIDENCE_THRESHOLD = 0.25  # Significantly lowered to detect more objects
NMS_THRESHOLD = 0.4  # Increased to allow more overlapping detections
MAX_DETECTIONS = 50  # Doubled to capture many more potential objects

# Loss function parameters
BBOX_LOSS_WEIGHT = 1.5  # Increase bbox regression weight to emphasize spatial accuracy
CONF_LOSS_WEIGHT = 1.0  # Base confidence weight
COUNT_LOSS_WEIGHT = 0.2  # Weight for count loss

# Visualization parameters
TENSORBOARD_TRAIN_IMAGES = 20  # Number of training images to show in tensorboard
TENSORBOARD_VAL_IMAGES = 20    # Number of validation images to show in tensorboard
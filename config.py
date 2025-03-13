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

# Training hyperparameters
GRADIENT_CLIP_VALUE = 1.0  # Maximum gradient norm for clipping
EARLY_STOPPING_PATIENCE = 15  # Number of epochs without improvement before stopping
WARMUP_STEPS_RATIO = 0.2  # Ratio of total steps to use for warmup (20%)
NUM_CYCLES = 0.5  # Number of cosine cycles in the learning rate schedule
LOG_IMAGES_INTERVAL = 50  # Log sample images every N steps
IMAGE_SAMPLES_TO_LOG = 16  # Number of validation images to log at end of epoch

# Model architecture parameters
DETECTION_HEAD_CHANNELS = 256  # Number of channels in detection head
LATERAL_CHANNELS = 256  # Number of channels in lateral connections

# Model parameters
NUM_CLASSES = 2  # Background and Dog
FEATURE_MAP_SIZE = 7  # Size of the feature map for detection
ANCHOR_SCALES = [0.3, 0.5, 0.75, 1.0, 1.5]  # Different scales of anchor boxes
ANCHOR_RATIOS = [0.5, 0.75, 1.0, 1.5, 2.0]  # Different aspect ratios
NUM_ANCHORS_PER_CELL = 25  # 5 scales × 5 ratios
TOTAL_ANCHORS = FEATURE_MAP_SIZE * FEATURE_MAP_SIZE * NUM_ANCHORS_PER_CELL  # 1225 total anchors

# Detection parameters
IOU_THRESHOLD = 0.5  # IoU threshold for positive matches
NEG_POS_RATIO = 3  # Ratio of negative to positive examples
# Use lower thresholds early in training to allow learning
TRAIN_CONFIDENCE_THRESHOLD = 0.05  # Much lower threshold during training
TRAIN_NMS_THRESHOLD = 0.45
# Inference thresholds - can be higher for better precision
CONFIDENCE_THRESHOLD = 0.3  # Lower threshold to allow more predictions
NMS_THRESHOLD = 0.4  # Slightly stricter NMS to avoid duplicate detections
MAX_DETECTIONS = 4  # Limit maximum detections per image

# Image normalization parameters
NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
NORMALIZE_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224  # Input image size

# Visualization parameters
TENSORBOARD_TRAIN_IMAGES = 20  # Number of training images to show in tensorboard
TENSORBOARD_VAL_IMAGES = 20    # Number of validation images to show in tensorboard

# CSV Metrics logging parameters
METRICS_DIR = os.path.join(OUTPUT_ROOT, 'metrics')  # Directory for storing CSV metrics

# Regularization parameters
DROPOUT_RATE = 0.2  # Dropout rate for regularization
WEIGHT_DECAY = 0.01  # Weight decay for AdamW optimizer

# Loss parameters
FOCAL_LOSS_ALPHA = 0.25  # Focal loss alpha parameter for handling class imbalance
FOCAL_LOSS_GAMMA = 2.0   # Focal loss gamma parameter for down-weighting easy examples

# Backbone parameters
BACKBONE_FROZEN_LAYERS = 30  # Number of layers to freeze in backbone
BACKBONE_LR_FACTOR = 0.1    # Learning rate factor for backbone (relative to base lr)
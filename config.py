import os
import torch

# Dataset parameters
DATA_ROOT = 'data'
DATASET_TYPE = 'coco'
DATA_SET_TO_USE = 1 
TRAIN_VAL_SPLIT = 0.8

# COCO specific parameters
COCO_YEAR = '2017'
COCO_DOG_CATEGORY_ID = 18
MIN_DOG_SCORE = 0.3  # Minimum confidence score for dog detections

OUTPUT_ROOT = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "outputs")

# Training parameters
BATCH_SIZE = 16  # Increased for better gradient estimates
NUM_WORKERS = 4
LEARNING_RATE = 1e-4  # Increased for faster initial learning
NUM_EPOCHS = 100
WEIGHT_DECAY = 1e-4  # Reduced weight decay

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Training hyperparameters
GRADIENT_CLIP_VALUE = 1.0  # Increased for better gradient flow
EARLY_STOPPING_PATIENCE = 15  # More patience for convergence
WARMUP_STEPS_RATIO = 0.1  # 10% warmup
NUM_CYCLES = 0.5  # Half cosine cycle for better final performance
IMAGE_SAMPLES_TO_LOG = 4

# Model architecture parameters
IMAGE_SIZE = 640  # Increased for better small dog detection
FEATURE_MAP_SIZE = 40  # Adjusted for new image size
NUM_CLASSES = 2  # Background and Dog
LATERAL_CHANNELS = 256
DETECTION_HEAD_CHANNELS = 256
NUM_ANCHORS_PER_CELL = 9
DROPOUT_RATE = 0.0  # Remove dropout initially
CONF_BIAS_INIT = -2.944  # ln(0.05/(1-0.05)) - less extreme initial bias

# Anchor configuration - optimized for dog detection
ANCHOR_SCALES = [0.05, 0.1, 0.2, 0.4]  # More varied scales
ANCHOR_RATIOS = [0.5, 0.7, 1.0, 1.4, 2.0]  # More varied ratios
ANCHOR_SIZES = [16, 32, 64, 128, 256]  # More varied sizes
ANCHOR_ASPECT_RATIOS = [0.5, 1.0, 2.0]  # Standard aspect ratios
TOTAL_ANCHORS = FEATURE_MAP_SIZE * FEATURE_MAP_SIZE * NUM_ANCHORS_PER_CELL

# Detection parameters
IOU_THRESHOLD = 0.4  # Reduced for easier matching during training
NEG_POS_RATIO = 3
TRAIN_CONFIDENCE_THRESHOLD = 0.01  # Lower threshold for training
TRAIN_NMS_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.1  # Lower initial confidence threshold
NMS_THRESHOLD = 0.45
MAX_DETECTIONS = 200  # Allow more detections initially

# Box filtering parameters
MIN_BOX_SIZE = 0.02
MIN_ASPECT_RATIO = 0.3
MAX_ASPECT_RATIO = 3.0

# Image normalization parameters
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Detection visualization parameters
DETECTION_COLOR_THRESHOLD = 0.4
DETECTION_HIGH_CONF = 0.7
DETECTION_MED_CONF = 0.5
DETECTION_LINE_THICKNESS_FACTOR = 4
DETECTION_FONT_SIZE = 14
DETECTION_BG_OPACITY_BASE = 120
DETECTION_BG_OPACITY_FACTOR = 110

# Visualization parameters
TENSORBOARD_TRAIN_IMAGES = 4
TENSORBOARD_VAL_IMAGES = 4

# CSV Metrics logging parameters
METRICS_DIR = os.path.join(OUTPUT_ROOT, 'metrics')

# Loss parameters
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 1.5  # Reduced to make loss less harsh initially
LOC_LOSS_WEIGHT = 2.0  # Increased importance of localization

# Backbone parameters
BACKBONE_FROZEN_LAYERS = 1  # Freeze fewer layers for better feature learning
BACKBONE_LR_FACTOR = 0.2  # Allow backbone to learn faster

# Regularization parameters
WEIGHT_DECAY = 1e-4  # Reduced weight decay

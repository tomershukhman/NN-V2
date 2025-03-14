import os
import torch
# Dataset parameters
DATA_ROOT = 'data'  # Root directory for datasets
DATASET_TYPE = 'coco'  # Use COCO dataset
DATA_SET_TO_USE = 0.3  # Use 30% of dataset initially for faster iteration
TRAIN_VAL_SPLIT = 0.8

# COCO specific parameters
COCO_YEAR = '2017'
COCO_DOG_CATEGORY_ID = 18
MIN_DOG_SCORE = 0.3  # Minimum confidence score for dog detections

OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# Training parameters
BATCH_SIZE = 8  # Reduced batch size
NUM_WORKERS = 4
LEARNING_RATE = 3e-5  # Reduced learning rate
NUM_EPOCHS = 100

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Training hyperparameters
GRADIENT_CLIP_VALUE = 5.0  # Increased for better stability
EARLY_STOPPING_PATIENCE = 10
WARMUP_STEPS_RATIO = 0.05  # Shorter warmup
NUM_CYCLES = 0.5
IMAGE_SAMPLES_TO_LOG = 8

# Model architecture parameters
IMAGE_SIZE = 416  # Increased image size for better detection
FEATURE_MAP_SIZE = 26  # Adjusted for new image size
NUM_CLASSES = 2  # Background and Dog - single consistent definition
LATERAL_CHANNELS = 256  # Increased channels for better feature representation
DETECTION_HEAD_CHANNELS = 256
NUM_ANCHORS_PER_CELL = 9  # 3 scales × 3 ratios
DROPOUT_RATE = 0.2  # Reduced dropout for better training
CONF_BIAS_INIT = -4.595  # ln(0.01/(1-0.01)) for initial confidence of 0.01

# Anchor configuration - single consistent definition
ANCHOR_SCALES = [0.2, 0.4, 0.6]  # Better coverage of small dogs
ANCHOR_RATIOS = [0.7, 1.0, 1.4]  # More appropriate for dog shapes
ANCHOR_SIZES = [32, 64, 128]  # Increased sizes for better coverage
ANCHOR_ASPECT_RATIOS = ANCHOR_RATIOS  # Use same ratios for consistency
TOTAL_ANCHORS = FEATURE_MAP_SIZE * FEATURE_MAP_SIZE * NUM_ANCHORS_PER_CELL

# Detection parameters
IOU_THRESHOLD = 0.3  # Reduced for easier positive matching
NEG_POS_RATIO = 3
TRAIN_CONFIDENCE_THRESHOLD = 0.05  # Lower threshold during training
TRAIN_NMS_THRESHOLD = 0.5  # Relaxed NMS during training
CONFIDENCE_THRESHOLD = 0.3  # Lowered for inference
NMS_THRESHOLD = 0.45
MAX_DETECTIONS = 100

# Box filtering parameters
MIN_BOX_SIZE = 0.01  # Reduced to catch smaller dogs
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
TENSORBOARD_TRAIN_IMAGES = 8
TENSORBOARD_VAL_IMAGES = 8

# CSV Metrics logging parameters
METRICS_DIR = os.path.join(OUTPUT_ROOT, 'metrics')

# Loss parameters
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.0
LOC_LOSS_WEIGHT = 1.0  # Balanced with confidence loss

# Backbone parameters
BACKBONE_FROZEN_LAYERS = 5  # Freeze fewer layers
BACKBONE_LR_FACTOR = 0.1  # Lower backbone learning rate

# Regularization parameters
WEIGHT_DECAY = 0.0001  # Increased weight decay for better regularization
# #dog_detector/config.py
# import os
# import torch

# class Config:
#     # Basic paths and device settings
#     if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#         DATA_ROOT = "./data"
#     else:
#         DATA_ROOT = "../coco/NN-V2/data"
#     OUTPUT_DIR = "./output"
#     CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
#     BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
#     DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
#     # Dataset settings
#     TRAIN_SET = "train2017"
#     VAL_SET = "val2017"
#     TRAIN_IMAGES_DIR = os.path.join(DATA_ROOT, "train2017")
#     VAL_IMAGES_DIR = os.path.join(DATA_ROOT, "val2017")
#     ANNOTATIONS_DIR = os.path.join(DATA_ROOT, "annotations")
#     NUM_CLASSES = 2  # Dog and person (background is handled separately)
#     CLASS_NAMES = ['dog', 'person']
#     DOG_CATEGORY_ID = 18
#     DATA_FRACTION = 0.1  # Use 10% of dataset
#     DATA_SET_TO_USE = 0.5  # Use 50% of available dog images
#     TRAIN_VAL_SPLIT = 0.8  # 80% training, 20% validation
#     IMAGE_SIZE = (512, 512)  # Fixed input size
#     MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
#     STD = [0.229, 0.224, 0.225]
    
#     # Training hyperparameters
#     BATCH_SIZE = 4
#     NUM_WORKERS = 4
#     LEARNING_RATE = 5e-5
#     NUM_EPOCHS = 20
#     REG_LOSS_WEIGHT = 2.0
#     PRETRAINED = True
#     WEIGHT_DECAY = 1e-4  # Add weight decay parameter
    
#     # Model architecture settings
#     ANCHOR_SCALES = [32, 64, 128, 256]
#     ANCHOR_RATIOS = [0.5, 1.0, 2.0]
#     BACKBONE_FROZEN_LAYERS = 2
    
#     # Detection settings
#     MAX_DETECTIONS_PER_IMAGE = 20
#     CONF_THRESHOLD = 0.5
#     NMS_THRESHOLD = 0.3
#     IOU_THRESHOLD = 0.5
    
#     # Visualization settings
#     NUM_VAL_IMAGES_TO_LOG = 20
#     DETECTION_COLOR_THRESHOLD = 0.5
#     DETECTION_HIGH_CONF = 0.7
#     DETECTION_MED_CONF = 0.5
#     DETECTION_LINE_THICKNESS_FACTOR = 5
#     DETECTION_FONT_SIZE = 12
#     DETECTION_BG_OPACITY_BASE = 150
#     DETECTION_BG_OPACITY_FACTOR = 100
    
#     # File handling
#     IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# config = Config()

#dog_detector/config.py
import os
import torch

class Config:
    # Dataset paths and settings
    if torch.mps.is_available():
          DATA_ROOT = "./data" 
    else:
        DATA_ROOT = "../coco/NN-V2/data"
    TRAIN_SET = "train2017"
    VAL_SET = "val2017"
    TRAIN_IMAGES_DIR = os.path.join(DATA_ROOT, "train2017")
    VAL_IMAGES_DIR = os.path.join(DATA_ROOT, "val2017")
    ANNOTATIONS_DIR = os.path.join(DATA_ROOT, "annotations")
    
    # Model checkpointing
    CHECKPOINT_DIR = "./checkpoints"
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    
    # Visualization settings
    NUM_VAL_IMAGES_TO_LOG = 20  # Number of validation images to log in tensorboard
    
    # Detection settings (single class: dog)
    NUM_CLASSES = 1  # Only dog (background is handled separately)
    PRETRAINED = True
    DATA_FRACTION = .5  # Use full dataset for better performance
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    IMAGE_SIZE = (512, 512)  # Resize images to fixed size
    # Training hyperparameters
    BATCH_SIZE = 4
    NUM_WORKERS = 4
    LEARNING_RATE = 5e-5  # Reduced for more stable training
    NUM_EPOCHS = 20  # Increased epochs for better convergence
    REG_LOSS_WEIGHT = 2.0  # Increased weight on regression loss for better localization
    # Anchor settings (for the detection head)
    ANCHOR_SCALES = [32, 64, 128, 256]  # Added smaller scale for small dogs
    ANCHOR_RATIOS = [0.5, 1.0, 2.0]  # Standard ratios that work well for dogs
    
    # Detection limits
    MAX_DETECTIONS_PER_IMAGE = 20  # Maximum number of detections to return per image
    
    # Post-processing thresholds
    CONF_THRESHOLD = 0.5  # Lowered from 0.85 to detect more dogs during training
    NMS_THRESHOLD = 0.3  # Decreased from 0.45 to remove more overlapping boxes
    IOU_THRESHOLD = 0.5  # IoU threshold for considering a detection as true positive
    # For inference
    IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
    OUTPUT_DIR = "./output"

config = Config()

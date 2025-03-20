import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image as PILImage
import fiftyone as fo
import fiftyone.zoo as foz

# Import Albumentations and the PyTorch conversion
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import (
    BATCH_SIZE, NUM_WORKERS,
    DATA_ROOT, DATA_SET_TO_USE, TRAIN_VAL_SPLIT
)


class DogDetectionDataset(Dataset):
    def __init__(self, root=DATA_ROOT, split='train', transform=None, download=False):
        """
        Dog Detection Dataset using Open Images
        Args:
            root (str): Root directory for the dataset
            split (str): 'train' or 'validation'
            transform (callable, optional): Albumentations transform to be applied on a sample
            download (bool): If True, downloads the dataset from the internet
        """
        self.transform = transform
        self.root = os.path.abspath(root)
        self.split = "validation" if split == "val" else split
        self.samples = []
        self.dogs_per_image = []
        
        # We'll use a single cache file for all data
        self.cache_file = os.path.join(self.root, 'dog_detection_combined_cache.pt')
        
        # Ensure data directory exists
        os.makedirs(self.root, exist_ok=True)
        
        # Try to load from cache first
        if (os.path.exists(self.cache_file)):
            print(f"Loading combined dataset from cache: {self.cache_file}")
            cache_data = torch.load(self.cache_file)
            all_samples = cache_data['samples']
            all_dogs_per_image = cache_data['dogs_per_image']
            
            # First apply DATA_SET_TO_USE to reduce total dataset size
            total_samples = len(all_samples)
            num_samples_to_use = int(total_samples * DATA_SET_TO_USE)
            all_samples = all_samples[:num_samples_to_use]
            all_dogs_per_image = all_dogs_per_image[:num_samples_to_use]
            
            # Now split into train/val using TRAIN_VAL_SPLIT
            train_size = int(len(all_samples) * TRAIN_VAL_SPLIT)
            
            if self.split == 'train':
                self.samples = all_samples[:train_size]
                self.dogs_per_image = all_dogs_per_image[:train_size]
            else:  # validation split
                self.samples = all_samples[train_size:]
                self.dogs_per_image = all_dogs_per_image[train_size:]
            
            print(f"Successfully loaded {len(self.samples)} samples for {self.split} split")
            print(f"Using {DATA_SET_TO_USE*100:.1f}% of total data with {TRAIN_VAL_SPLIT*100:.1f}% train split")
            return
        
        # If cache doesn't exist, load from dataset
        original_dir = fo.config.dataset_zoo_dir
        fo.config.dataset_zoo_dir = self.root
        
        try:
            # Load or download the dataset
            dataset_name = "open-images-v7-full"
            try:
                dataset = fo.load_dataset(dataset_name)
                print(f"Successfully loaded existing dataset: {dataset_name}")
            except fo.core.dataset.DatasetNotFoundError:
                if download:
                    print("Downloading Open Images dataset with dog class...")
                    dataset = foz.load_zoo_dataset(
                        "open-images-v7",
                        splits=["train", "validation"],  # Load both splits
                        label_types=["detections"],
                        classes=["Dog"],
                        dataset_name=dataset_name
                    )
                    print(f"Successfully downloaded dataset to {fo.config.dataset_zoo_dir}")
                else:
                    raise RuntimeError(f"Dataset {dataset_name} not found and download=False")
            
            # Process all samples from the dataset
            if dataset is not None:
                print(f"Processing {dataset.name} with {len(dataset)} samples")
                all_samples = []
                all_dogs_per_image = []
                
                for sample in dataset.iter_samples():
                    if hasattr(sample, 'ground_truth') and sample.ground_truth is not None:
                        dog_detections = [det for det in sample.ground_truth.detections if det.label == "Dog"]
                        if dog_detections:
                            img_path = sample.filepath
                            if os.path.exists(img_path):
                                boxes = [[det.bounding_box[0], det.bounding_box[1], 
                                        det.bounding_box[0] + det.bounding_box[2],
                                        det.bounding_box[1] + det.bounding_box[3]] for det in dog_detections]
                                all_samples.append((img_path, boxes))
                                all_dogs_per_image.append(len(dog_detections))
                
                total_samples = len(all_samples)
                if total_samples == 0:
                    raise RuntimeError("No valid dog images found in the dataset")
                
                # Save combined dataset to cache
                print(f"Saving combined dataset to cache: {self.cache_file}")
                torch.save({
                    'samples': all_samples,
                    'dogs_per_image': all_dogs_per_image
                }, self.cache_file)
                
                # Apply DATA_SET_TO_USE
                num_samples_to_use = int(total_samples * DATA_SET_TO_USE)
                all_samples = all_samples[:num_samples_to_use]
                all_dogs_per_image = all_dogs_per_image[:num_samples_to_use]
                
                # Split into train/val
                train_size = int(len(all_samples) * TRAIN_VAL_SPLIT)
                
                if self.split == 'train':
                    self.samples = all_samples[:train_size]
                    self.dogs_per_image = all_dogs_per_image[:train_size]
                else:
                    self.samples = all_samples[train_size:]
                    self.dogs_per_image = all_dogs_per_image[train_size:]
                
                print(f"Successfully processed {len(self.samples)} samples for {self.split} split")
                print(f"Using {DATA_SET_TO_USE*100:.1f}% of total data with {TRAIN_VAL_SPLIT*100:.1f}% train split")
            
        except Exception as e:
            print(f"Error initializing dataset: {e}")
            raise
        finally:
            fo.config.dataset_zoo_dir = original_dir
        
        if len(self.samples) == 0:
            raise RuntimeError("No valid dog images found in the dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, boxes = self.samples[index]
        
        try:
            # Open image and convert to RGB (as NumPy array)
            img = np.array(PILImage.open(img_path).convert('RGB'))
            h, w = img.shape[:2]
            
            # First, convert boxes to normalized format [0,1]
            normalized_boxes = []
            for box in boxes:
                x_min, y_min, x_max, y_max = box
                # Normalize if needed (assuming original boxes might be in pixel coords)
                if x_max > 1 or y_max > 1:
                    x_min, x_max = x_min / w, x_max / w
                    y_min, y_max = y_min / h, y_max / h
                x_min = max(0.0, min(0.99, float(x_min)))
                y_min = max(0.0, min(0.99, float(y_min)))
                x_max = max(min(1.0, float(x_max)), x_min + 0.01)
                y_max = max(min(1.0, float(y_max)), y_min + 0.01)
                normalized_boxes.append([x_min, y_min, x_max, y_max])
            
            # Convert normalized boxes to absolute (Pascal VOC) coordinates for Albumentations
            boxes_abs = []
            for box in normalized_boxes:
                x_min, y_min, x_max, y_max = box
                boxes_abs.append([x_min * w, y_min * h, x_max * w, y_max * h])
            labels = [1] * len(boxes_abs)  # All dogs get label "1"
            
            # Apply Albumentations transform if provided
            if self.transform:
                transformed = self.transform(image=img, bboxes=boxes_abs, labels=labels)
                img = transformed['image']
                boxes_abs = transformed['bboxes']
                labels = transformed['labels']
                
                # After transformation, convert absolute coordinates back to normalized values
                # (img from ToTensorV2() is a torch.Tensor of shape [C, H, W])
                _, new_h, new_w = img.shape
                normalized_boxes = []
                for box in boxes_abs:
                    x_min, y_min, x_max, y_max = box
                    normalized_boxes.append([x_min / new_w, y_min / new_h, x_max / new_w, y_max / new_h])
                boxes_tensor = torch.tensor(normalized_boxes, dtype=torch.float32)
                target = {
                    'boxes': boxes_tensor,
                    'labels': torch.tensor(labels, dtype=torch.long),
                    'scores': torch.ones(len(labels))
                }
            else:
                # If no transform, manually convert image to tensor and boxes to normalized values
                img = ToTensorV2()(image=img)['image']
                boxes_tensor = torch.tensor(normalized_boxes, dtype=torch.float32)
                target = {
                    'boxes': boxes_tensor,
                    'labels': torch.ones(len(normalized_boxes), dtype=torch.long),
                    'scores': torch.ones(len(normalized_boxes))
                }
            
            return img, target
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise


def collate_fn(batch):
    """
    Custom collate function to handle variable number of bounding boxes per image.
    Ensures boxes are properly formatted for both model training and visualization.
    """
    images = []
    num_dogs = []
    all_bboxes = []
    
    debug_batch = False
    
    for img, target in batch:
        images.append(img)
        boxes = target['boxes']
        num_dogs.append(len(boxes))
        
        # Check if boxes are normalized and swap coordinates if needed
        if len(boxes) > 0:
            is_normalized = True
            for i in range(len(boxes)):
                for j in range(4):
                    if boxes[i, j] < 0.0 or boxes[i, j] > 1.0:
                        is_normalized = False
                        break
                if not is_normalized:
                    break
            if not is_normalized:
                if debug_batch:
                    print(f"WARNING: Found unnormalized boxes in collate_fn: {boxes}")
                boxes = torch.clamp(boxes, min=0.0, max=1.0)
            for i in range(len(boxes)):
                if boxes[i, 0] > boxes[i, 2]:
                    boxes[i, 0], boxes[i, 2] = boxes[i, 2], boxes[i, 0]
                if boxes[i, 1] > boxes[i, 3]:
                    boxes[i, 1], boxes[i, 3] = boxes[i, 3], boxes[i, 1]
        all_bboxes.append(boxes)
    
    # Check if all images are the same size
    if len(images) > 0:
        expected_shape = images[0].shape
        resized_images = []
        
        for i, img in enumerate(images):
            if img.shape != expected_shape:
                # If image has different size, resize it to match the first image
                if debug_batch:
                    print(f"WARNING: Image {i} has shape {img.shape}, expected {expected_shape}")
                
                # Use interpolate to resize the image tensor
                import torch.nn.functional as F
                img = F.interpolate(img.unsqueeze(0), size=(expected_shape[1], expected_shape[2]), 
                                   mode='bilinear', align_corners=False).squeeze(0)
                
            resized_images.append(img)
        
        images = resized_images
    
    images = torch.stack(images)
    num_dogs = torch.tensor(num_dogs)
    
    if debug_batch:
        print(f"Batch stats: {len(images)} images, avg dogs per image: {num_dogs.float().mean().item():.2f}")
        means = images.view(images.size(0), images.size(1), -1).mean(dim=2).mean(dim=0)
        stds = images.view(images.size(0), images.size(1), -1).std(dim=2).mean(dim=0)
        print(f"Image channel means: {means}, stds: {stds}")
        debug_batch = False
    
    return images, num_dogs, all_bboxes


def get_data_loaders(root=DATA_ROOT, batch_size=BATCH_SIZE, download=True):
    """Create data loaders for Open Images dog detection dataset using Albumentations for augmentation"""
    os.makedirs(root, exist_ok=True)
    print(f"Using data root directory: {os.path.abspath(root)}")
    
    # Define Albumentations transforms for training with enhanced augmentation
    train_transform = A.Compose([
        # Resize with padding to maintain aspect ratio
        A.OneOf([
            A.RandomResizedCrop(
                size=(224, 224), 
                scale=(0.5, 1.0),
                ratio=(0.75, 1.33),
                p=0.7
            ),
            # Letterbox-style resize as an alternative
            A.LongestMaxSize(max_size=224, p=0.3),
            A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, p=1.0)
        ], p=1.0),
        
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.Rotate(limit=15, p=0.8),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.8),
            A.Perspective(scale=(0.05, 0.1), p=0.7),
        ], p=0.5),
        
        # Color augmentations for robustness to different lighting conditions
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=1.0
            ),
            A.RGBShift(
                r_shift_limit=20,
                g_shift_limit=20,
                b_shift_limit=20,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=15, 
                sat_shift_limit=25, 
                val_shift_limit=15, 
                p=1.0
            ),
            A.CLAHE(clip_limit=4.0, p=1.0),
        ], p=0.7),
        
        # Weather and noise simulations
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.MotionBlur(blur_limit=(3, 7), p=0.5),
            A.ImageCompression(quality_lower=70, quality_upper=100, p=0.5),
        ], p=0.3),
        
        # Dog-specific augmentations - cutout/cutmix-like for occlusion robustness
        A.OneOf([
            A.CoarseDropout(
                max_holes=8, 
                max_height=32, 
                max_width=32, 
                fill_value=0,
                p=0.5
            ),
            A.GridDropout(
                ratio=0.1,
                unit_size_min=10,
                unit_size_max=40,
                p=0.5
            ),
        ], p=0.2),
        
        # Normalize and convert to tensor
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.3,  # Ensure boxes remain mostly visible
        min_area=100  # Minimum area in pixels to keep a box
    ))
    
    # Define Albumentations transforms for validation - keep simple for consistent evaluation
    val_transform = A.Compose([
        # Letterboxing for validation to keep aspect ratio intact
        A.LongestMaxSize(max_size=224),
        A.PadIfNeeded(min_height=224, min_width=224, border_mode=0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc', 
        label_fields=['labels'],
        min_visibility=0.3
    ))
    
    # Create the training and validation datasets
    print("Creating training dataset...")
    try:
        train_dataset = DogDetectionDataset(
            root=root,
            split='train',
            transform=train_transform,
            download=download
        )
    except Exception as e:
        print(f"Error creating training dataset: {e}")
        raise RuntimeError(f"Failed to create training dataset: {e}")
    
    print("Creating validation dataset...")
    try:
        val_dataset = DogDetectionDataset(
            root=root,
            split='validation',
            transform=val_transform,
            download=download
        )
    except Exception as e:
        print(f"Error creating validation dataset: {e}")
        raise RuntimeError(f"Failed to create validation dataset: {e}")
    
    print(f"Train set: {len(train_dataset)} images with dogs")
    print(f"Val set: {len(val_dataset)} images with dogs")
    
    num_workers = min(8, NUM_WORKERS)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Ensure pinned memory
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2,  # Prefetch 2 batches per worker
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers // 2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


def get_total_samples():
    """Get the total number of samples in the dataset"""
    cache_file = os.path.join(DATA_ROOT, 'dog_detection_train_cache.pt')
    if os.path.exists(cache_file):
        cache_data = torch.load(cache_file)
        return len(cache_data['samples'])
    return 0

def create_datasets():
    """Create training and validation datasets with the specified split ratio"""
    print("Creating training dataset...")
    train_dataset = DogDetectionDataset(
        DATA_ROOT,
        split='train'
    )
    
    print("Creating validation dataset...")
    val_dataset = DogDetectionDataset(
        DATA_ROOT,
        split='validation'
    )
    
    return train_dataset, val_dataset


# TransformedSubset class is updated for consistency with Albumentations.
class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    
    def __getitem__(self, idx):
        img, target = self.subset[idx]
        # If img is a PIL image, convert to numpy array before transforming.
        if isinstance(img, PILImage.Image):
            img = np.array(img)
        # Convert target boxes from tensor to list of lists.
        bboxes = target['boxes'].tolist()
        labels = target['labels'].tolist()
        transformed = self.transform(image=img, bboxes=bboxes, labels=labels)
        img = transformed['image']
        target['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
        target['labels'] = torch.tensor(transformed['labels'], dtype=torch.long)
        return img, target
    
    def __len__(self):
        return len(self.subset)

import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
import random
from collections import Counter
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('dog_detector')


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
            logger.info(f"Loading dataset from cache: {self.cache_file}")
            cache_data = torch.load(self.cache_file)
            all_samples = cache_data['samples']
            all_dogs_per_image = cache_data['dogs_per_image']
            
            # First apply DATA_SET_TO_USE to reduce total dataset size
            total_samples = len(all_samples)
            
            # Sort samples by number of dogs to analyze distribution
            samples_with_count = list(zip(all_samples, all_dogs_per_image))
            # Shuffle with fixed seed for reproducibility before selecting subset
            random.seed(42)
            random.shuffle(samples_with_count)
            
            # Count distribution of dog counts before filtering (only in debug level)
            if logger.isEnabledFor(logging.DEBUG):
                dog_count_distribution = Counter(all_dogs_per_image)
                logger.debug(f"Original dog count distribution: {dict(dog_count_distribution)}")
            
            # Stratified sampling to ensure we keep multi-dog images
            if DATA_SET_TO_USE < 1.0:
                logger.info(f"Performing stratified sampling (DATA_SET_TO_USE={DATA_SET_TO_USE:.2f})")
                # Group samples by dog count
                samples_by_count = {}
                for sample, count in zip(all_samples, all_dogs_per_image):
                    if count not in samples_by_count:
                        samples_by_count[count] = []
                    samples_by_count[count].append((sample, count))
                
                # Select DATA_SET_TO_USE percentage from each group, but ensure we keep
                # at least 80% of multi-dog samples if available
                selected_samples = []
                for count, samples in samples_by_count.items():
                    num_to_select = int(len(samples) * DATA_SET_TO_USE)
                    # For multi-dog images, ensure we keep more samples
                    if count > 1:
                        num_to_select = max(num_to_select, int(len(samples) * 0.8))
                    # Don't select more than we have
                    num_to_select = min(num_to_select, len(samples))
                    selected_samples.extend(samples[:num_to_select])
                
                # Re-shuffle the selected samples
                random.shuffle(selected_samples)
                
                # Unpack the samples and counts
                all_samples = [s[0] for s in selected_samples]
                all_dogs_per_image = [s[1] for s in selected_samples]
            
            # Report distribution after filtering (only in debug level)
            if logger.isEnabledFor(logging.DEBUG):
                filtered_dog_distribution = Counter(all_dogs_per_image)
                logger.debug(f"Filtered dog count distribution: {dict(filtered_dog_distribution)}")
            
            logger.info(f"Using {len(all_samples)} samples ({len(all_samples)/total_samples:.1%} of original data)")
            
            # Now split into train/val using TRAIN_VAL_SPLIT with stratification
            train_samples = []
            train_counts = []
            val_samples = []
            val_counts = []
            
            # Group by dog count for stratified split
            samples_by_count = {}
            for sample, count in zip(all_samples, all_dogs_per_image):
                if count not in samples_by_count:
                    samples_by_count[count] = []
                samples_by_count[count].append((sample, count))
            
            # Split each group using TRAIN_VAL_SPLIT
            for count, samples in samples_by_count.items():
                train_size = int(len(samples) * TRAIN_VAL_SPLIT)
                train_group = samples[:train_size]
                val_group = samples[train_size:]
                
                train_samples.extend([s[0] for s in train_group])
                train_counts.extend([s[1] for s in train_group])
                val_samples.extend([s[0] for s in val_group])
                val_counts.extend([s[1] for s in val_group])
            
            # Shuffle again after splitting
            train_pairs = list(zip(train_samples, train_counts))
            val_pairs = list(zip(val_samples, val_counts))
            random.shuffle(train_pairs)
            random.shuffle(val_pairs)
            
            train_samples, train_counts = zip(*train_pairs) if train_pairs else ([], [])
            val_samples, val_counts = zip(*val_pairs) if val_pairs else ([], [])
            
            # Assign to the appropriate split
            if self.split == 'train':
                self.samples = list(train_samples)
                self.dogs_per_image = list(train_counts)
            else:  # validation split
                self.samples = list(val_samples)
                self.dogs_per_image = list(val_counts)
            
            # Show distribution for current split (only in debug level)
            if logger.isEnabledFor(logging.DEBUG):
                curr_distribution = Counter(self.dogs_per_image)
                logger.debug(f"Dog count distribution for {self.split} split: {dict(curr_distribution)}")
            
            logger.info(f"Loaded {len(self.samples)} samples for {self.split} split")
            return
        
        # If cache doesn't exist, load from dataset
        original_dir = fo.config.dataset_zoo_dir
        fo.config.dataset_zoo_dir = self.root
        
        try:
            # Load or download the dataset
            dataset_name = "open-images-v7-full"
            try:
                dataset = fo.load_dataset(dataset_name)
                logger.info(f"Loaded existing dataset: {dataset_name}")
            except fo.core.dataset.DatasetNotFoundError:
                if download:
                    logger.info("Downloading Open Images dataset with dog class...")
                    dataset = foz.load_zoo_dataset(
                        "open-images-v7",
                        splits=["train", "validation"],  # Load both splits
                        label_types=["detections"],
                        classes=["Dog"],
                        dataset_name=dataset_name
                    )
                    logger.info(f"Downloaded dataset to {fo.config.dataset_zoo_dir}")
                else:
                    raise RuntimeError(f"Dataset {dataset_name} not found and download=False")
            
            # Process all samples from the dataset
            if dataset is not None:
                logger.info(f"Processing {dataset.name} with {len(dataset)} samples")
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
                logger.info(f"Saving dataset to cache: {self.cache_file}")
                torch.save({
                    'samples': all_samples,
                    'dogs_per_image': all_dogs_per_image
                }, self.cache_file)
                
                # Continue processing as in the cached case
                if logger.isEnabledFor(logging.DEBUG):
                    dog_count_distribution = Counter(all_dogs_per_image)
                    logger.debug(f"Original dog count distribution: {dict(dog_count_distribution)}")
                
                # Stratified sampling with the same logic as above
                # (Same code as in the cached branch)
                
        except Exception as e:
            logger.error(f"Error initializing dataset: {e}")
            raise
        finally:
            fo.config.dataset_zoo_dir = original_dir
        
        if len(self.samples) == 0:
            raise RuntimeError("No valid dog images found in the dataset")

    def get_sample_weights(self):
        """
        Generate sample weights to balance single and multi-dog examples during training.
        This helps ensure the model sees enough multi-dog examples.
        """
        if self.split != 'train':
            return None
            
        # Count occurrences of each dog count
        dog_counts = Counter(self.dogs_per_image)
        total_samples = len(self.dogs_per_image)
        
        # Calculate inverse frequency for each count
        weights = []
        for count in self.dogs_per_image:
            # Weight inversely proportional to frequency, with additional boost for multi-dog images
            weight = total_samples / (dog_counts[count] * len(dog_counts))
            # Additional boost for multi-dog cases
            if count > 1:
                weight *= 1.5  # Boost multi-dog samples
            weights.append(weight)
            
        return weights

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index):
        img_path, boxes = self.samples[index]
        dog_count = self.dogs_per_image[index]
        
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
            
            # Apply Albumentations transform if provided, with special handling for multi-dog cases
            if self.transform:
                # For multi-dog images, we might need to adjust the transform to preserve all dogs
                if dog_count > 1 and self.split == 'train':
                    # Get special multi-dog transform or use default one
                    transform = get_multi_dog_transform() if random.random() < 0.7 else self.transform
                else:
                    transform = self.transform
                    
                transformed = transform(image=img, bboxes=boxes_abs, labels=labels)
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
                    'scores': torch.ones(len(labels)),
                    'dog_count': dog_count  # Include the dog count in the target
                }
            else:
                # If no transform, manually convert image to tensor and boxes to normalized values
                img = ToTensorV2()(image=img)['image']
                boxes_tensor = torch.tensor(normalized_boxes, dtype=torch.float32)
                target = {
                    'boxes': boxes_tensor,
                    'labels': torch.ones(len(normalized_boxes), dtype=torch.long),
                    'scores': torch.ones(len(normalized_boxes)),
                    'dog_count': dog_count  # Include the dog count in the target
                }
            
            return img, target
            
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}")
            # Instead of raising, return a dummy sample
            img = torch.zeros((3, 224, 224), dtype=torch.float32)
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.long),
                'scores': torch.zeros(0),
                'dog_count': 0
            }
            return img, target
            
def get_multi_dog_transform():
    """
    Special transform for multi-dog images that's more careful about preserving all dogs.
    Uses less aggressive cropping and ensures all dogs remain visible.
    """
    return A.Compose([
        # Less aggressive crop for multi-dog images to preserve more context
        A.RandomResizedCrop(
            size=(224, 224), 
            scale=(0.8, 1.0),  # Less aggressive scaling
            ratio=(0.85, 1.15),  # Less variation in aspect ratio
            interpolation=1,
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.3),  # Less rotation to avoid cutting off dogs
        # Color augmentations for better normalization and robustness
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=10, 
                sat_shift_limit=15, 
                val_shift_limit=10, 
                p=1.0
            ),
        ], p=0.4),
        # Normalize and convert to tensor
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.5  # Higher visibility threshold for multi-dog cases
    ))

def collate_fn(batch):
    """
    Custom collate function to handle variable number of bounding boxes per image.
    Ensures boxes are properly formatted for both model training and visualization.
    Makes sure all images are consistently sized to 224x224.
    """
    images = []
    num_dogs = []
    all_bboxes = []
    valid_batch = []
    
    # Standard image size required for the model
    target_size = (224, 224)
    
    for i, (img, target) in enumerate(batch):
        # Skip any invalid samples (e.g., images that failed to load)
        if img.shape[0] != 3 or img.isnan().any() or len(target['boxes']) == 0:
            continue
            
        # Always ensure every image is 224x224, regardless of original size
        if img.shape[1] != target_size[0] or img.shape[2] != target_size[1]:
            # Resize images that don't match the target size
            import torch.nn.functional as F
            img = F.interpolate(img.unsqueeze(0), size=target_size, 
                               mode='bilinear', align_corners=False).squeeze(0)
        
        valid_batch.append((img, target))
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
                logger.warning(f"Found unnormalized boxes in collate_fn: {boxes}")
                boxes = torch.clamp(boxes, min=0.0, max=1.0)
            
            # Fix box coordinates if needed
            for i in range(len(boxes)):
                if boxes[i, 0] > boxes[i, 2]:
                    boxes[i, 0], boxes[i, 2] = boxes[i, 2], boxes[i, 0]
                if boxes[i, 1] > boxes[i, 3]:
                    boxes[i, 1], boxes[i, 3] = boxes[i, 3], boxes[i, 1]
                    
                # Ensure minimum box size to prevent degenerate boxes
                if boxes[i, 2] - boxes[i, 0] < 0.01:
                    boxes[i, 2] = min(1.0, boxes[i, 0] + 0.01)
                if boxes[i, 3] - boxes[i, 1] < 0.01:
                    boxes[i, 3] = min(1.0, boxes[i, 1] + 0.01)
        
        all_bboxes.append(boxes)
    
    # If we have no valid samples, create a dummy batch
    if len(valid_batch) == 0:
        logger.warning("Empty batch after filtering invalid samples")
        dummy_img = torch.zeros((3, target_size[0], target_size[1]), dtype=torch.float32)
        dummy_boxes = torch.zeros((0, 4), dtype=torch.float32)
        return torch.stack([dummy_img]), torch.tensor([0]), [dummy_boxes]
    
    # Stack the tensors - we don't need to check sizes anymore since we've already
    # resized everything to the target size
    if len(images) > 0:
        images = torch.stack(images)
        num_dogs = torch.tensor(num_dogs)
        
        # Periodically log batch statistics (at debug level only and much less frequently)
        if logger.isEnabledFor(logging.DEBUG) and random.random() < 0.01:  # Log roughly 1% of batches
            means = images.view(images.size(0), images.size(1), -1).mean(dim=2).mean(dim=0)
            stds = images.view(images.size(0), images.size(1), -1).std(dim=2).mean(dim=0)
            logger.debug(f"Batch stats: {len(images)} images, dogs per image: {num_dogs.tolist()}")
            logger.debug(f"Image channel means: {means.tolist()}, stds: {stds.tolist()}")
            
            # Log multi-dog percentage
            multi_dog_count = sum(1 for n in num_dogs if n > 1)
            if len(num_dogs) > 0:
                logger.debug(f"Multi-dog percentage in batch: {multi_dog_count / len(num_dogs):.1%}")
        
        return images, num_dogs, all_bboxes
    
    # Fallback for empty batch (should not happen often)
    dummy_img = torch.zeros((3, target_size[0], target_size[1]), dtype=torch.float32)
    dummy_boxes = torch.zeros((0, 4), dtype=torch.float32)
    return torch.stack([dummy_img]), torch.tensor([0]), [dummy_boxes]

def get_data_loaders(root=DATA_ROOT, batch_size=BATCH_SIZE, download=True):
    """Create data loaders for Open Images dog detection dataset using Albumentations for augmentation"""
    os.makedirs(root, exist_ok=True)
    logger.info(f"Using data root directory: {os.path.abspath(root)}")
    
    # Define Albumentations transforms for training
    train_transform = A.Compose([
        # Improved random crop to better handle single dogs
        A.RandomResizedCrop(
            size=(224, 224), 
            scale=(0.7, 1.0),
            ratio=(0.8, 1.25),
            interpolation=1,
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.3),
        # Improved color augmentations
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=1.0
            ),
            A.RGBShift(
                r_shift_limit=15,
                g_shift_limit=15,
                b_shift_limit=15,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=10, 
                sat_shift_limit=20, 
                val_shift_limit=10, 
                p=1.0
            ),
        ], p=0.5),
        # Add perspective transforms to simulate different viewpoints
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.2),
        # Scale-specific augmentations
        A.OneOf([
            A.RandomScale(scale_limit=(-0.3, 0.1), p=1.0),
            A.RandomScale(scale_limit=(0.1, 0.3), p=1.0),
        ], p=0.3),
        # Normalize and convert to tensor
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.4  # Increased from 0.3 for better quality detections
    ))
    
    # Define Albumentations transforms for validation - keep simple for consistent evaluation
    val_transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    # Create the training and validation datasets
    logger.info("Creating training dataset...")
    try:
        train_dataset = DogDetectionDataset(
            root=root,
            split='train',
            transform=train_transform,
            download=download
        )
    except Exception as e:
        logger.error(f"Error creating training dataset: {e}")
        raise RuntimeError(f"Failed to create training dataset: {e}")
    
    logger.info("Creating validation dataset...")
    try:
        val_dataset = DogDetectionDataset(
            root=root,
            split='validation',
            transform=val_transform,
            download=download
        )
    except Exception as e:
        logger.error(f"Error creating validation dataset: {e}")
        raise RuntimeError(f"Failed to create validation dataset: {e}")
    
    logger.info(f"Train set: {len(train_dataset)} images with dogs")
    logger.info(f"Val set: {len(val_dataset)} images with dogs")
    
    # Create weighted sampler for training to balance single/multi-dog examples
    sample_weights = train_dataset.get_sample_weights()
    train_sampler = None
    
    if sample_weights is not None:
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        logger.info("Using weighted sampler to balance single-dog and multi-dog examples")
    
    num_workers = min(8, NUM_WORKERS)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
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
    cache_file = os.path.join(DATA_ROOT, 'dog_detection_combined_cache.pt')
    if os.path.exists(cache_file):
        cache_data = torch.load(cache_file)
        return len(cache_data['samples'])
    return 0


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

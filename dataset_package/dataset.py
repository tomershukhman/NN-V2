# dataset.py
import os
import cv2
import numpy as np
import torch
import fiftyone as fo
import fiftyone.zoo as foz
import logging
import pickle
import hashlib
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('dog_detector')

# Mapping from class name to numerical label
LABEL_MAP = {"Person": 0, "Dog": 1}

class ObjectDetectionDataset(torch.utils.data.Dataset):
    """
    Custom Dataset for object detection with caching support.
    """
    def __init__(self, samples_data, transforms, label_map, cache_size=1000):
        """
        Args:
            samples_data: List of dictionaries containing sample data
            transforms: Albumentations transform
            label_map: Dictionary mapping class names to numeric labels
            cache_size: Maximum number of images to keep in memory
        """
        self.samples_data = samples_data
        self.transforms = transforms
        self.label_map = label_map
        self.cache_size = cache_size
        self.image_cache = {}
        self.cache_keys = []
    
    def __len__(self):
        return len(self.samples_data)
    
    def _load_and_cache_image(self, image_path):
        """Load image and add to cache if not present"""
        if image_path in self.image_cache:
            return self.image_cache[image_path]
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Add to cache if we have space
        if len(self.cache_keys) >= self.cache_size:
            # Remove oldest cached image
            oldest_key = self.cache_keys.pop(0)
            del self.image_cache[oldest_key]
        
        self.image_cache[image_path] = image
        self.cache_keys.append(image_path)
        
        return image

    def __getitem__(self, idx):
        sample_data = self.samples_data[idx]
        image_path = sample_data["filepath"]
        
        # Load image with caching
        try:
            image = self._load_and_cache_image(image_path)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a small black image as fallback
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        h, w, _ = image.shape
        
        # Use pre-extracted annotations
        normalized_bboxes = sample_data["bboxes"]
        category_ids = sample_data["category_ids"]
        
        # Convert normalized bboxes to absolute coordinates for Pascal VOC format
        bboxes = []
        for x, y, bw, bh in normalized_bboxes:
            x_min = x * w
            y_min = y * h
            x_max = (x + bw) * w
            y_max = (y + bh) * h
            bboxes.append([x_min, y_min, x_max, y_max])
        
        # Process with transforms
        if len(bboxes) == 0:
            # Handle case with no valid bounding boxes
            transformed = self.transforms(image=image)
            image = transformed["image"]
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64)
            }
        else:
            # Use "labels" instead of "category_ids" to match the label_fields in transforms
            transformed = self.transforms(image=image, bboxes=bboxes, labels=category_ids)
            image = transformed["image"]
            bboxes = transformed["bboxes"]
            labels = transformed["labels"]
            
            target = {
                "boxes": torch.as_tensor(bboxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64)
            }
        
        return image, target
    
    def clear_cache(self):
        """Clear the image cache to free memory"""
        self.image_cache.clear()
        self.cache_keys.clear()

def get_cache_path(max_samples, dataset_name):
    """Generate a unique cache path based on dataset parameters"""
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    
    # Create a unique identifier based on parameters
    params = f"{dataset_name}-{max_samples}"
    cache_id = hashlib.md5(params.encode()).hexdigest()[:10]
    
    return cache_dir / f"dataset_cache_{cache_id}.pkl"

def download_and_prepare_dataset(max_samples=500):
    """
    Downloads a subset of the Open Images dataset (v6) using FiftyOne,
    with caching support for faster subsequent loads.
    """
    dataset_name = "open-images-person-dog"
    cache_path = get_cache_path(max_samples, dataset_name)
    
    # If cache doesn't exist or is invalid, download dataset
    try:
        dataset = fo.load_dataset(dataset_name)
        logger.info(f"Loaded existing dataset '{dataset_name}' with {len(dataset)} samples")
    except ValueError:
        logger.info(f"Downloading Open Images dataset with max_samples={max_samples}")
        dataset = foz.load_zoo_dataset(
            "open-images-v6",
            split="train",
            label_types="detections",
            classes=["Person", "Dog"],
            max_samples=max_samples,
            shuffle=True,
            dataset_name=dataset_name
        )
        logger.info(f"Downloaded dataset with {len(dataset)} samples")
    
    # Extract essential data into a simpler, serializable format
    try:
        serializable_data = {
            'samples': [],
            'max_samples': max_samples,
            'dataset_name': dataset_name,
            'timestamp': str(np.datetime64('now'))
        }
        
        for sample in dataset:
            if hasattr(sample, 'ground_truth') and hasattr(sample.ground_truth, 'detections'):
                sample_data = {
                    'filepath': sample.filepath,
                    'detections': []
                }
                
                for det in sample.ground_truth.detections:
                    if hasattr(det, 'label') and hasattr(det, 'bounding_box'):
                        detection = {
                            'label': det.label,
                            'bounding_box': det.bounding_box.tolist() if hasattr(det.bounding_box, 'tolist') else det.bounding_box
                        }
                        sample_data['detections'].append(detection)
                
                serializable_data['samples'].append(sample_data)
        
        # Save to cache
        logger.info(f"Saving serializable dataset to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(serializable_data, f)
            
        return dataset
        
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")
        # Return the dataset even if caching fails
        return dataset

def create_balanced_samples(foset):
    """
    Create a balanced list of samples with caching support.
    Handles both FiftyOne dataset objects and serialized dataset format.
    """
    cache_path = Path("cache/balanced_samples.pkl")
    cache_path.parent.mkdir(exist_ok=True)
    
    # Try to load from cache
    if cache_path.exists():
        try:
            logger.info("Loading balanced samples from cache")
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                
            # Basic validation of cache data
            if isinstance(cache_data, dict) and 'samples' in cache_data and 'weights' in cache_data:
                logger.info("Using cached balanced samples")
                return cache_data['samples'], cache_data['weights']
        except Exception as e:
            logger.warning(f"Failed to load balanced samples cache: {e}")
    
    logger.info(f"Creating balanced samples from {len(foset)} samples")
    
    if len(foset) == 0:
        logger.warning("Empty dataset provided to create_balanced_samples")
        return [], []
    
    # Initialize counters and collectors
    class_counts = {"Person": 0, "Dog": 0}
    both_classes = 0
    multi_person = 0
    multi_dog = 0
    person_bbox_sizes = []
    dog_bbox_sizes = []
    
    valid_samples_data = []
    valid_sample_flags = []
    n_person = 0
    n_dog = 0
    
    # Handle both FiftyOne dataset and serialized format
    for sample in foset:
        has_person = False
        has_dog = False
        sample_bboxes = []
        sample_category_ids = []
        
        # Handle serialized format
        if isinstance(sample, dict) and 'detections' in sample:
            detections = sample['detections']
            filepath = sample['filepath']
            
            for det in detections:
                if 'label' in det and 'bounding_box' in det:
                    label = det['label']
                    bbox = det['bounding_box']
                    
                    if label in LABEL_MAP:
                        sample_bboxes.append(bbox)
                        sample_category_ids.append(LABEL_MAP[label])
                        
                        if label == "Person":
                            has_person = True
                            area = bbox[2] * bbox[3]  # w * h
                            person_bbox_sizes.append(area)
                        elif label == "Dog":
                            has_dog = True
                            area = bbox[2] * bbox[3]  # w * h
                            dog_bbox_sizes.append(area)
            
        # Handle FiftyOne dataset format
        elif hasattr(sample, 'ground_truth') and hasattr(sample.ground_truth, 'detections'):
            filepath = sample.filepath
            for det in sample.ground_truth.detections:
                if hasattr(det, 'label') and hasattr(det, 'bounding_box'):
                    label = det.label
                    if label in LABEL_MAP:
                        try:
                            bbox = det.bounding_box
                            if hasattr(bbox, 'tolist'):
                                bbox = bbox.tolist()
                            
                            sample_bboxes.append(bbox)
                            sample_category_ids.append(LABEL_MAP[label])
                            
                            if label == "Person":
                                has_person = True
                                area = bbox[2] * bbox[3]  # w * h
                                person_bbox_sizes.append(area)
                            elif label == "Dog":
                                has_dog = True
                                area = bbox[2] * bbox[3]  # w * h
                                dog_bbox_sizes.append(area)
                        except Exception as e:
                            logger.warning(f"Error processing bbox: {e}")
                            continue
        
        # Update statistics
        if has_person:
            class_counts["Person"] += 1
        if has_dog:
            class_counts["Dog"] += 1
        if has_person and has_dog:
            both_classes += 1
        
        # Only include samples with at least one valid detection
        if len(sample_bboxes) > 0:
            sample_data = {
                "filepath": filepath,
                "bboxes": sample_bboxes,
                "category_ids": sample_category_ids
            }
            valid_samples_data.append(sample_data)
            valid_sample_flags.append((has_person, has_dog))
            if has_person:
                n_person += 1
            if has_dog:
                n_dog += 1
    
    # Display statistics
    logger.info("=== Dataset Statistics ===")
    total_samples = len(foset)
    logger.info(f"Images with Person: {class_counts['Person']} ({class_counts['Person']/total_samples*100:.1f}%)")
    logger.info(f"Images with Dog: {class_counts['Dog']} ({class_counts['Dog']/total_samples*100:.1f}%)")
    logger.info(f"Images with both: {both_classes} ({both_classes/total_samples*100:.1f}%)")
    logger.info(f"Found {len(valid_samples_data)} valid samples")
    logger.info("========================")
    
    # Handle no valid samples case
    if len(valid_samples_data) == 0:
        logger.error("No valid samples found!")
        return [], []
    
    # Calculate sample weights
    total_samples = len(valid_samples_data)
    class_weights = {
        "Person": total_samples / max(n_person, 1),
        "Dog": total_samples / max(n_dog, 1)
    }
    
    sample_weights = []
    for has_person, has_dog in valid_sample_flags:
        weight = 0.0
        if has_person and has_dog:
            weight = max(class_weights["Dog"] * 1.2, class_weights["Person"])
        elif has_dog:
            weight = class_weights["Dog"]
        elif has_person:
            weight = class_weights["Person"]
        sample_weights.append(weight)
    
    # Normalize weights
    total_weight = sum(sample_weights)
    if total_weight > 0:
        sample_weights = [w / total_weight * len(sample_weights) for w in sample_weights]
    else:
        logger.warning("All sample weights are zero. Using uniform weights.")
        sample_weights = [1.0] * len(valid_samples_data)
    
    # Save to cache
    try:
        cache_data = {
            'samples': valid_samples_data,
            'weights': sample_weights,
            'timestamp': str(np.datetime64('now'))
        }
        logger.info("Saving balanced samples to cache")
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
    except Exception as e:
        logger.warning(f"Failed to save balanced samples cache: {e}")
    
    return valid_samples_data, sample_weights

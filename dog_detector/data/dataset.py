"""
Dog detection dataset module.

This module defines the core dataset class for loading and processing dog detection data
from the Open Images dataset.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image as PILImage
import fiftyone as fo
import fiftyone.zoo as foz
from albumentations.pytorch import ToTensorV2

from config import DATA_ROOT
from .cache import load_from_cache, save_to_cache


class DogDetectionDataset(Dataset):
    """
    Dataset for dog detection using Open Images.
    
    Loads images and bounding box annotations for dog detection from Open Images dataset.
    Uses caching to speed up subsequent data loading.
    """
    
    def __init__(self, root=DATA_ROOT, split='train', transform=None, download=False):
        """
        Initialize the dog detection dataset.
        
        Args:
            root (str): Root directory for the dataset
            split (str): 'train' or 'validation'
            transform (callable, optional): Transform to be applied on samples
            download (bool): If True, downloads the dataset if not available
        """
        self.transform = transform
        self.root = os.path.abspath(root)
        self.split = "validation" if split == "val" else split
        self.samples = []
        self.dogs_per_image = []
        
        # Ensure data directory exists
        os.makedirs(self.root, exist_ok=True)
        
        # Try to load from cache first
        self.samples, self.dogs_per_image = load_from_cache(self.root, self.split)
        if self.samples is not None:
            return
        
        # If cache doesn't exist, load from dataset
        self._load_from_dataset(download)
        
        if len(self.samples) == 0:
            raise RuntimeError("No valid dog images found in the dataset")
    
    def _load_from_dataset(self, download):
        """
        Load dataset from original source (Open Images).
        
        Args:
            download: Whether to download dataset if not found
        """
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
                save_to_cache(self.root, all_samples, all_dogs_per_image)

                # Load the processed data for this split
                self.samples, self.dogs_per_image = load_from_cache(self.root, self.split)

        except Exception as e:
            print(f"Error initializing dataset: {e}")
            raise
        finally:
            fo.config.dataset_zoo_dir = original_dir
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        Get a dataset item by index.
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            tuple: (image, target) where target is a dict with 'boxes', 'labels', and 'scores'
        """
        img_path, boxes = self.samples[index]
        
        try:
            # Open image and convert to RGB (as NumPy array)
            img = np.array(PILImage.open(img_path).convert('RGB'))
            h, w = img.shape[:2]
            
            # Convert boxes to normalized format [0,1]
            normalized_boxes = self._normalize_boxes(boxes, w, h)
            
            # Apply transforms or convert to tensor
            if self.transform:
                # Convert normalized boxes to absolute for Albumentations
                boxes_abs = self._normalized_to_absolute(normalized_boxes, w, h)
                labels = [1] * len(boxes_abs)  # All dogs get label "1"
                
                # Apply the transform
                transformed = self.transform(image=img, bboxes=boxes_abs, labels=labels)
                img = transformed['image']
                boxes_abs = transformed['bboxes']
                labels = transformed['labels']
                
                # After transformation, convert absolute boxes back to normalized
                _, new_h, new_w = img.shape
                normalized_boxes = self._absolute_to_normalized(boxes_abs, new_w, new_h)
                boxes_tensor = torch.tensor(normalized_boxes, dtype=torch.float32)
                target = {
                    'boxes': boxes_tensor,
                    'labels': torch.tensor(labels, dtype=torch.long),
                    'scores': torch.ones(len(labels))
                }
            else:
                # If no transform, manually convert image to tensor
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
    
    def _normalize_boxes(self, boxes, width, height):
        """
        Convert boxes to normalized format [0,1].
        
        Args:
            boxes: List of [x1, y1, x2, y2] coordinates
            width: Image width
            height: Image height
            
        Returns:
            list: Normalized boxes as [[x1, y1, x2, y2], ...]
        """
        normalized_boxes = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            # Normalize if needed (assuming original boxes might be in pixel coords)
            if x_max > 1 or y_max > 1:
                x_min, x_max = x_min / width, x_max / width
                y_min, y_max = y_min / height, y_max / height
            
            # Clamp values to valid range
            x_min = max(0.0, min(0.99, float(x_min)))
            y_min = max(0.0, min(0.99, float(y_min)))
            x_max = max(min(1.0, float(x_max)), x_min + 0.01)
            y_max = max(min(1.0, float(y_max)), y_min + 0.01)
            
            normalized_boxes.append([x_min, y_min, x_max, y_max])
        return normalized_boxes
    
    def _normalized_to_absolute(self, boxes, width, height):
        """
        Convert normalized boxes to absolute coordinates.
        
        Args:
            boxes: List of normalized [x1, y1, x2, y2] coordinates
            width: Image width
            height: Image height
            
        Returns:
            list: Absolute pixel boxes as [[x1, y1, x2, y2], ...]
        """
        return [[box[0] * width, box[1] * height, box[2] * width, box[3] * height] for box in boxes]
    
    def _absolute_to_normalized(self, boxes, width, height):
        """
        Convert absolute boxes to normalized coordinates.
        
        Args:
            boxes: List of absolute [x1, y1, x2, y2] coordinates
            width: Image width
            height: Image height
            
        Returns:
            list: Normalized boxes as [[x1, y1, x2, y2], ...]
        """
        return [[box[0] / width, box[1] / height, box[2] / width, box[3] / height] for box in boxes]
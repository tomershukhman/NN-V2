import os
import argparse
import random
import fiftyone as fo
import fiftyone.zoo as foz
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pickle
import numpy as np
import torch


class DogDataset(Dataset):
    """
    PyTorch Dataset for dog detection using cached metadata.
    """
    
    def __init__(self, samples, transform=None, max_boxes=10):
        """
        Initialize the dataset.
        
        Args:
            samples (list): List of sample metadata dictionaries
            transform (callable, optional): Transform to apply to images
            max_boxes (int): Maximum number of bounding boxes to return (padding will be added if fewer)
        """
        self.samples = samples
        self.transform = transform
        self.max_boxes = max_boxes
        
        # Verify dataset integrity
        self._verify_dataset()
        
        # Shuffle samples
        random.shuffle(self.samples)
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def _verify_dataset(self):
        """Verify dataset integrity"""
        if not self.samples:
            raise ValueError("Dataset is empty!")
            
        # Check for duplicate filepaths
        filepaths = [sample['filepath'] for sample in self.samples]
        unique_filepaths = set(filepaths)
        if len(filepaths) != len(unique_filepaths):
            raise ValueError(f"Found {len(filepaths) - len(unique_filepaths)} duplicate filepaths in dataset!")
            
        # Verify all files exist
        missing_files = []
        for sample in self.samples:
            if not os.path.exists(sample['filepath']):
                missing_files.append(sample['filepath'])
        
        if missing_files:
            raise ValueError(f"Found {len(missing_files)} missing files in dataset! First few: {missing_files[:5]}")
            
        # Verify labels are valid
        invalid_labels = [sample for sample in self.samples if sample['label'] not in [0, 1]]
        if invalid_labels:
            raise ValueError(f"Found {len(invalid_labels)} samples with invalid labels! Should be 0 or 1.")
            
        # Try loading first few images to verify they're valid
        for sample in self.samples[:5]:
            try:
                with Image.open(sample['filepath']) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
            except Exception as e:
                raise ValueError(f"Failed to load image {sample['filepath']}: {str(e)}")
    
    def __len__(self):
        """Return the total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index
            
        Returns:
            tuple: (image, num_dogs, boxes) where:
                - image is the preprocessed image tensor
                - num_dogs is the number of dogs in the image
                - boxes is a tensor of bounding boxes in [x1, y1, x2, y2] format, padded to max_boxes
        """
        sample = self.samples[idx]
        filepath = sample['filepath']
        
        # Open and transform image
        try:
            image = Image.open(filepath).convert('RGB')
            w, h = image.size
        except Exception as e:
            print(f"Error loading image {filepath}: {e}")
            # Return a black image in case of error
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            w, h = image.size

        # Get bounding boxes if they exist
        if 'detections' in sample and sample['label'] == 1:
            # Convert detections to normalized coordinates
            boxes = []
            for det in sample['detections']:
                if det['label'] == 'Dog':
                    x1 = det['bounding_box'][0]
                    y1 = det['bounding_box'][1]
                    x2 = x1 + det['bounding_box'][2]
                    y2 = y1 + det['bounding_box'][3]
                    boxes.append([x1, y1, x2, y2])
            boxes = torch.tensor(boxes, dtype=torch.float32)
        else:
            # For non-dog images or images without detection info
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        
        if self.transform:
            # Apply transforms that handle both image and boxes
            if isinstance(self.transform, dict):
                transformed = self.transform(image=np.array(image), bboxes=boxes.numpy())
                image = transformed['image']
                boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            else:
                # Regular torchvision transforms
                image = self.transform(image)
        
        num_dogs = len(boxes)
        
        # Pad boxes tensor to fixed size with zeros
        if len(boxes) < self.max_boxes:
            padding = torch.zeros((self.max_boxes - len(boxes), 4), dtype=torch.float32)
            boxes = torch.cat([boxes, padding], dim=0)
        else:
            boxes = boxes[:self.max_boxes]  # Truncate if more boxes than max_boxes
        
        return image, num_dogs, boxes

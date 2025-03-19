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


class DogDataset(Dataset):
    """
    PyTorch Dataset for dog/non-dog classification using cached metadata.
    """
    
    def __init__(self, samples, transform=None):
        """
        Initialize the dataset.
        
        Args:
            samples (list): List of sample metadata dictionaries
            transform (callable, optional): Transform to apply to images
        """
        self.samples = samples
        self.transform = transform
        
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
            tuple: (image, label) where label is 1 for dog and 0 for non-dog
        """
        sample = self.samples[idx]
        filepath = sample['filepath']
        label = sample['label']
        
        # Open and transform image
        try:
            image = Image.open(filepath).convert('RGB')
        except Exception as e:
            print(f"Error loading image {filepath}: {e}")
            # Return a black image in case of error
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

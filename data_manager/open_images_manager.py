import os
import random
import fiftyone.zoo as foz
from torchvision import transforms
from tqdm import tqdm
import pickle
from .dataset import DogDataset

from config import (
    DATA_SET_TO_USE, TRAIN_VAL_SPLIT, DATA_ROOT,REQUIRED_IMAGES,MAX_DATA_SET_SIZE)

class OpenImagesV7Manager:
    """
    Downloads and manages dog and non-dog images from Open Images V7 using FiftyOne.
    
    Uses a caching approach to avoid moving files when parameters change.
    Provides methods to get dataset statistics and create PyTorch datasets.
    """
    
    def __init__(self, force_download=False):
        """
        Initialize the Open Images manager.
        
        Args:
            data_dir (str): Directory to store the downloaded images and metadata
            force_download (bool): Force re-download even if cache exists
        """
        self.data_dir = DATA_ROOT
        self.force_download = force_download

        
        # Cache paths
        self.cache_dir = os.path.join(self.data_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        

        # self.dog_metadata_path = os.path.join(self.cache_dir, "dog_metadata_.pkl")
        # self.non_dog_metadata_path = os.path.join(self.cache_dir, "non_dog_metadata.pkl")

        cache_suffix = f"{DATA_SET_TO_USE}_{TRAIN_VAL_SPLIT}.pkl"
        dog_metadata_file_name = f"dog_metadata_{cache_suffix}"
        non_dog_metadata_file_name = f"non_dog_metadata_{cache_suffix}"
        split_cache_file_name = f"split_cache_{cache_suffix}"


        self.dog_metadata_path = os.path.join(self.cache_dir, dog_metadata_file_name)
        self.non_dog_metadata_path = os.path.join(self.cache_dir, non_dog_metadata_file_name)
        self.split_path = os.path.join(self.cache_dir, split_cache_file_name)
        
        # OpenImages class ID for dog
        self.dog_label = "Dog"
        
        # List of classes to use for non-dog images
        self.non_dog_labels = ["House", "Car", "Cat", "Person", "Tree", "Bird", "Horse"]
        print(f"MAX_DATA_SET_SIZE:{MAX_DATA_SET_SIZE} -> DATA_SET_TO_USE: {DATA_SET_TO_USE}" )
        print(f"Target download size: {REQUIRED_IMAGES} images (dog {REQUIRED_IMAGES//2}  and non-dog {REQUIRED_IMAGES//2})")

    def download_dataset(self):
        """
        Download dog and non-dog images from Open Images V7 if not in cache.
        
        Returns:
            tuple: (dog_samples, non_dog_samples) - Lists of sample metadata
        """
        # Check if metadata already exists in cache
        if not self.force_download and os.path.exists(self.dog_metadata_path) and os.path.exists(self.non_dog_metadata_path):
            print("Loading cached metadata...")
            try:
                with open(self.dog_metadata_path, 'rb') as f:
                    dog_samples = pickle.load(f)
                with open(self.non_dog_metadata_path, 'rb') as f:
                    non_dog_samples = pickle.load(f)
                
                print(f"Loaded {len(dog_samples)} dog samples and {len(non_dog_samples)} non-dog samples from cache")
                
                # Check if the cached data is valid (non-empty)
                if len(dog_samples) == 0 or len(non_dog_samples) == 0:
                    print("WARNING: Cached data is empty, forcing re-download")
                    if os.path.exists(self.dog_metadata_path):
                        os.remove(self.dog_metadata_path)
                    if os.path.exists(self.non_dog_metadata_path):
                        os.remove(self.non_dog_metadata_path)
                else:
                    return dog_samples, non_dog_samples
            except Exception as e:
                print(f"Error loading cache: {str(e)}. Forcing re-download.")
                if os.path.exists(self.dog_metadata_path):
                    os.remove(self.dog_metadata_path)
                if os.path.exists(self.non_dog_metadata_path):
                    os.remove(self.non_dog_metadata_path)
        
        # Try downloading from Open Images V7
        print("Downloading dataset from Open Images V7...")
        dog_metadata = []
        non_dog_metadata = []
        
        try:
            # Create a directory for FiftyOne datasets if it doesn't exist
            fiftyone_dir = os.path.join(self.data_dir, "fiftyone", "open-images-v7")
            os.makedirs(fiftyone_dir, exist_ok=True)


            print("Downloading dog images...")
            # Download dog images for both train and validation
            # Fix: Use only name parameter without dataset_dir
            dog_dataset = foz.load_zoo_dataset(
                "open-images-v7",
                splits=["train", "validation"],
                label_types=["detections"],
                classes=[self.dog_label],
                max_samples= REQUIRED_IMAGES // 2,
                name="dogs_openimages_v7"
            )
            
            print("Downloading non-dog images...")
            # Download non-dog images for both train and validation
            non_dog_dataset = foz.load_zoo_dataset(
                "open-images-v7",
                splits=["train", "validation"],
                label_types=["detections"],
                classes=self.non_dog_labels,
                max_samples=len(dog_dataset),
                name="non_dogs_openimages_v7"
            )
            print(f"Downloaded {len(non_dog_dataset)} non-dog images")
            
            # Filter datasets
            print("Filtering datasets...")
            
            # Get samples that contain dogs using MongoDB-style query
            dog_view = dog_dataset.match({"ground_truth.detections.label": "Dog"})
            dog_samples = list(dog_view.iter_samples())
            print(f"Found {len(dog_samples)} samples with dogs")
            
            # For non-dog dataset, exclude any images that might contain dogs
            non_dog_view = non_dog_dataset.match({"ground_truth.detections.label": {"$ne": "Dog"}})
            non_dog_samples = list(non_dog_view.iter_samples())
            print(f"Found {len(non_dog_samples)} samples without dogs")
            
            # Extract and store relevant metadata to avoid keeping the entire FiftyOne sample
            dog_metadata = []
            for sample in tqdm(dog_samples, desc="Processing dog samples"):
                detections = []
                for det in sample.ground_truth.detections:
                    if det.label == "Dog":
                        detections.append({
                            'label': det.label,
                            'bounding_box': det.bounding_box  # Already normalized [x,y,width,height]
                        })
                
                metadata = {
                    'filepath': sample.filepath,
                    'filename': os.path.basename(sample.filepath),
                    'split': sample.tags[0] if sample.tags else 'train',  # Original split (train/validation)
                    'id': sample.id,
                    'label': 1,  # 1 for dog
                    'detections': detections
                }
                dog_metadata.append(metadata)
            
            non_dog_metadata = []
            for sample in tqdm(non_dog_samples, desc="Processing non-dog samples"):
                metadata = {
                    'filepath': sample.filepath,
                    'filename': os.path.basename(sample.filepath),
                    'split': sample.tags[0] if sample.tags else 'train',  # Original split (train/validation)
                    'id': sample.id,
                    'label': 0,  # 0 for non-dog
                    'detections': []  # No dog detections
                }
                non_dog_metadata.append(metadata)
                
            # Cache metadata if we have data
            if len(dog_metadata) > 0 and len(non_dog_metadata) > 0:
                with open(self.dog_metadata_path, 'wb') as f:
                    pickle.dump(dog_metadata, f)
                
                with open(self.non_dog_metadata_path, 'wb') as f:
                    pickle.dump(non_dog_metadata, f)
                
                print("Metadata cached successfully")
                print(f"Cached {len(dog_metadata)} dog samples and {len(non_dog_metadata)} non-dog samples")
                
                return dog_metadata, non_dog_metadata
            else:
                raise ValueError("No samples found in Open Images dataset")
            
        except Exception as e:
            # Remove any potentially corrupt or incomplete cache files
            if os.path.exists(self.dog_metadata_path):
                os.remove(self.dog_metadata_path)
            if os.path.exists(self.non_dog_metadata_path):
                os.remove(self.non_dog_metadata_path)
            
            # Raise a more descriptive error
            raise ValueError(f"Failed to download dataset from Open Images: {str(e)}") from e

    def create_dataset_splits(self):
        """
        Create dataset splits based on parameters without copying files.
        
        Args:
            data_set_to_use (float): Fraction of total available dog images to use (0.0-1.0)
            train_val_split (float): Fraction of images to use for training (0.0-1.0)
            
        Returns:
            dict: Dictionary with file paths and labels for train and val splits
        """
        # Download/load dataset
        dog_samples, non_dog_samples = self.download_dataset()
        
        print(f"Creating splits with DATA_SET_TO_USE={DATA_SET_TO_USE}, TRAIN_VAL_SPLIT={TRAIN_VAL_SPLIT}...")
        
        # Set a fixed random seed for reproducibility
        random.seed(42)
        
        # Apply DATA_SET_TO_USE

        
        # Shuffle and select subset using indices to avoid duplicates
        dog_indices = list(range(len(dog_samples)))
        non_dog_indices = list(range(len(non_dog_samples)))
        
        random.shuffle(dog_indices)
        random.shuffle(non_dog_indices)
        
        # Select subset of indices
        selected_dog_indices = dog_indices[:REQUIRED_IMAGES // 2]
        selected_non_dog_indices = non_dog_indices[:REQUIRED_IMAGES // 2]
        
        # Get the actual samples
        selected_dog_samples = [dog_samples[i] for i in selected_dog_indices]
        selected_non_dog_samples = [non_dog_samples[i] for i in selected_non_dog_indices]
        
        print(f"Selected {len(selected_dog_samples)} dog samples and {len(selected_non_dog_samples)} non-dog samples")
        
        # Calculate split points
        train_dog_count = int(len(selected_dog_samples) * TRAIN_VAL_SPLIT)
        train_non_dog_count = int(len(selected_non_dog_samples) * TRAIN_VAL_SPLIT)
        
        # Split into train and validation sets (no shuffling needed as indices were already shuffled)
        train_dogs = selected_dog_samples[:train_dog_count]
        val_dogs = selected_dog_samples[train_dog_count:]
        
        train_non_dogs = selected_non_dog_samples[:train_non_dog_count]
        val_non_dogs = selected_non_dog_samples[train_non_dog_count:]
        
        print(f"Train: {len(train_dogs)} dogs, {len(train_non_dogs)} non-dogs")
        print(f"Val: {len(val_dogs)} dogs, {len(val_non_dogs)} non-dogs")
        
        # Create split information with guaranteed non-overlapping sets
        splits = {
            'train': train_dogs + train_non_dogs,
            'val': val_dogs + val_non_dogs
        }
        
        # Verify no overlap between train and val sets
        train_paths = {sample['filepath'] for sample in splits['train']}
        val_paths = {sample['filepath'] for sample in splits['val']}
        overlap = train_paths.intersection(val_paths)
        
        if overlap:
            raise ValueError(f"Found {len(overlap)} overlapping samples between train and val sets!")
        
        # Save split information

        
        with open(self.split_path, 'wb') as f:
            pickle.dump(splits, f)
        
        print(f"Splits saved to {self.split_path}")
        print(f"Verified: No overlap between train and validation sets")
        
        return splits
    
    def get_dataset_splits(self ):
        """
        Get dataset splits, either from cache or by creating new ones.
        
        Args:
            data_set_to_use (float): Fraction of total available dog images to use (0.0-1.0)
            train_val_split (float): Fraction of images to use for training (0.0-1.0)
            
        Returns:
            dict: Dictionary with file paths and labels for train and val splits
        """
        # Check if splits already exist in cache

        
        if os.path.exists(self.split_path) and not self.force_download:
            print(f"Loading cached splits from {self.split_path}...")
            try:
                with open(self.split_path, 'rb') as f:
                    splits = pickle.load(f)
                
                # Verify the splits are valid and non-empty
                if not splits or not splits.get('train') or not splits.get('val'):
                    print("WARNING: Cached splits are empty or invalid. Creating new splits...")
                    os.remove(self.split_path)  # Delete the invalid cache file
                else:
                    # Verify that the filepaths actually exist
                    sample_exists = False
                    for sample in (splits['train'][:5] + splits['val'][:5]):
                        if os.path.exists(sample['filepath']):
                            sample_exists = True
                            break
                    
                    if not sample_exists:
                        print("WARNING: No valid files found in cached splits. Creating new splits...")
                        os.remove(self.split_path)  # Delete the invalid cache fil
                    else:
                        return splits
            except Exception as e:
                print(f"Error loading cached splits: {str(e)}. Creating new splits...")
                if os.path.exists(self.split_path):
                    os.remove(self.split_path)
        
        # Create new splits
        return self.create_dataset_splits()
        
    def get_total_counts(self):
        """
        Get total counts of dog and non-dog images available in Open Images V7.
        
        Returns:
            dict: Dictionary with total counts
        """
        counts = {}
            
            
        if os.path.exists(self.dog_metadata_path) and os.path.exists(self.non_dog_metadata_path):
            with open(self.dog_metadata_path, 'rb') as f:
                all_dogs = pickle.load(f)
            with open(self.non_dog_metadata_path, 'rb') as f:
                all_non_dogs = pickle.load(f)
            
            counts['total_dogs'] = len(all_dogs)
            counts['total_non_dogs'] = len(all_non_dogs)
            
            print(f"Total downloaded dogs: {counts['total_dogs']}")
            print(f"Total downloaded non-dogs: {counts['total_non_dogs']}")
        else:
            print("No cached metadata found. Run download_dataset() first.")
            counts['total_dogs'] = 0
            counts['total_non_dogs'] = 0
                
        return counts
        
    def get_datasets(self, train_transform=None, val_transform=None):
        """
        Get PyTorch datasets for training and validation.
        
        Args:
            data_set_to_use (float): Fraction of total available dog images to use (0.0-1.0)
            train_val_split (float): Fraction of images to use for training (0.0-1.0)
            train_transform (callable, optional): Transform to apply to training images
            val_transform (callable, optional): Transform to apply to validation images
            
        Returns:
            tuple: (train_dataset, val_dataset) - PyTorch Dataset objects
        """
        # Get total counts
        total_counts = self.get_total_counts()
        
        # Get splits
        splits = self.get_dataset_splits()
        
        # Count dogs and non-dogs in each split
        

        
        # Create datasets
        train_dataset = DogDataset(splits['train'], transform=None)
        val_dataset = DogDataset(splits['val'], transform=None)
        
        return train_dataset, val_dataset

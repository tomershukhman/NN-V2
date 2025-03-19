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

MAX_DATA_SET_SIZE = 20000  # Maximum dataset size limit
DATASET_TO_USE = 0.3  # Use 80% of available data
TRAIN_VAL_SPLIT = 0.9  # 90% training, 10% validation

class OpenImagesV7Manager:
    """
    Downloads and manages dog and non-dog images from Open Images V7 using FiftyOne.
    
    Uses a caching approach to avoid moving files when parameters change.
    Provides methods to get dataset statistics and create PyTorch datasets.
    """
    
    def __init__(self, data_dir="./data"):
        """
        Initialize the Open Images manager.
        
        Args:
            data_dir (str): Directory to store the downloaded images and metadata
        """
        self.data_dir = data_dir
        
        # Cache paths
        self.cache_dir = os.path.join(data_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.dog_metadata_path = os.path.join(self.cache_dir, "dog_metadata.pkl")
        self.non_dog_metadata_path = os.path.join(self.cache_dir, "non_dog_metadata.pkl")
        
        # OpenImages class ID for dog
        self.dog_label = "Dog"
        
        # List of classes to use for non-dog images
        self.non_dog_labels = ["House", "Car", "Cat", "Person", "Tree", "Bird", "Horse"]

    def should_download_data(self):
        """Check if we need to download or update the data"""
        if not os.path.exists(self.dog_metadata_path) or not os.path.exists(self.non_dog_metadata_path):
            return True
            
        try:
            # Try loading the metadata to verify it's valid
            with open(self.dog_metadata_path, 'rb') as f:
                dog_samples = pickle.load(f)
            with open(self.non_dog_metadata_path, 'rb') as f:
                non_dog_samples = pickle.load(f)
                
            # Check if the data is valid
            if not dog_samples or not non_dog_samples:
                return True
                
            # Check if sample files exist
            for samples in [dog_samples[:5], non_dog_samples[:5]]:
                found_valid = False
                for sample in samples:
                    if os.path.exists(sample['filepath']):
                        found_valid = True
                        break
                if not found_valid:
                    return True
                    
            return False
        except Exception:
            return True

    def should_recreate_splits(self, data_set_to_use, train_val_split):
        """Check if we need to recreate the dataset splits"""
        split_path = os.path.join(
            self.cache_dir, 
            f"splits_use{data_set_to_use}_split{train_val_split}.pkl"
        )
        
        if not os.path.exists(split_path):
            return True
            
        try:
            with open(split_path, 'rb') as f:
                splits = pickle.load(f)
                
            # Verify splits are valid
            if not splits or not splits.get('train') or not splits.get('val'):
                return True
                
            # Check if files still exist
            for samples in [splits['train'][:5], splits['val'][:5]]:
                found_valid = False
                for sample in samples:
                    if os.path.exists(sample['filepath']):
                        found_valid = True
                        break
                if not found_valid:
                    return True
                    
            return False
        except Exception:
            return True

    def download_dataset(self):
        """
        Download dog and non-dog images from Open Images V7 if not in cache.
        
        Returns:
            tuple: (dog_samples, non_dog_samples) - Lists of sample metadata
        """
        # Check if metadata already exists in cache
        if not self.should_download_data():
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
            dog_dataset = foz.load_zoo_dataset(
                "open-images-v7",
                splits=["train", "validation"],
                label_types=["detections"],
                classes=[self.dog_label],
                max_samples=int(MAX_DATA_SET_SIZE * DATASET_TO_USE * TRAIN_VAL_SPLIT * 0.5),  # Convert to integer
                dataset_name="dogs_openimages_v7",
                dataset_dir=fiftyone_dir
            )
            print(f"Downloaded {len(dog_dataset)} dog images")
            
            print("Downloading non-dog images...")
            # Download non-dog images for both train and validation
            non_dog_dataset = foz.load_zoo_dataset(
                "open-images-v7",
                splits=["train", "validation"],
                label_types=["detections"],
                classes=self.non_dog_labels,
                max_samples=int(MAX_DATA_SET_SIZE * DATASET_TO_USE * TRAIN_VAL_SPLIT * 0.5),  # Convert to integer
                dataset_name="non_dogs_openimages_v7",
                dataset_dir=fiftyone_dir
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
                metadata = {
                    'filepath': sample.filepath,
                    'filename': os.path.basename(sample.filepath),
                    'split': sample.tags[0] if sample.tags else 'train',  # Original split (train/validation)
                    'id': sample.id,
                    'label': 1  # 1 for dog
                }
                dog_metadata.append(metadata)
            
            non_dog_metadata = []
            for sample in tqdm(non_dog_samples, desc="Processing non-dog samples"):
                metadata = {
                    'filepath': sample.filepath,
                    'filename': os.path.basename(sample.filepath),
                    'split': sample.tags[0] if sample.tags else 'train',  # Original split (train/validation)
                    'id': sample.id,
                    'label': 0  # 0 for non-dog
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

    def create_dataset_splits(self, data_set_to_use=0.1, train_val_split=0.8):
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
        
        print(f"Creating splits with DATA_SET_TO_USE={data_set_to_use}, TRAIN_VAL_SPLIT={train_val_split}...")
        
        # Set a fixed random seed for reproducibility
        random.seed(42)
        
        # Apply DATA_SET_TO_USE
        target_dog_count = max(10, int(len(dog_samples) * data_set_to_use))
        target_non_dog_count = max(10, int(len(non_dog_samples) * data_set_to_use))
        
        # Shuffle and select subset using indices to avoid duplicates
        dog_indices = list(range(len(dog_samples)))
        non_dog_indices = list(range(len(non_dog_samples)))
        
        random.shuffle(dog_indices)
        random.shuffle(non_dog_indices)
        
        # Select subset of indices
        selected_dog_indices = dog_indices[:target_dog_count]
        selected_non_dog_indices = non_dog_indices[:target_non_dog_count]
        
        # Get the actual samples
        selected_dog_samples = [dog_samples[i] for i in selected_dog_indices]
        selected_non_dog_samples = [non_dog_samples[i] for i in selected_non_dog_indices]
        
        print(f"Selected {len(selected_dog_samples)} dog samples and {len(selected_non_dog_samples)} non-dog samples")
        
        # Calculate split points
        train_dog_count = int(len(selected_dog_samples) * train_val_split)
        train_non_dog_count = int(len(selected_non_dog_samples) * train_val_split)
        
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
        split_path = os.path.join(
            self.cache_dir, 
            f"splits_use{data_set_to_use}_split{train_val_split}.pkl"
        )
        
        with open(split_path, 'wb') as f:
            pickle.dump(splits, f)
        
        print(f"Splits saved to {split_path}")
        print(f"Verified: No overlap between train and validation sets")
        
        return splits
    
    def get_dataset_splits(self, data_set_to_use=0.1, train_val_split=0.8):
        """
        Get dataset splits, automatically creating new ones if needed.
        
        Args:
            data_set_to_use (float): Fraction of total available dog images to use (0.0-1.0)
            train_val_split (float): Fraction of images to use for training (0.0-1.0)
            
        Returns:
            dict: Dictionary with file paths and labels for train and val splits
        """
        # Get the current configuration values from the module
        current_dataset_to_use = DATASET_TO_USE
        current_train_val_split = TRAIN_VAL_SPLIT
        
        # Check if the provided parameters match the current configuration
        if data_set_to_use != current_dataset_to_use or train_val_split != current_train_val_split:
            print(f"Note: Using provided parameters (data_set_to_use={data_set_to_use}, train_val_split={train_val_split}) "
                  f"instead of configured values (DATASET_TO_USE={current_dataset_to_use}, TRAIN_VAL_SPLIT={current_train_val_split})")
        
        # Check if we need to download data
        if self.should_download_data():
            print("Downloading dataset (missing or invalid cache)...")
            self.download_dataset()
        
        # Check if we need to recreate splits
        if self.should_recreate_splits(data_set_to_use, train_val_split):
            print(f"Creating new splits with data_set_to_use={data_set_to_use}, train_val_split={train_val_split}...")
            return self.create_dataset_splits(data_set_to_use, train_val_split)
        
        # Load existing splits
        split_path = os.path.join(
            self.cache_dir, 
            f"splits_use{data_set_to_use}_split{train_val_split}.pkl"
        )
        print(f"Loading cached splits from {split_path}...")
        with open(split_path, 'rb') as f:
            return pickle.load(f)

    def get_total_counts(self):
        """
        Get total counts of dog and non-dog images available in Open Images V7.
        
        Returns:
            dict: Dictionary with total counts
        """
        counts = {}
        
        try:
            # Get counts from FiftyOne API
            dataset_info = foz.zoo.load_zoo_dataset_info("open-images-v7")
            
            # Get total dog count
            counts['total_dogs'] = dataset_info.classes.get(self.dog_label, {}).get("count", 0)
            
            # Get total non-dog count for selected classes
            counts['total_non_dogs'] = 0
            for label in self.non_dog_labels:
                counts['total_non_dogs'] += dataset_info.classes.get(label, {}).get("count", 0)
                
            print(f"Total dogs available in Open Images V7: {counts['total_dogs']}")
            print(f"Total non-dogs available from selected classes: {counts['total_non_dogs']}")
            
        except Exception as e:
            print(f"Could not get total counts from FiftyOne API: {e}")
            print("Using downloaded sample counts instead...")
            
            # If API access fails, use local counts from cached metadata
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
        
    def get_datasets(self, data_set_to_use=0.1, train_val_split=0.8, train_transform=None, val_transform=None):
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
        splits = self.get_dataset_splits(data_set_to_use, train_val_split)
        
        # Count dogs and non-dogs in each split
        
        # Default transforms if none provided
        if train_transform is None:
            train_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Create datasets
        train_dataset = DogDataset(splits['train'], transform=train_transform)
        val_dataset = DogDataset(splits['val'], transform=val_transform)
        
        return train_dataset, val_dataset


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


def create_dataloaders(data_dir="./data", batch_size=32, num_workers=4, 
                       data_set_to_use=None, train_val_split=None):
    """
    Create PyTorch DataLoaders for dog/non-dog classification.
    
    Args:
        data_dir (str): Directory to store the dataset
        batch_size (int): Batch size for DataLoader
        num_workers (int): Number of worker processes for data loading
        data_set_to_use (float, optional): Override DATASET_TO_USE constant
        train_val_split (float, optional): Override TRAIN_VAL_SPLIT constant
    """
    # Use module constants if not overridden
    data_set_to_use = data_set_to_use if data_set_to_use is not None else DATASET_TO_USE
    train_val_split = train_val_split if train_val_split is not None else TRAIN_VAL_SPLIT
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(data_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Get dataset manager and datasets
    print("\n" + "="*50)
    print("Dataset Statistics Summary")
    print("="*50)
    
    manager = OpenImagesV7Manager(data_dir=data_dir)
    
    # Get total counts first
    total_counts = manager.get_total_counts()
    total_available = total_counts['total_dogs'] + total_counts['total_non_dogs']
    print("\nTotal Available Images in Dataset:")
    print(f"Total Available Dogs: {total_counts['total_dogs']:,}")
    print(f"Total Available Non-Dogs: {total_counts['total_non_dogs']:,}")
    print(f"Total Available Images: {total_available:,}")
        
    # Get datasets with splits info
    train_dataset, val_dataset = manager.get_datasets(data_set_to_use, train_val_split)
    
    print("\nDataset Split Configuration:")
    print(f"Dataset Usage: {data_set_to_use*100:.1f}% of available images")
    print(f"Train/Val Split: {train_val_split*100:.1f}% / {(1-train_val_split)*100:.1f}%")
    
    # Count dogs and non-dogs in train dataset
    train_dogs = sum(1 for sample in train_dataset.samples if sample['label'] == 1)
    train_non_dogs = sum(1 for sample in train_dataset.samples if sample['label'] == 0)
    train_total = len(train_dataset.samples)
    
    # Count dogs and non-dogs in val dataset
    val_dogs = sum(1 for sample in val_dataset.samples if sample['label'] == 1)
    val_non_dogs = sum(1 for sample in val_dataset.samples if sample['label'] == 0)
    val_total = len(val_dataset.samples)
    
    print("\nCurrent Dataset Split Statistics:")
    print("Training Set:")
    print(f"  - Dogs: {train_dogs:,} ({train_dogs/train_total*100:.1f}% of training set)")
    print(f"  - Non-Dogs: {train_non_dogs:,} ({train_non_dogs/train_total*100:.1f}% of training set)")
    print(f"  - Total: {train_total:,}")
    
    print("\nValidation Set:")
    print(f"  - Dogs: {val_dogs:,} ({val_dogs/val_total*100:.1f}% of validation set)")
    print(f"  - Non-Dogs: {val_non_dogs:,} ({val_non_dogs/val_total*100:.1f}% of validation set)")
    print(f"  - Total: {val_total:,}")
    
    total_dogs_used = train_dogs + val_dogs
    total_non_dogs_used = train_non_dogs + val_non_dogs
    total_used = total_dogs_used + total_non_dogs_used
    
    print("\nTotal Used in Current Split:")
    print(f"  - Dogs Being Used: {total_dogs_used:,} ({total_dogs_used/total_counts['total_dogs']*100:.1f}% of available dogs)")
    print(f"  - Non-Dogs Being Used: {total_non_dogs_used:,} ({total_non_dogs_used/total_counts['total_non_dogs']*100:.1f}% of available non-dogs)")
    print(f"  - Total Images Being Used: {total_used:,} ({total_used/total_available*100:.1f}% of available images)")
    print("="*50 + "\n")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare dog/non-dog dataset from Open Images V7")
    parser.add_argument("--data_dir", default="./data", help="Directory to store the dataset")
    parser.add_argument("--data_set_to_use", type=float, default=DATASET_TO_USE, help="Fraction of total dog images to use")
    parser.add_argument("--train_val_split", type=float, default=TRAIN_VAL_SPLIT, help="Fraction for training split")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--stats_only", action="store_true", help="Only show dataset statistics without creating DataLoaders")
    
    args = parser.parse_args()
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        data_set_to_use=args.data_set_to_use,
        train_val_split=args.train_val_split
    )

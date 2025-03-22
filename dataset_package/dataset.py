"""
Core dataset class for multi-class object detection.
"""
import os
import torch
import numpy as np
from PIL import Image as PILImage
import fiftyone as fo
import fiftyone.zoo as foz
import random
from collections import Counter
import logging
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from config import DATA_ROOT, TRAIN_VAL_SPLIT

logger = logging.getLogger('dog_detector')

# Class names used for detection labels
CLASS_NAMES = ["background", "dog", "person"]

class DogDetectionDataset(Dataset):
    def __init__(self, root=DATA_ROOT, split='train', transform=None, download=False, max_samples=25000):
        """
        Multi-class Object Detection Dataset using Open Images
        Args:
            root (str): Root directory for the dataset
            split (str): 'train' or 'validation'
            transform (callable, optional): Albumentations transform to be applied on a sample
            download (bool): If True, downloads the dataset from the internet
            max_samples (int): Maximum number of samples to load (default: 25000)
        """
        self.transform = transform
        self.root = os.path.abspath(root)
        self.split = "validation" if split == "val" else split
        self.samples = []
        self.objects_per_image = []
        self.max_samples = max_samples
        
        # We'll use a single cache file for all data
        self.cache_file = os.path.join(self.root, 'multiclass_detection_cache.pt')
        
        # Ensure data directory exists
        os.makedirs(self.root, exist_ok=True)
        
        # Load from cache or download dataset
        self._initialize_dataset(download)

    def _initialize_dataset(self, download):
        """Initialize the dataset, either from cache or by downloading/processing the data"""
        # Try to load from cache first
        if os.path.exists(self.cache_file):
            self._load_from_cache()
        else:
            # If cache doesn't exist, load from dataset
            self._load_from_source(download)
        
        if len(self.samples) == 0:
            raise RuntimeError("No valid images found in the dataset")
    
    def _load_from_cache(self):
        """Load the dataset from the cached file"""
        logger.info(f"Loading dataset from cache: {self.cache_file}")
        cache_data = torch.load(self.cache_file)
        all_samples = cache_data['samples']
        all_objects_per_image = cache_data['objects_per_image']
        
        # Apply max_samples limit
        total_samples = len(all_samples)
        all_samples, all_objects_per_image = self._apply_sample_limits(all_samples, all_objects_per_image)
        
        # Split into train/val using TRAIN_VAL_SPLIT with stratification
        train_samples, train_counts, val_samples, val_counts = self._stratified_split(
            all_samples, all_objects_per_image
        )
        
        # Assign to the appropriate split
        if self.split == 'train':
            self.samples = list(train_samples)
            self.objects_per_image = list(train_counts)
        else:  # validation split
            self.samples = list(val_samples)
            self.objects_per_image = list(val_counts)
        
        # Show distribution for current split (only in debug level)
        if logger.isEnabledFor(logging.DEBUG):
            curr_distribution = Counter(self.objects_per_image)
            logger.debug(f"Object count distribution for {self.split} split: {dict(curr_distribution)}")
        
        logger.info(f"Loaded {len(self.samples)} samples for {self.split} split")

    def _apply_sample_limits(self, all_samples, all_objects_per_image):
        """Apply max_samples limit while maintaining stratification"""
        total_samples = len(all_samples)
        
        # Sort samples by number of objects to analyze distribution
        samples_with_count = list(zip(all_samples, all_objects_per_image))
        # Shuffle with fixed seed for reproducibility before selecting subset
        random.seed(42)
        random.shuffle(samples_with_count)
        
        if logger.isEnabledFor(logging.DEBUG):
            object_count_distribution = Counter(all_objects_per_image)
            logger.debug(f"Original object count distribution: {dict(object_count_distribution)}")
        
        # Limit total samples while maintaining stratification
        if self.max_samples and len(samples_with_count) > self.max_samples:
            logger.info(f"Limiting dataset to {self.max_samples} samples")
            # Group samples by object count
            samples_by_count = {}
            for sample, count in zip(all_samples, all_objects_per_image):
                if count not in samples_by_count:
                    samples_by_count[count] = []
                samples_by_count[count].append((sample, count))
            
            # Calculate proportions to maintain for each group
            selected_samples = []
            remaining_samples = self.max_samples
            
            # First pass: ensure minimum representation
            min_per_group = max(1, self.max_samples // (len(samples_by_count) * 10))  # At least 10% per group
            for count, samples in samples_by_count.items():
                num_to_select = min(min_per_group, len(samples))
                selected_samples.extend(samples[:num_to_select])
                remaining_samples -= num_to_select
            
            # Second pass: distribute remaining samples proportionally
            if remaining_samples > 0:
                total_count = len(samples_with_count)
                for count, samples in samples_by_count.items():
                    remaining_in_group = samples[min_per_group:]
                    if not remaining_in_group:
                        continue
                    proportion = len(remaining_in_group) / (total_count - len(selected_samples))
                    num_additional = int(remaining_samples * proportion)
                    selected_samples.extend(remaining_in_group[:num_additional])
            
            # Re-shuffle the selected samples
            random.shuffle(selected_samples)
            
            # Unpack the samples and counts
            all_samples = [s[0] for s in selected_samples]
            all_objects_per_image = [s[1] for s in selected_samples]
            
            logger.info(f"Selected {len(all_samples)} samples after limiting")
        
        # Report distribution after filtering (only in debug level)
        if logger.isEnabledFor(logging.DEBUG):
            filtered_object_distribution = Counter(all_objects_per_image)
            logger.debug(f"Filtered object count distribution: {dict(filtered_object_distribution)}")
        
        logger.info(f"Using {len(all_samples)} samples ({len(all_samples)/total_samples:.1%} of original data)")
        return all_samples, all_objects_per_image

    def _stratified_split(self, all_samples, all_objects_per_image):
        """Split dataset into train/val using stratification by object count"""
        train_samples = []
        train_counts = []
        val_samples = []
        val_counts = []
        
        # Group by object count for stratified split
        samples_by_count = {}
        for sample, count in zip(all_samples, all_objects_per_image):
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
        
        return train_samples, train_counts, val_samples, val_counts

    def _load_from_source(self, download):
        """Load dataset from source (FiftyOne) and cache it"""
        original_dir = fo.config.dataset_zoo_dir
        fo.config.dataset_zoo_dir = self.root
        
        try:
            # Load or download the dataset
            dataset_name = "open-images-v7-full"
            dataset = self._get_dataset(dataset_name, download)
            
            # Process the dataset and prepare samples
            dog_samples, dog_object_counts, person_samples, person_object_counts = self._process_samples(dataset)
            
            # Balance the dataset between dogs and people
            all_samples, all_objects_per_image = self._balance_samples(
                dog_samples, dog_object_counts, person_samples, person_object_counts
            )
            
            total_samples = len(all_samples)
            if total_samples == 0:
                raise RuntimeError("No valid images found in the dataset")
            
            # Save combined dataset to cache
            logger.info(f"Saving dataset to cache: {self.cache_file}")
            torch.save({
                'samples': all_samples,
                'objects_per_image': all_objects_per_image
            }, self.cache_file)
            
            # Assign samples to this instance
            self.samples = list(all_samples)
            self.objects_per_image = list(all_objects_per_image)
            
            logger.info(f"Loaded {len(self.samples)} samples for {self.split} split")
                
        except Exception as e:
            logger.error(f"Error initializing dataset: {e}")
            raise
        finally:
            fo.config.dataset_zoo_dir = original_dir

    def _get_dataset(self, dataset_name, download):
        """Get or download the FiftyOne dataset"""
        try:
            dataset = fo.load_dataset(dataset_name)
            logger.info(f"Loaded existing dataset: {dataset_name}")
        except fo.core.dataset.DatasetNotFoundError:
            if download:
                logger.info("Downloading Open Images dataset with dog and person classes...")
                # Download dataset
                dataset = foz.load_zoo_dataset(
                    "open-images-v7",
                    splits=["train", "validation"],
                    label_types=["detections"],
                    classes=["Dog", "Person"],
                    dataset_name=dataset_name,
                    max_samples=25000
                )
                logger.info(f"Downloaded dataset to {fo.config.dataset_zoo_dir}")
            else:
                raise RuntimeError(f"Dataset {dataset_name} not found and download=False")
        
        # Filter for the requested split
        if dataset is not None:
            split_dataset = dataset.match({"split": self.split})
            logger.info(f"Processing {split_dataset.name} split with {len(split_dataset)} samples")
            
            if len(split_dataset) == 0:
                logger.error(f"No samples found for split: {self.split}")
                raise RuntimeError(f"No samples found for split: {self.split}")
            
            return split_dataset
        return None

    def _process_samples(self, dataset):
        """Process samples from the dataset and extract dog and person samples"""
        dog_samples = []
        person_samples = []
        dog_object_counts = []
        person_object_counts = []
        skipped_labels = Counter()
        
        # Process samples from the filtered dataset
        for sample in dataset.iter_samples():
            if hasattr(sample, 'ground_truth') and sample.ground_truth is not None:
                detections = sample.ground_truth.detections
                if detections:
                    img_path = sample.filepath
                    if os.path.exists(img_path):
                        boxes = []
                        labels = []
                        has_dog = False
                        has_person = False
                        
                        for det in detections:
                            try:
                                # Only add detections for labels we care about
                                if det.label.lower() in [c.lower() for c in CLASS_NAMES]:
                                    class_idx = CLASS_NAMES.index(det.label.lower())
                                    if class_idx > 0:  # Skip background class
                                        boxes.append([
                                            det.bounding_box[0],
                                            det.bounding_box[1],
                                            det.bounding_box[0] + det.bounding_box[2],
                                            det.bounding_box[1] + det.bounding_box[3]
                                        ])
                                        labels.append(class_idx)
                                        if class_idx == 1:  # Dog
                                            has_dog = True
                                        elif class_idx == 2:  # Person
                                            has_person = True
                                else:
                                    skipped_labels[det.label.lower()] += 1
                            except ValueError:
                                # Should not happen now, but keep track just in case
                                skipped_labels[det.label.lower()] += 1
                                continue
                        
                        if boxes:  # Only add if we have valid detections
                            sample_data = (img_path, boxes, labels)
                            num_objects = len(boxes)
                            if has_dog and not has_person:
                                dog_samples.append(sample_data)
                                dog_object_counts.append(num_objects)
                            elif has_person and not has_dog:
                                person_samples.append(sample_data)
                                person_object_counts.append(num_objects)
        
        # Log skipped labels if there are any (at debug level)
        if skipped_labels and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Skipped labels: {dict(skipped_labels)}")
            
        return dog_samples, dog_object_counts, person_samples, person_object_counts

    def _balance_samples(self, dog_samples, dog_object_counts, person_samples, person_object_counts):
        """Balance the dataset between dog and person samples"""
        min_class_size = min(len(dog_samples), len(person_samples))
        max_samples_per_class = min(min_class_size, self.max_samples // 2)  # Ensure equal split
        
        logger.info(f"Found {len(dog_samples)} dog samples and {len(person_samples)} person samples")
        logger.info(f"Balancing dataset to {max_samples_per_class} samples per class")
        
        # Randomly sample equal numbers from each class
        random.seed(42)  # For reproducibility
        dog_indices = random.sample(range(len(dog_samples)), max_samples_per_class)
        person_indices = random.sample(range(len(person_samples)), max_samples_per_class)
        
        all_samples = []
        all_objects_per_image = []
        
        # Add balanced samples
        for idx in dog_indices:
            all_samples.append(dog_samples[idx])
            all_objects_per_image.append(dog_object_counts[idx])
        
        for idx in person_indices:
            all_samples.append(person_samples[idx])
            all_objects_per_image.append(person_object_counts[idx])
        
        # Shuffle the combined dataset
        combined = list(zip(all_samples, all_objects_per_image))
        random.shuffle(combined)
        all_samples, all_objects_per_image = zip(*combined)
        
        return list(all_samples), list(all_objects_per_image)

    def get_sample_weights(self):
        """
        Generate sample weights to balance single and multi-object examples during training.
        This helps ensure the model sees enough multi-object examples.
        """
        if self.split != 'train':
            return None
            
        # Count occurrences of each object count
        object_counts = Counter(self.objects_per_image)
        total_samples = len(self.objects_per_image)
        
        # Calculate inverse frequency for each count
        weights = []
        for count in self.objects_per_image:
            # Weight inversely proportional to frequency, with additional boost for multi-object images
            weight = total_samples / (object_counts[count] * len(object_counts))
            # Additional boost for multi-object cases
            if count > 1:
                weight *= 1.5  # Boost multi-object samples
            weights.append(weight)
            
        return weights

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index):
        img_path, boxes, labels = self.samples[index]
        object_count = self.objects_per_image[index]
        
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
            
            # Apply Albumentations transform if provided
            if self.transform:
                transformed = self.transform(
                    image=img,
                    bboxes=boxes_abs,
                    labels=labels
                )
                img = transformed['image']
                boxes_abs = transformed['bboxes']
                labels = transformed['labels']
                
                # After transformation, convert absolute coordinates back to normalized values
                _, new_h, new_w = img.shape
                normalized_boxes = []
                for box in boxes_abs:
                    x_min, y_min, x_max, y_max = box
                    normalized_boxes.append([
                        x_min / new_w,
                        y_min / new_h,
                        x_max / new_w,
                        y_max / new_h
                    ])
                boxes_tensor = torch.tensor(normalized_boxes, dtype=torch.float32)
                target = {
                    'boxes': boxes_tensor,
                    'labels': torch.tensor(labels, dtype=torch.long),
                    'scores': torch.ones(len(labels)),
                    'object_count': object_count
                }
            else:
                # If no transform, manually convert image to tensor and boxes to normalized values
                img = ToTensorV2()(image=img)['image']
                boxes_tensor = torch.tensor(normalized_boxes, dtype=torch.float32)
                target = {
                    'boxes': boxes_tensor,
                    'labels': torch.tensor(labels, dtype=torch.long),
                    'scores': torch.ones(len(labels)),
                    'object_count': object_count
                }
            
            return img, target
            
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}")
            # Return a dummy sample
            img = torch.zeros((3, 224, 224), dtype=torch.float32)
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.long),
                'scores': torch.zeros(0),
                'object_count': 0
            }
            return img, target
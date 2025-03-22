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

# Class names used for detection labels; index 0 is background (skipped)
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
        self.stats = {'dog': 0, 'person': 0, 'total_images': 0}
        
        # We'll use a single cache file for all data
        self.cache_file = os.path.join(self.root, 'multiclass_detection_cache.pt')
        
        # Ensure data directory exists
        os.makedirs(self.root, exist_ok=True)
        
        # Load from cache or download dataset
        self._initialize_dataset(download)

    def _initialize_dataset(self, download):
        """Initialize the dataset, either from cache or by downloading/processing the data"""
        if os.path.exists(self.cache_file):
            self._load_from_cache()
            # Print stats right after loading
            self.print_stats()
        else:
            self._load_from_source(download)
            
        if len(self.samples) == 0:
            raise RuntimeError("No valid images found in the dataset")
    
    def _load_from_cache(self):
        """Load the dataset from the cached file"""
        logger.info(f"Loading dataset from cache: {self.cache_file}")
        cache_data = torch.load(self.cache_file)
        all_samples = cache_data['samples']
        all_objects_per_image = cache_data['objects_per_image']
        
        total_samples = len(all_samples)
        all_samples, all_objects_per_image = self._apply_sample_limits(all_samples, all_objects_per_image)
        
        train_samples, train_counts, val_samples, val_counts = self._stratified_split(
            all_samples, all_objects_per_image
        )
        
        # Reset stats before processing
        self.stats = {'dog': 0, 'person': 0, 'total_images': 0}
        
        if self.split == 'train':
            self.samples = list(train_samples)
            self.objects_per_image = list(train_counts)
        else:
            self.samples = list(val_samples)
            self.objects_per_image = list(val_counts)
        
        # Calculate stats for the current split
        for _, _, labels in self.samples:
            self.stats['total_images'] += 1
            for label in labels:
                if label == CLASS_NAMES.index('dog'):
                    self.stats['dog'] += 1
                elif label == CLASS_NAMES.index('person'):
                    self.stats['person'] += 1
        
        if logger.isEnabledFor(logging.DEBUG):
            curr_distribution = Counter(self.objects_per_image)
            logger.debug(f"Object count distribution for {self.split} split: {dict(curr_distribution)}")
        
        logger.info(f"Loaded {len(self.samples)} samples for {self.split} split")

    def _apply_sample_limits(self, all_samples, all_objects_per_image):
        """Apply max_samples limit while maintaining stratification"""
        total_samples = len(all_samples)
        samples_with_count = list(zip(all_samples, all_objects_per_image))
        random.seed(42)
        random.shuffle(samples_with_count)
        
        if logger.isEnabledFor(logging.DEBUG):
            object_count_distribution = Counter(all_objects_per_image)
            logger.debug(f"Original object count distribution: {dict(object_count_distribution)}")
        
        if self.max_samples and len(samples_with_count) > self.max_samples:
            logger.info(f"Limiting dataset to {self.max_samples} samples")
            samples_by_count = {}
            for sample, count in samples_with_count:
                samples_by_count.setdefault(count, []).append((sample, count))
            
            selected_samples = []
            remaining_samples = self.max_samples
            min_per_group = max(1, self.max_samples // (len(samples_by_count) * 10))
            for count, samples in samples_by_count.items():
                num_to_select = min(min_per_group, len(samples))
                selected_samples.extend(samples[:num_to_select])
                remaining_samples -= num_to_select
            
            if remaining_samples > 0:
                total_count = len(samples_with_count)
                for count, samples in samples_by_count.items():
                    remaining_in_group = samples[min_per_group:]
                    if not remaining_in_group:
                        continue
                    proportion = len(remaining_in_group) / (total_count - len(selected_samples))
                    num_additional = int(remaining_samples * proportion)
                    selected_samples.extend(remaining_in_group[:num_additional])
            
            random.shuffle(selected_samples)
            all_samples = [s[0] for s in selected_samples]
            all_objects_per_image = [s[1] for s in selected_samples]
            logger.info(f"Selected {len(all_samples)} samples after limiting")
        
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
        
        samples_by_count = {}
        for sample, count in zip(all_samples, all_objects_per_image):
            samples_by_count.setdefault(count, []).append((sample, count))
        
        for count, samples in samples_by_count.items():
            train_size = int(len(samples) * TRAIN_VAL_SPLIT)
            train_group = samples[:train_size]
            val_group = samples[train_size:]
            
            train_samples.extend([s[0] for s in train_group])
            train_counts.extend([s[1] for s in train_group])
            val_samples.extend([s[0] for s in val_group])
            val_counts.extend([s[1] for s in val_group])
        
        random.shuffle(train_samples)
        random.shuffle(val_samples)
        return train_samples, train_counts, val_samples, val_counts

    def _load_from_source(self, download):
        """Load dataset from source (FiftyOne) and cache it"""
        original_dir = fo.config.dataset_zoo_dir
        fo.config.dataset_zoo_dir = self.root
        
        try:
            dataset_name = "open-images-v7-full"
            dataset = self._get_dataset(dataset_name, download)
            
            all_samples, all_object_counts = self._process_samples(dataset)
            
            total_samples = len(all_samples)
            if total_samples == 0:
                raise RuntimeError("No valid images found in the dataset")
            
            all_samples, all_object_counts = self._apply_sample_limits(all_samples, all_object_counts)
            
            logger.info(f"Saving dataset to cache: {self.cache_file}")
            torch.save({
                'samples': all_samples,
                'objects_per_image': all_object_counts
            }, self.cache_file)
            
            self.samples = list(all_samples)
            self.objects_per_image = list(all_object_counts)
            
            logger.info(f"Loaded {len(self.samples)} samples for {self.split} split")
                
        except Exception as e:
            logger.error(f"Error initializing dataset: {e}")
            raise
        finally:
            fo.config.dataset_zoo_dir = original_dir

    def _get_dataset(self, dataset_name, download):
        """Get or download the FiftyOne dataset and filter by split using file path"""
        try:
            dataset = fo.load_dataset(dataset_name)
            logger.info(f"Loaded existing dataset: {dataset_name}")
        except fo.core.dataset.DatasetNotFoundError:
            if download:
                logger.info("Downloading Open Images dataset with dog and person classes...")
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
        
        if dataset is not None:
            # Instead of filtering by a field, filter based on file path if possible.
            if self.split == "train":
                split_dataset = dataset.match({"filepath": {"$regex": r"/train/"}})
            else:
                split_dataset = dataset.match({"filepath": {"$regex": r"/validation/"}})
            
            logger.info(f"Processing {split_dataset.name} split with {len(split_dataset)} samples")
            if len(split_dataset) == 0:
                logger.error(f"No samples found for split: {self.split}")
                raise RuntimeError(f"No samples found for split: {self.split}")
            
            return split_dataset
        return None

    def _process_samples(self, dataset):
        """
        Process samples from the dataset and extract images with valid dog and/or person detections.
        Returns:
            all_samples (list): List of tuples (img_path, boxes, labels)
            all_object_counts (list): Number of detections per image
        """
        all_samples = []
        all_object_counts = []
        skipped_labels = Counter()
        valid_labels = [c.lower() for c in CLASS_NAMES]
        
        # Reset stats for this split
        self.stats = {'dog': 0, 'person': 0, 'total_images': 0}
        
        for sample in dataset.iter_samples():
            if hasattr(sample, 'ground_truth') and sample.ground_truth is not None:
                detections = sample.ground_truth.detections
                if detections:
                    img_path = sample.filepath
                    if os.path.exists(img_path):
                        boxes = []
                        labels = []
                        has_valid_objects = False
                        
                        for det in detections:
                            label_lower = det.label.lower()
                            if label_lower in valid_labels:
                                class_idx = CLASS_NAMES.index(label_lower)
                                if class_idx > 0:
                                    boxes.append([
                                        det.bounding_box[0],
                                        det.bounding_box[1],
                                        det.bounding_box[0] + det.bounding_box[2],
                                        det.bounding_box[1] + det.bounding_box[3]
                                    ])
                                    labels.append(class_idx)
                                    has_valid_objects = True
                                    if label_lower == 'dog':
                                        self.stats['dog'] += 1
                                    elif label_lower == 'person':
                                        self.stats['person'] += 1
                            else:
                                skipped_labels[label_lower] += 1
                                
                        if has_valid_objects:
                            sample_data = (img_path, boxes, labels)
                            num_objects = len(boxes)
                            all_samples.append(sample_data)
                            all_object_counts.append(num_objects)
                            self.stats['total_images'] += 1
        
        if skipped_labels and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Skipped labels: {dict(skipped_labels)}")
        return all_samples, all_object_counts

    def get_sample_weights(self):
        """
        Generate sample weights to balance single and multi-object examples during training.
        """
        if self.split != 'train':
            return None
            
        object_counts = Counter(self.objects_per_image)
        total_samples = len(self.objects_per_image)
        weights = []
        for count in self.objects_per_image:
            weight = total_samples / (object_counts[count] * len(object_counts))
            if count > 1:
                weight *= 1.5
            weights.append(weight)
        return weights

    def print_stats(self):
        """Print statistics about the dataset split"""
        logger.info("=" * 50)
        logger.info(f"Dataset Statistics for {self.split} split:")
        logger.info("-" * 50)
        logger.info(f"Total images: {self.stats['total_images']}")
        logger.info(f"Total dogs: {self.stats['dog']}")
        logger.info(f"Total persons: {self.stats['person']}")
        logger.info(f"Average dogs per image: {self.stats['dog'] / self.stats['total_images']:.2f}")
        logger.info(f"Average persons per image: {self.stats['person'] / self.stats['total_images']:.2f}")
        logger.info("=" * 50)

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index):
        img_path, boxes, labels = self.samples[index]
        object_count = self.objects_per_image[index]
        
        try:
            img = np.array(PILImage.open(img_path).convert('RGB'))
            h, w = img.shape[:2]
            
            normalized_boxes = []
            for box in boxes:
                x_min, y_min, x_max, y_max = box
                if x_max > 1 or y_max > 1:
                    x_min, x_max = x_min / w, x_max / w
                    y_min, y_max = y_min / h, y_max / h
                x_min = max(0.0, min(0.99, float(x_min)))
                y_min = max(0.0, min(0.99, float(y_min)))
                x_max = max(min(1.0, float(x_max)), x_min + 0.01)
                y_max = max(min(1.0, float(y_max)), y_min + 0.01)
                normalized_boxes.append([x_min, y_min, x_max, y_max])
            
            boxes_abs = [[x_min * w, y_min * h, x_max * w, y_max * h] 
                         for x_min, y_min, x_max, y_max in normalized_boxes]
            
            if self.transform:
                transformed = self.transform(
                    image=img,
                    bboxes=boxes_abs,
                    labels=labels
                )
                img = transformed['image']
                boxes_abs = transformed['bboxes']
                labels = transformed['labels']
                
                _, new_h, new_w = img.shape
                normalized_boxes = [[x_min / new_w, y_min / new_h, x_max / new_w, y_max / new_h] 
                                    for x_min, y_min, x_max, y_max in boxes_abs]
                boxes_tensor = torch.tensor(normalized_boxes, dtype=torch.float32)
                target = {
                    'boxes': boxes_tensor,
                    'labels': torch.tensor(labels, dtype=torch.long),
                    'scores': torch.ones(len(labels)),
                    'object_count': object_count
                }
            else:
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
            img = torch.zeros((3, 224, 224), dtype=torch.float32)
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.long),
                'scores': torch.zeros(0),
                'object_count': 0
            }
            return img, target

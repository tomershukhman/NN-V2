# dataset.py
import os
import cv2
import numpy as np
import torch
import fiftyone as fo
import fiftyone.zoo as foz
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('dog_detector')

# Mapping from class name to numerical label
LABEL_MAP = {"Person": 0, "Dog": 1}

class ObjectDetectionDataset(torch  .utils.data.Dataset):
    """
    Custom Dataset for object detection.
    Expects a list of FiftyOne samples.
    Loads images, converts normalized bounding boxes to absolute Pascal VOC format,
    and applies provided augmentations.
    """
    def __init__(self, samples_data, transforms, label_map):
        """
        Args:
            samples_data: List of dictionaries containing sample data (filepath and annotations)
            transforms: Albumentations transform.
            label_map: Dictionary mapping class names to numeric labels.
        """
        self.samples_data = samples_data
        self.transforms = transforms
        self.label_map = label_map

    def __len__(self):
        return len(self.samples_data)

    def __getitem__(self, idx):
        sample_data = self.samples_data[idx]
        image_path = sample_data["filepath"]
        
        # Load image using cv2 (BGR) and convert to RGB
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

def download_and_prepare_dataset(max_samples=500):
    """
    Downloads a subset of the Open Images dataset (v6) using FiftyOne,
    filtered to include only "Person" and "Dog" detections.
    """
    dataset_name = "open-images-person-dog"
    
    try:
        # Try to load existing dataset
        dataset = fo.load_dataset(dataset_name)
        logger.info(f"Loaded existing dataset '{dataset_name}' with {len(dataset)} samples")
    except ValueError:
        # If dataset doesn't exist, download it
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
    
    return dataset

def create_balanced_samples(foset):
    """
    Create a balanced list of samples based on the presence of each class.
    Computes per-sample weights inversely proportional to class frequency.
    Returns a list of sample data dictionaries and a corresponding list of weights.
    """
    logger.info(f"Creating balanced samples from {len(foset)} FiftyOne samples")
    
    if len(foset) == 0:
        logger.warning("Empty dataset provided to create_balanced_samples")
        return [], []
    
    # Analyze dataset composition before processing
    logger.info("=== Dataset Statistics ===")
    logger.info(f"Total samples in dataset: {len(foset)}")
    
    # Sample count per class
    class_counts = {"Person": 0, "Dog": 0}
    # Images with both classes
    both_classes = 0
    # Counts for images with multiple instances of each class
    multi_person = 0
    multi_dog = 0
    # Collect bounding box sizes
    person_bbox_sizes = []
    dog_bbox_sizes = []
    
    # First analyze dataset composition
    for sample in foset:
        has_person = False
        has_dog = False
        person_count = 0
        dog_count = 0
        
        if hasattr(sample, 'ground_truth') and hasattr(sample.ground_truth, 'detections'):
            for det in sample.ground_truth.detections:
                if hasattr(det, 'label'):
                    if det.label == "Person":
                        has_person = True
                        person_count += 1
                        # Collect bbox size info (area as percentage of image)
                        if hasattr(det, 'bounding_box'):
                            x, y, w, h = det.bounding_box
                            area = w * h  # Normalized area (0-1)
                            person_bbox_sizes.append(area)
                    elif det.label == "Dog":
                        has_dog = True
                        dog_count += 1
                        # Collect bbox size info
                        if hasattr(det, 'bounding_box'):
                            x, y, w, h = det.bounding_box
                            area = w * h  # Normalized area (0-1)
                            dog_bbox_sizes.append(area)
            
            if has_person:
                class_counts["Person"] += 1
            if has_dog:
                class_counts["Dog"] += 1
            if has_person and has_dog:
                both_classes += 1
            if person_count > 1:
                multi_person += 1
            if dog_count > 1:
                multi_dog += 1
    
    # Display statistics
    logger.info(f"Images with Person: {class_counts['Person']} ({class_counts['Person']/len(foset)*100:.1f}%)")
    logger.info(f"Images with Dog: {class_counts['Dog']} ({class_counts['Dog']/len(foset)*100:.1f}%)")
    logger.info(f"Images with both Person and Dog: {both_classes} ({both_classes/len(foset)*100:.1f}%)")
    logger.info(f"Images with multiple People: {multi_person} ({multi_person/len(foset)*100:.1f}%)")
    logger.info(f"Images with multiple Dogs: {multi_dog} ({multi_dog/len(foset)*100:.1f}%)")
    
    # Calculate average bbox sizes
    if person_bbox_sizes:
        avg_person_size = sum(person_bbox_sizes) / len(person_bbox_sizes)
        logger.info(f"Average Person bbox size: {avg_person_size*100:.2f}% of image area")
    if dog_bbox_sizes:
        avg_dog_size = sum(dog_bbox_sizes) / len(dog_bbox_sizes)
        logger.info(f"Average Dog bbox size: {avg_dog_size*100:.2f}% of image area")
    
    logger.info("========================")
    
    valid_samples_data = []  # Will store dictionaries instead of FiftyOne sample objects
    valid_sample_flags = []  # Each element is a tuple (has_person, has_dog)
    n_person = 0
    n_dog = 0
    
    # Now continue with the regular sample processing
    for sample in foset:
        has_person = False
        has_dog = False
        is_valid = False
        
        # Extract bounding boxes and category IDs for this sample
        sample_bboxes = []
        sample_category_ids = []
        
        # For Open Images dataset, detections may be in the ground_truth field
        if hasattr(sample, 'ground_truth'):
            detections = sample.ground_truth
            
            # Ensure detections has the expected structure
            if hasattr(detections, 'detections'):
                for det in detections.detections:
                    if hasattr(det, 'label'):
                        label = det.label
                        if label in LABEL_MAP:
                            # Convert normalized bbox [x, y, w, h] to absolute Pascal VOC [x_min, y_min, x_max, y_max]
                            try:
                                x, y, bw, bh = det.bounding_box
                                # Store bbox in normalized format, will convert to absolute in __getitem__
                                sample_bboxes.append([x, y, bw, bh])
                                sample_category_ids.append(LABEL_MAP[label])
                                
                                if label == "Person":
                                    has_person = True
                                    is_valid = True
                                elif label == "Dog":
                                    has_dog = True
                                    is_valid = True
                            except (ValueError, TypeError, AttributeError) as e:
                                logger.warning(f"Error processing bbox: {e}")
                                continue
            
            # Only include samples with at least one valid detection
            if is_valid:
                # Create a serializable dictionary instead of using the FiftyOne sample
                sample_data = {
                    "filepath": sample.filepath,
                    "bboxes": sample_bboxes,
                    "category_ids": sample_category_ids,
                    "id": str(sample.id) if hasattr(sample, "id") else "unknown"
                }
                valid_samples_data.append(sample_data)
                valid_sample_flags.append((has_person, has_dog))
                if has_person:
                    n_person += 1
                if has_dog:
                    n_dog += 1
    
    logger.info(f"Found {len(valid_samples_data)} valid samples with detections")
    logger.info(f"Class distribution: Person: {n_person}, Dog: {n_dog}")
    
    # Handle the case where no valid samples are found
    if len(valid_samples_data) == 0:
        logger.error("No valid samples found with the required detections!")
        return [], []
    
    # Calculate weights based on class frequency
    w_person = 1.0 / max(n_person, 1)
    w_dog = 1.0 / max(n_dog, 1)
    
    sample_weights = []
    for has_person, has_dog in valid_sample_flags:
        weight = 0.0
        if has_person:
            weight = max(weight, w_person)
        if has_dog:
            weight = max(weight, w_dog)
        sample_weights.append(weight)
    
    # If all weights are zero (unlikely), use uniform weights
    if all(w == 0 for w in sample_weights):
        logger.warning("All sample weights are zero. Using uniform weights.")
        sample_weights = [1.0 / len(valid_samples_data)] * len(valid_samples_data)
    
    return valid_samples_data, sample_weights

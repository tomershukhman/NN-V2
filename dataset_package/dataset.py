# dataset.py
import os
import cv2
import numpy as np
import torch
import fiftyone as fo
import fiftyone.zoo as foz

# Mapping from class name to numerical label
LABEL_MAP = {"Person": 0, "Dog": 1}

class ObjectDetectionDataset(torch.utils.data.Dataset):
    """
    Custom Dataset for object detection.
    Expects a list of FiftyOne samples.
    Loads images, converts normalized bounding boxes to absolute Pascal VOC format,
    and applies provided augmentations.
    """
    def __init__(self, samples, transforms, label_map):
        """
        Args:
            samples: List of FiftyOne sample objects.
            transforms: Albumentations transform.
            label_map: Dictionary mapping class names to numeric labels.
        """
        self.samples = samples
        self.transforms = transforms
        self.label_map = label_map

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample.filepath
        
        # Load image using cv2 (BGR) and convert to RGB
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        
        # Retrieve detections (assumes detections are stored in the 'detections' field)
        bboxes = []
        category_ids = []
        detections = sample.get_field("detections")
        if detections is not None:
            for det in detections.detections:
                label = det.label
                if label not in self.label_map:
                    continue  # skip labels not in our mapping
                # Convert normalized bbox [x, y, w, h] to absolute Pascal VOC [x_min, y_min, x_max, y_max]
                x, y, bw, bh = det.bounding_box
                x_min = x * w
                y_min = y * h
                x_max = (x + bw) * w
                y_max = (y + bh) * h
                bboxes.append([x_min, y_min, x_max, y_max])
                category_ids.append(self.label_map[label])
        
        transformed = self.transforms(image=image, bboxes=bboxes, category_ids=category_ids)
        image = transformed["image"]
        bboxes = transformed["bboxes"]
        labels = transformed["category_ids"]
        
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
    dataset = foz.load_zoo_dataset(
        "open-images-v6",
        split="train",
        label_types="detections",
        classes=["Person", "Dog"],
        max_samples=max_samples,
        shuffle=True,
        dataset_name="open-images-person-dog"
    )
    return dataset

def create_balanced_samples(foset):
    """
    Create a balanced list of samples based on the presence of each class.
    Computes per-sample weights inversely proportional to class frequency.
    Returns a list of samples and a corresponding list of weights.
    """
    samples = list(foset.iter_samples())  # Convert samples iterator to list
    n_person = 0
    n_dog = 0
    sample_flags = []  # Each element is a tuple (has_person, has_dog)
    
    # Count how many images contain each class.
    for sample in samples:
        detections = sample.get_field("detections")
        has_person = False
        has_dog = False
        if detections is not None:
            for det in detections.detections:
                if det.label == "Person":
                    has_person = True
                elif det.label == "Dog":
                    has_dog = True
        if has_person:
            n_person += 1
        if has_dog:
            n_dog += 1
        sample_flags.append((has_person, has_dog))
    
    w_person = 1.0 / n_person if n_person > 0 else 0.0
    w_dog = 1.0 / n_dog if n_dog > 0 else 0.0

    sample_weights = []
    for has_person, has_dog in sample_flags:
        weights = []
        if has_person:
            weights.append(w_person)
        if has_dog:
            weights.append(w_dog)
        sample_weights.append(max(weights) if weights else 0.0)
    
    return samples, sample_weights

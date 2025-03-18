import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image as PILImage
import fiftyone as fo
import fiftyone.zoo as foz

from config import DATA_ROOT, DATA_SET_TO_USE, TRAIN_VAL_SPLIT

class DogDetectionDataset(Dataset):
    def __init__(self, root=DATA_ROOT, split='train', transform=None, download=False):
        """
        Dog Detection Dataset using Open Images
        Args:
            root (str): Root directory for the dataset
            split (str): 'train' or 'validation'
            transform (callable, optional): Albumentations transform to be applied on a sample
            download (bool): If True, downloads the dataset from the internet
        """
        self.transform = transform
        self.root = os.path.abspath(root)
        self.split = "validation" if split == "val" else split
        self.samples = []
        self.dogs_per_image = []
        
        self.cache_file = os.path.join(self.root, 'dog_detection_combined_cache.pt')
        os.makedirs(self.root, exist_ok=True)
        
        if self._try_load_cache():
            return
            
        self._load_or_download_dataset(download)

    def _try_load_cache(self):
        """Attempt to load dataset from cache"""
        if not os.path.exists(self.cache_file):
            return False
            
        print(f"Loading combined dataset from cache: {self.cache_file}")
        cache_data = torch.load(self.cache_file)
        all_samples = cache_data['samples']
        all_dogs_per_image = cache_data['dogs_per_image']
        
        # Apply DATA_SET_TO_USE to reduce total dataset size
        total_samples = len(all_samples)
        num_samples_to_use = int(total_samples * DATA_SET_TO_USE)
        all_samples = all_samples[:num_samples_to_use]
        all_dogs_per_image = all_dogs_per_image[:num_samples_to_use]
        
        # Split into train/val using TRAIN_VAL_SPLIT
        train_size = int(len(all_samples) * TRAIN_VAL_SPLIT)
        
        if self.split == 'train':
            self.samples = all_samples[:train_size]
            self.dogs_per_image = all_dogs_per_image[:train_size]
        else:
            self.samples = all_samples[train_size:]
            self.dogs_per_image = all_dogs_per_image[train_size:]
        
        print(f"Successfully loaded {len(self.samples)} samples for {self.split} split")
        print(f"Using {DATA_SET_TO_USE*100:.1f}% of total data with {TRAIN_VAL_SPLIT*100:.1f}% train split")
        return True

    def _load_or_download_dataset(self, download):
        """Load or download the dataset from FiftyOne"""
        original_dir = fo.config.dataset_zoo_dir
        fo.config.dataset_zoo_dir = self.root
        
        try:
            dataset_name = "open-images-v7-full"
            try:
                dataset = fo.load_dataset(dataset_name)
                print(f"Successfully loaded existing dataset: {dataset_name}")
            except fo.core.dataset.DatasetNotFoundError:
                if download:
                    print("Downloading Open Images dataset with dog class...")
                    dataset = foz.load_zoo_dataset(
                        "open-images-v7",
                        splits=["train", "validation"],
                        label_types=["detections"],
                        classes=["Dog"],
                        dataset_name=dataset_name
                    )
                    print(f"Successfully downloaded dataset to {fo.config.dataset_zoo_dir}")
                else:
                    raise RuntimeError(f"Dataset {dataset_name} not found and download=False")
            
            self._process_dataset(dataset)
            
        finally:
            fo.config.dataset_zoo_dir = original_dir
            
        if len(self.samples) == 0:
            raise RuntimeError("No valid dog images found in the dataset")

    def _process_dataset(self, dataset):
        """Process the dataset and create cache"""
        if dataset is None:
            return
            
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
                                det.bounding_box[2], det.bounding_box[3]] for det in dog_detections]
                        all_samples.append((img_path, boxes))
                        all_dogs_per_image.append(len(dog_detections))
        
        if len(all_samples) > 0:
            print(f"Saving combined dataset to cache: {self.cache_file}")
            torch.save({
                'samples': all_samples,
                'dogs_per_image': all_dogs_per_image
            }, self.cache_file)
            
            # Apply splits
            num_samples_to_use = int(len(all_samples) * DATA_SET_TO_USE)
            all_samples = all_samples[:num_samples_to_use]
            all_dogs_per_image = all_dogs_per_image[:num_samples_to_use]
            
            train_size = int(len(all_samples) * TRAIN_VAL_SPLIT)
            if self.split == 'train':
                self.samples = all_samples[:train_size]
                self.dogs_per_image = all_dogs_per_image[:train_size]
            else:
                self.samples = all_samples[train_size:]
                self.dogs_per_image = all_dogs_per_image[train_size:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, boxes = self.samples[index]
        
        try:
            img = np.array(PILImage.open(img_path).convert('RGB'))
            
            validated_boxes = []
            for box in boxes:
                x_min, y_min, x_max, y_max = box
                x_min = max(0.0, min(0.99, float(x_min)))
                y_min = max(0.0, min(0.99, float(y_min)))
                x_max = max(min(1.0, float(x_max)), x_min + 0.01)
                y_max = max(min(1.0, float(y_max)), y_min + 0.01)
                validated_boxes.append([x_min, y_min, x_max, y_max])
            
            h, w = img.shape[:2]
            boxes_abs = []
            for box in validated_boxes:
                x_min, y_min, x_max, y_max = box
                boxes_abs.append([x_min * w, y_min * h, x_max * w, y_max * h])
            labels = [1] * len(boxes_abs)
            
            if self.transform:
                transformed = self.transform(image=img, bboxes=boxes_abs, labels=labels)
                img = transformed['image']
                boxes_abs = transformed['bboxes']
                labels = transformed['labels']
                
                _, new_h, new_w = img.shape
                normalized_boxes = []
                for box in boxes_abs:
                    x_min, y_min, x_max, y_max = box
                    normalized_boxes.append([x_min / new_w, y_min / new_h, x_max / new_w, y_max / new_h])
                boxes_tensor = torch.tensor(normalized_boxes, dtype=torch.float32)
            else:
                boxes_tensor = torch.tensor(validated_boxes, dtype=torch.float32)
            
            target = {
                'boxes': boxes_tensor,
                'labels': torch.ones(len(boxes_tensor), dtype=torch.long),
                'scores': torch.ones(len(boxes_tensor))
            }
            
            return img, target
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise
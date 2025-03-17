"""
Data loading module for dog detection dataset.
Handles loading COCO data, filtering for dogs and people,
and creating appropriate train/val splits.
"""

import os
import torch
import random
import json
import pickle
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pycocotools.coco import COCO
import config


class CocoDogsDataset(Dataset):
    """
    Dataset for dog detection using COCO.
    - Loads only images containing dogs or people.
    - Applies DOG_USAGE_RATIO to determine how many dog images to use.
    - Splits data according to TRAIN_VAL_SPLIT, regardless of original split.
    """

    def __init__(self, data_root, is_train=True, transform=None):
        """
        Initialize dataset for either training or validation.
        
        Args:
            data_root (str): Path to the COCO dataset root
            is_train (bool): If True, load training set, otherwise validation set
            transform: Optional image transformations
        """
        self.data_root = data_root
        self.is_train = is_train
        self.transform = transform
        self.dog_category_id = config.COCO_DOG_CATEGORY_ID
        self.person_category_id = 1
        
        # Create cache directory
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize empty collections
        self.img_ids = []
        self.annotations = {}
        self.img_info = {}
        
        # Process dataset
        self._prepare_dataset()
        
        # Set up default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(config.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.MEAN, std=config.STD)
            ])

    def _prepare_dataset(self):
        """
        Prepare the dataset by:
        1. Loading all dog and person images (if they exist on disk)
        2. Applying DOG_USAGE_RATIO to select subset of dog images
        3. Creating train/val split using TRAIN_VAL_SPLIT
        4. Processing annotations for selected images
        """
        # Key for dataset caching
        cache_key = f"{config.DOG_USAGE_RATIO}_{config.TRAIN_VAL_SPLIT}"
        cache_file = os.path.join(self.cache_dir, f"{'train' if self.is_train else 'val'}_{cache_key}.pkl")
        
        # Try to load from cache first
        if os.path.exists(cache_file):
            print(f"Loading dataset from cache for {'training' if self.is_train else 'validation'}...")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.img_ids = cache_data['img_ids']
                self.annotations = cache_data['annotations']
                self.img_info = cache_data['img_info']
                self.stats = cache_data.get('stats', {})
                
            # Re-verify all images exist
            self._verify_images()
            return
            
        # If we need to build the dataset from scratch
        print("Building dataset from scratch...")
        
        # Load both train and val annotations 
        train_ann_file = os.path.join(self.data_root, "annotations", "instances_train2017.json")
        val_ann_file = os.path.join(self.data_root, "annotations", "instances_val2017.json")
        
        # Initialize COCO API for both splits
        train_coco = COCO(train_ann_file)
        val_coco = COCO(val_ann_file)
        
        # Get all dog and person images from both splits
        train_dog_imgs = self._get_verified_images(train_coco, "train2017", self.dog_category_id)
        train_person_imgs = self._get_verified_images(train_coco, "train2017", self.person_category_id)
        val_dog_imgs = self._get_verified_images(val_coco, "val2017", self.dog_category_id)
        val_person_imgs = self._get_verified_images(val_coco, "val2017", self.person_category_id)
        
        # Combine all images, tracking their original dataset
        all_dog_imgs = [(img_id, "train2017", train_coco) for img_id in train_dog_imgs]
        all_dog_imgs.extend([(img_id, "val2017", val_coco) for img_id in val_dog_imgs])
        
        # Remove people images that also contain dogs
        train_person_only = train_person_imgs - train_dog_imgs
        val_person_only = val_person_imgs - val_dog_imgs
        
        all_person_imgs = [(img_id, "train2017", train_coco) for img_id in train_person_only]
        all_person_imgs.extend([(img_id, "val2017", val_coco) for img_id in val_person_only])
        
        # Store total counts for statistics
        total_dog_images = len(all_dog_imgs)
        total_person_images = len(all_person_imgs)
        
        
        # Apply DOG_USAGE_RATIO to select subset of dog images
        num_dogs_to_use = int(total_dog_images * config.DOG_USAGE_RATIO)
        
        # Ensure we don't exceed available person-only images (for balance)
        num_persons_to_use = min(num_dogs_to_use, total_person_images)
        
        # If we have fewer person images than dogs, adjust dog count to match
        if num_persons_to_use < num_dogs_to_use:
            num_dogs_to_use = num_persons_to_use
                    
        # Randomly select images
        random.seed(42)  # For reproducibility
        selected_dog_imgs = random.sample(all_dog_imgs, num_dogs_to_use)
        selected_person_imgs = random.sample(all_person_imgs, num_persons_to_use)
        
        # Calculate train/val split
        train_dog_count = int(num_dogs_to_use * config.TRAIN_VAL_SPLIT)
        train_person_count = int(num_persons_to_use * config.TRAIN_VAL_SPLIT)
        
        # Split data
        if self.is_train:
            dog_subset = selected_dog_imgs[:train_dog_count]
            person_subset = selected_person_imgs[:train_person_count]
        else:
            dog_subset = selected_dog_imgs[train_dog_count:]
            person_subset = selected_person_imgs[train_person_count:]
            
        # Create the full dataset
        all_selected = dog_subset + person_subset
        random.shuffle(all_selected)
        
        # Process the selected images
        self._process_selected_images(all_selected)
        
        # Cache the processed data
        stats = {
            'with_dogs': len(dog_subset),
            'without_dogs': len(person_subset),
            'total_available_dogs': total_dog_images,
            'total_available_persons': total_person_images,
            'dog_usage_ratio': config.DOG_USAGE_RATIO,
            'train_val_split': config.TRAIN_VAL_SPLIT
        }
        
        self.stats = stats
        
        cache_data = {
            'img_ids': self.img_ids,
            'annotations': self.annotations,
            'img_info': self.img_info,
            'stats': stats
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
            
        print(f"Created {'training' if self.is_train else 'validation'} dataset with "
              f"{len(dog_subset)} dog images and {len(person_subset)} person-only images")
            
    def _get_verified_images(self, coco, split_name, category_id):
        """Get set of image IDs for a category that exist on disk with parallel downloads"""
        from dog_detector.utils import download_file_torch
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # First get all image IDs for this category
        img_ids = coco.getImgIds(catIds=[category_id])
        total_count = len(img_ids)
        verified_imgs = set()
        missing_count = 0
        
        print(f"\nProcessing {split_name} - {coco.loadCats([category_id])[0]['name']}: Found {total_count} images in annotations")
        
        # Create list of download tasks for missing images
        download_tasks = []
        for img_id in img_ids:
            img_info = coco.loadImgs(img_id)[0]
            file_name = img_info["file_name"]
            img_path = os.path.join(self.data_root, split_name, file_name)
            
            if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
                url = f"http://images.cocodataset.org/{split_name}/{file_name}"
                download_tasks.append((url, img_path, img_id))
            else:
                verified_imgs.add(img_id)
        
        if download_tasks:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(download_tasks[0][1]), exist_ok=True)
            
            # Use ThreadPoolExecutor for parallel downloads
            max_workers = min(32, len(download_tasks))  # Limit max workers
            print(f"Downloading {len(download_tasks)} missing images in parallel with {max_workers} workers...")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {
                    executor.submit(download_file_torch, url, path): (img_id, path)
                    for url, path, img_id in download_tasks
                }
                
                for future in as_completed(future_to_task):
                    img_id, path = future_to_task[future]
                    try:
                        if future.result():
                            verified_imgs.add(img_id)
                        else:
                            missing_count += 1
                            print(f"Failed to download {path}")
                    except Exception as e:
                        missing_count += 1
                        print(f"Failed to download {path}: {e}")
        
        if missing_count > 0:
            print(f"{split_name} - {coco.loadCats([category_id])[0]['name']}: {missing_count} images failed to download")
                
        return verified_imgs
    
    def _process_selected_images(self, selected_images):
        """Process annotations for selected images"""
        for img_id, split_name, coco in selected_images:
            # Get image info
            img_info = coco.loadImgs(img_id)[0]
            
            # Only get dog annotations (not person)
            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[self.dog_category_id], iscrowd=False)
            anns = coco.loadAnns(ann_ids)
            
            # Process dog annotations
            dog_boxes = []
            for ann in anns:
                if ann["category_id"] == self.dog_category_id:
                    x, y, w, h = ann["bbox"]
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(x + w, img_info["width"])
                    y2 = min(y + h, img_info["height"]) 
                    dog_boxes.append([x1, y1, x2, y2, 0])  # 0 is the class index for dog
            
            # Only keep images with valid annotations
            if split_name == "train2017" or split_name == "val2017":
                self.img_ids.append(img_id)
                self.annotations[img_id] = dog_boxes
                self.img_info[img_id] = {
                    "file_name": img_info["file_name"],
                    "height": img_info["height"],
                    "width": img_info["width"],
                    "split": split_name  # Track the original split
                }
    
    def _verify_images(self):
        """Verify all images exist and remove those that don't"""
        valid_img_ids = []
        
        print(f"Verifying {'training' if self.is_train else 'validation'} images exist...")
        
        for img_id in self.img_ids:
            img_info = self.img_info[img_id]
            split = img_info.get("split", "val2017" if "val2017" in img_info["file_name"] else "train2017")
            
            image_path = os.path.join(self.data_root, split, img_info["file_name"])
            
            if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                valid_img_ids.append(img_id)
            else:
                print(f"Removing missing image: {image_path}")
                
        # Update dataset with only valid images
        self.img_ids = valid_img_ids
        
        print(f"Found {len(valid_img_ids)} valid images for {'training' if self.is_train else 'validation'}")
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        """Get a single item from the dataset"""
        if idx >= len(self.img_ids) or idx < 0:
            idx = random.randint(0, max(0, len(self.img_ids) - 1))
            
        img_id = self.img_ids[idx]
        img_info = self.img_info[img_id]
        
        # Get the correct image path
        split = img_info.get("split", "val2017" if "val2017" in img_info["file_name"] else "train2017")
        image_path = os.path.join(self.data_root, split, img_info["file_name"])
        
        try:
            img = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, IOError) as e:
            print(f"Error loading image {image_path}, skipping...")
            
            # Return another random image
            if len(self.img_ids) > 1:
                return self.__getitem__(random.randint(0, len(self.img_ids) - 1))
            else:
                # Create a dummy image if no other images are available
                img = Image.new('RGB', (100, 100), color=(73, 109, 137))
        
        # Process annotations
        boxes = self.annotations.get(img_id, [])
        
        # Convert to tensors
        if boxes:
            boxes_tensor = torch.tensor([box[:4] for box in boxes], dtype=torch.float32)
            labels_tensor = torch.tensor([box[4] + 1 for box in boxes], dtype=torch.int64)  # +1 because 0 is background
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            
        # Apply transformations
        if self.transform:
            img = self.transform(img)
            
        # Create target dictionary
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([img_id]),
            "orig_size": (img_info["height"], img_info["width"])
        }
        
        return img, target
    
    @staticmethod
    def get_dataset_stats(data_root):
        """Get statistics about the datasets"""
        # Check for cached stats
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
        cache_key = f"{config.DOG_USAGE_RATIO}_{config.TRAIN_VAL_SPLIT}"
        train_cache = os.path.join(cache_dir, f"train_{cache_key}.pkl")
        val_cache = os.path.join(cache_dir, f"val_{cache_key}.pkl")
        
        stats = {}
        
        # Try to load from cache
        if os.path.exists(train_cache) and os.path.exists(val_cache):
            try:
                with open(train_cache, 'rb') as f:
                    train_data = pickle.load(f)
                    train_stats = train_data.get('stats', {})
                
                with open(val_cache, 'rb') as f:
                    val_data = pickle.load(f)
                    val_stats = val_data.get('stats', {})
                
                # Combine stats
                stats = {
                    'dog_usage_ratio': config.DOG_USAGE_RATIO,
                    'train_val_split': config.TRAIN_VAL_SPLIT,
                    'total_available_dogs': train_stats.get('total_available_dogs', 0),
                    'total_available_persons': train_stats.get('total_available_persons', 0),
                    'train_with_dogs': train_stats.get('with_dogs', 0),
                    'train_without_dogs': train_stats.get('without_dogs', 0),
                    'val_with_dogs': val_stats.get('with_dogs', 0),
                    'val_without_dogs': val_stats.get('without_dogs', 0)
                }
                
                # Print dataset stats
                print("\nDataset Configuration:")
                print("--------------------")
                print(f"DOG_USAGE_RATIO: {config.DOG_USAGE_RATIO * 100:.1f}%")
                print(f"TRAIN_VAL_SPLIT: {config.TRAIN_VAL_SPLIT * 100:.1f}%/{(1-config.TRAIN_VAL_SPLIT) * 100:.1f}%")
                
                print("\nTotal Available Images:")
                print("--------------------")
                print(f"Images with dogs: {stats['total_available_dogs']}")
                print(f"Images without dogs (person only): {stats['total_available_persons']}")
                
                print("\nTraining Set ({config.TRAIN_VAL_SPLIT * 100:.1f}%):")
                print("--------------------")
                print(f"Images with dogs: {stats['train_with_dogs']}")
                print(f"Images without dogs: {stats['train_without_dogs']}")
                print(f"Total training images: {stats['train_with_dogs'] + stats['train_without_dogs']}")
                
                print("\nValidation Set ({(1-config.TRAIN_VAL_SPLIT) * 100:.1f}%):")
                print("--------------------")
                print(f"Images with dogs: {stats['val_with_dogs']}")
                print(f"Images without dogs: {stats['val_without_dogs']}")
                print(f"Total validation images: {stats['val_with_dogs'] + stats['val_without_dogs']}\n")
                
                return stats
            
            except (EOFError, pickle.UnpicklingError, KeyError) as e:
                print(f"Error loading cached stats: {e}")
                print("Generating new dataset...")
        
        # If we're here, we need to compute stats from scratch by creating the datasets 
        print("Computing dataset statistics from scratch...")
        train_dataset = CocoDogsDataset(data_root, is_train=True)
        val_dataset = CocoDogsDataset(data_root, is_train=False)
        
        # Compute and return stats
        train_stats = train_dataset.stats
        val_stats = val_dataset.stats
        
        stats = {
            'dog_usage_ratio': config.DOG_USAGE_RATIO,
            'train_val_split': config.TRAIN_VAL_SPLIT,
            'total_available_dogs': train_stats.get('total_available_dogs', 0),
            'total_available_persons': train_stats.get('total_available_persons', 0),
            'train_with_dogs': train_stats.get('with_dogs', 0),
            'train_without_dogs': train_stats.get('without_dogs', 0),
            'val_with_dogs': val_stats.get('with_dogs', 0),
            'val_without_dogs': val_stats.get('without_dogs', 0)
        }
        
        # Print dataset stats
        print("\nDataset Configuration:")
        print("--------------------")
        print(f"DOG_USAGE_RATIO: {config.DOG_USAGE_RATIO * 100:.1f}%")
        print(f"TRAIN_VAL_SPLIT: {config.TRAIN_VAL_SPLIT * 100:.1f}%/{(1-config.TRAIN_VAL_SPLIT) * 100:.1f}%")
        
        print("\nTotal Available Images:")
        print("--------------------")
        print(f"Images with dogs: {stats['total_available_dogs']}")
        print(f"Images without dogs (person only): {stats['total_available_persons']}")
        
        print("\nTraining Set ({config.TRAIN_VAL_SPLIT * 100:.1f}%):")
        print("--------------------")
        print(f"Images with dogs: {stats['train_with_dogs']}")
        print(f"Images without dogs: {stats['train_without_dogs']}")
        print(f"Total training images: {stats['train_with_dogs'] + stats['train_without_dogs']}")
        
        print("\nValidation Set ({(1-config.TRAIN_VAL_SPLIT) * 100:.1f}%):")
        print("--------------------")
        print(f"Images with dogs: {stats['val_with_dogs']}")
        print(f"Images without dogs: {stats['val_without_dogs']}")
        print(f"Total validation images: {stats['val_with_dogs'] + stats['val_without_dogs']}\n")
        
        return stats


def collate_fn(batch):
    """Custom collate function for variable size data"""
    return tuple(zip(*batch))


def get_data_loaders(root=None, download=True, batch_size=None):
    """Get train and validation data loaders"""
    if root is None:
        root = config.DATA_ROOT
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    # Create datasets
    train_dataset = CocoDogsDataset(root, is_train=True)
    val_dataset = CocoDogsDataset(root, is_train=False)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn
    )

    return train_loader, val_loader

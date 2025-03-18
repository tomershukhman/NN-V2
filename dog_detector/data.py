"""
Data loading module for dog detection dataset.
Handles loading COCO data, filtering for dogs and people,
and creating appropriate train/val splits.
"""

import os
import torch
import random
import pickle
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pycocotools.coco import COCO
import config
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from dog_detector.utils import download_file_torch
from config import (
    DOG_USAGE_RATIO,
    TRAIN_VAL_SPLIT,
    COCO_DOG_CATEGORY_ID,
    IMAGE_SIZE,
    MEAN,
    STD,
    NUM_WORKERS,
    DATA_ROOT,
    BATCH_SIZE
)


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
        self.dog_category_id = COCO_DOG_CATEGORY_ID
        self.person_category_id = 1
        
        # Create cache directory
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize empty collections
        self.img_ids = []
        self.annotations = {}
        self.img_info = {}
        self.stats = {}
        
        # Process dataset
        self._prepare_dataset()
        
        # Set up default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD)
            ])

    @classmethod
    def get_dataset_stats(cls, data_root):
        """
        Get dataset statistics without creating a full dataset instance.
        Uses cached stats if available or computes them if needed.
        
        Args:
            data_root (str): Path to the COCO dataset root
            
        Returns:
            dict: Statistics about the dataset
        """
        # Create cache directory path
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
        
        # Check for all available cache files
        stats = {}
        combined_stats = {}
        
        # Try to load from both train and val cache files to get complete stats
        cache_key = f"{DOG_USAGE_RATIO}_{TRAIN_VAL_SPLIT}"
        train_cache_file = os.path.join(cache_dir, f"train_{cache_key}.pkl")
        val_cache_file = os.path.join(cache_dir, f"val_{cache_key}.pkl")
        
        if os.path.exists(train_cache_file):
            try:
                with open(train_cache_file, 'rb') as f:
                    train_cache_data = pickle.load(f)
                    train_stats = train_cache_data.get('stats', {})
                    combined_stats.update(train_stats)
                    combined_stats['train_with_dogs'] = train_stats.get('with_dogs', 0)
                    combined_stats['train_without_dogs'] = train_stats.get('without_dogs', 0)
            except (pickle.PickleError, EOFError):
                pass
                
        if os.path.exists(val_cache_file):
            try:
                with open(val_cache_file, 'rb') as f:
                    val_cache_data = pickle.load(f)
                    val_stats = val_cache_data.get('stats', {})
                    combined_stats['val_with_dogs'] = val_stats.get('with_dogs', 0)
                    combined_stats['val_without_dogs'] = val_stats.get('without_dogs', 0)
                    
                    # Use these values if not already present from train cache
                    for key in ['total_available_dogs', 'total_available_persons', 'dog_usage_ratio', 'train_val_split']:
                        if key not in combined_stats and key in val_stats:
                            combined_stats[key] = val_stats[key]
            except (pickle.PickleError, EOFError):
                pass
                
        # Calculate totals
        if 'train_with_dogs' in combined_stats and 'val_with_dogs' in combined_stats:
            combined_stats['total_with_dogs'] = combined_stats['train_with_dogs'] + combined_stats['val_with_dogs']
            
        if 'train_without_dogs' in combined_stats and 'val_without_dogs' in combined_stats:
            combined_stats['total_without_dogs'] = combined_stats['train_without_dogs'] + combined_stats['val_without_dogs']
            
        if 'total_with_dogs' in combined_stats and 'total_without_dogs' in combined_stats:
            combined_stats['total_images'] = combined_stats['total_with_dogs'] + combined_stats['total_without_dogs']
        
        return combined_stats

    def _prepare_dataset(self):
        """
        Prepare the dataset by:
        1. Loading all dog and person images (if they exist on disk)
        2. Applying DOG_USAGE_RATIO to select subset of dog images
        3. Creating train/val split using TRAIN_VAL_SPLIT
        4. Processing annotations for selected images
        """
        # Key for dataset caching
        cache_key = f"{DOG_USAGE_RATIO}_{TRAIN_VAL_SPLIT}"
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
        num_dogs_to_use = int(total_dog_images * DOG_USAGE_RATIO)
        
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
        train_dog_count = int(num_dogs_to_use * TRAIN_VAL_SPLIT)
        train_person_count = int(num_persons_to_use * TRAIN_VAL_SPLIT)
        
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
            'dog_usage_ratio': DOG_USAGE_RATIO,
            'train_val_split': TRAIN_VAL_SPLIT
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
        """Get set of image IDs for a category that exist on disk with optimized parallel downloading"""
        # First get all image IDs for this category
        img_ids = coco.getImgIds(catIds=[category_id])
        total_count = len(img_ids)
        verified_imgs = set()
        missing_count = 0
        
        print(f"\nProcessing {split_name} - {coco.loadCats([category_id])[0]['name']}: Found {total_count} images in annotations")
        
        # Batch process image verification for better performance
        batch_size = 1000
        download_tasks = []
        
        for i in range(0, len(img_ids), batch_size):
            batch_ids = img_ids[i:i + batch_size]
            batch_infos = coco.loadImgs(batch_ids)
            
            for img_info in batch_infos:
                file_name = img_info["file_name"]
                img_path = os.path.join(self.data_root, split_name, file_name)
                
                if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
                    url = f"http://images.cocodataset.org/{split_name}/{file_name}"
                    download_tasks.append((url, img_path, img_info["id"]))
                else:
                    verified_imgs.add(img_info["id"])
        
        if download_tasks:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(download_tasks[0][1]), exist_ok=True)
            
            # Adaptive system parameters
            cpu_count = psutil.cpu_count(logical=True)
            total_memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
            
            # Use adaptive parameters based on system capabilities
            max_workers = min(cpu_count * 2, int(total_memory_gb * 2), 32)
            initial_batch_size = min(int(total_memory_gb * 15), 120)
            
            print(f"Downloading {len(download_tasks)} missing images using {max_workers} workers")
            print(f"Initial batch size: {initial_batch_size}, adapts based on memory usage")
            
            batch_size = initial_batch_size
            total_batches = (len(download_tasks) + batch_size - 1) // batch_size
            
            # Process batches
            for batch_idx in range(0, len(download_tasks), batch_size):
                batch = download_tasks[batch_idx:batch_idx + batch_size]
                current_batch_num = batch_idx // batch_size + 1
                
                # Monitor and adjust memory usage
                current_memory = psutil.virtual_memory()
                if current_memory.percent > 85:
                    batch_size = max(30, int(batch_size * 0.7))
                    max_workers = max(8, int(max_workers * 0.8))
                    print(f"\nMemory usage high ({current_memory.percent}%), reducing batch size to {batch_size}")
                elif current_memory.percent < 60 and batch_size < initial_batch_size:
                    batch_size = min(initial_batch_size, batch_size + 20)
                    print(f"\nMemory usage low ({current_memory.percent}%), increasing batch size to {batch_size}")
                
                print(f"\rProcessing batch {current_batch_num}/{total_batches}", end="", flush=True)
                
                # Download batch with ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {}
                    for url, path, img_id in batch:
                        future = executor.submit(download_file_torch, url, path)
                        futures[future] = img_id
                    
                    for future in as_completed(futures):
                        img_id = futures[future]
                        try:
                            if future.result():
                                verified_imgs.add(img_id)
                            else:
                                missing_count += 1
                        except Exception:
                            missing_count += 1
                
                # Memory management between batches
                gc.collect()
                if psutil.virtual_memory().percent > 85:
                    pass
        
        print(f"\n{split_name} - {coco.loadCats([category_id])[0]['name']}: {missing_count} images failed to download")
        print(f"Total images available: {len(verified_imgs)}")
        
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
            
            # Only keep images with valid annotations or person-only images
            # We want to keep person-only images as negative samples
            if split_name == "train2017" or split_name == "val2017":
                self.img_ids.append(img_id)
                self.annotations[img_id] = dog_boxes  # This will be empty for person-only images
                self.img_info[img_id] = {
                    "file_name": img_info["file_name"],
                    "height": img_info["height"],
                    "width": img_info["width"],
                    "split": split_name,  # Track the original split
                    "has_dogs": len(dog_boxes) > 0  # Track if image contains dogs
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
            orig_width, orig_height = img.size
        except (FileNotFoundError, IOError) as e:
            print(f"Error loading image {image_path}, skipping...")
            
            # Return another random image
            if len(self.img_ids) > 1:
                return self.__getitem__(random.randint(0, len(self.img_ids) - 1))
            else:
                # Create a dummy image if no other images are available
                img = Image.new('RGB', (100, 100), color=(73, 109, 137))
                orig_width, orig_height = 100, 100
        
        # Process annotations
        boxes = self.annotations.get(img_id, [])
        
        # Apply transformations to image first
        if self.transform:
            img = self.transform(img)
        
        # Convert boxes to tensors and scale to target size
        if boxes:
            boxes_tensor = torch.tensor([box[:4] for box in boxes], dtype=torch.float32)
            labels_tensor = torch.tensor([box[4] + 1 for box in boxes], dtype=torch.int64)  # +1 because 0 is background
            
            # Scale boxes to match the target image size (512x512)
            # Note: IMAGE_SIZE is now expected to be a tuple of (width, height)
            scale_x = IMAGE_SIZE[0] / orig_width
            scale_y = IMAGE_SIZE[1] / orig_height
            
            # Scale the coordinates - keep in absolute pixel coordinates
            boxes_tensor[:, [0, 2]] *= scale_x  # Scale x coordinates
            boxes_tensor[:, [1, 3]] *= scale_y  # Scale y coordinates
            
            # Clamp to ensure boxes stay within image bounds
            boxes_tensor[:, [0, 2]] = boxes_tensor[:, [0, 2]].clamp(0, IMAGE_SIZE[0])
            boxes_tensor[:, [1, 3]] = boxes_tensor[:, [1, 3]].clamp(0, IMAGE_SIZE[1])
            
            # Ensure x2 > x1 and y2 > y1
            boxes_tensor[:, 2] = torch.max(boxes_tensor[:, 2], boxes_tensor[:, 0] + 1.0)
            boxes_tensor[:, 3] = torch.max(boxes_tensor[:, 3], boxes_tensor[:, 1] + 1.0)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            
        # Create target dictionary
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([img_id]),
            "orig_size": (orig_height, orig_width)
        }
        
        return img, target


def collate_fn(batch):
    """Custom collate function for variable size data"""
    return tuple(zip(*batch))


def get_data_loaders(root=None, download=True, batch_size=None):
    """Get train and validation data loaders"""
    if root is None:
        root = DATA_ROOT
    if batch_size is None:
        batch_size = BATCH_SIZE

    # Create datasets
    train_dataset = CocoDogsDataset(root, is_train=True)
    val_dataset = CocoDogsDataset(root, is_train=False)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    return train_loader, val_loader

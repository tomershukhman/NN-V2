import os
import torch
import random
import json
import pickle
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from dog_detector.config import config
from concurrent.futures import ThreadPoolExecutor

class CocoDogsDataset(Dataset):
    """
    A dataset for dog detection using the COCO 2017 dataset.
    Returns a balanced set of images with and without dogs.
    Returns a transformed image and a target dict with:
      - boxes (Tensor[N, 4]): in (x1, y1, x2, y2) format.
      - labels (Tensor[N]): dog labels (1)
      - orig_size (tuple): original image dimensions before transform
    """
    def __init__(self, data_root, set_name, transform=None):
        self.data_root = data_root
        self.set_name = set_name
        self.transform = transform
        self.dog_category_id = 18  # COCO category id for dog
        
        # Create local cache directory in current working directory
        cache_dir = os.path.join(".", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Cache files in local directory
        self.cache_file = os.path.join(cache_dir, f"{set_name}_processed_data.pkl")
        
        # Try to load from cache first
        if os.path.exists(self.cache_file):
            print(f"Loading processed data from cache for {set_name}")
            with open(self.cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.img_ids = cached_data['img_ids']
                self.annotations = cached_data['annotations']
                self.img_info = cached_data['img_info']
        else:
            print(f"Processing dataset for {set_name} (this may take a while...)")
            self._initialize_dataset()
        
        if set_name == config.TRAIN_SET:
            self.images_dir = os.path.join(data_root, "train2017")
        elif set_name == config.VAL_SET:
            self.images_dir = os.path.join(data_root, "val2017")
        else:
            raise ValueError("set_name must be either 'train2017' or 'val2017'")

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(config.IMAGE_SIZE, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.MEAN, std=config.STD)
            ])

    def _initialize_dataset(self):
        """Initialize dataset with parallel processing and caching"""
        ann_file = os.path.join(self.data_root, "annotations", f"instances_{self.set_name}.json")
        coco = COCO(ann_file)
        
        # Get all image IDs with dogs
        dog_img_ids = coco.getImgIds(catIds=[self.dog_category_id])
        
        # Pre-fetch all annotations and image info in parallel
        with ThreadPoolExecutor() as executor:
            # Process dog images
            dog_futures = [executor.submit(self._process_image, img_id, coco) 
                         for img_id in dog_img_ids]
            
            ann_counts = {}
            annotations = {}
            img_info = {}
            
            for future in dog_futures:
                img_id, img_anns, img_data = future.result()
                if img_anns:  # Only include images with valid annotations
                    ann_counts[img_id] = len(img_anns)
                    annotations[img_id] = img_anns
                    img_info[img_id] = img_data
        
        # Sort images by number of dogs (descending)
        sorted_dog_ids = sorted(ann_counts.keys(), 
                              key=lambda k: ann_counts[k], 
                              reverse=True)
        
        # Apply data fraction
        target_dog_images = int(len(sorted_dog_ids) * config.DATA_FRACTION)
        sampled_dog_ids = sorted_dog_ids[:target_dog_images]
        
        # Get non-dog images
        all_img_ids = coco.getImgIds()
        non_dog_img_ids = list(set(all_img_ids) - set(dog_img_ids))
        
        # Sample equal number of non-dog images
        random.seed(42)
        sampled_non_dog_ids = random.sample(non_dog_img_ids, target_dog_images)
        
        # Process non-dog images in parallel
        with ThreadPoolExecutor() as executor:
            non_dog_futures = [executor.submit(self._process_image, img_id, coco) 
                             for img_id in sampled_non_dog_ids]
            
            for future in non_dog_futures:
                img_id, _, img_data = future.result()
                img_info[img_id] = img_data
                annotations[img_id] = []  # Empty annotations for non-dog images
        
        # Combine and shuffle
        self.img_ids = sampled_dog_ids + sampled_non_dog_ids
        random.shuffle(self.img_ids)
        
        # Store processed data
        self.annotations = annotations
        self.img_info = img_info
        
        # Cache the processed data
        cache_data = {
            'img_ids': self.img_ids,
            'annotations': self.annotations,
            'img_info': self.img_info
        }
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

    def _process_image(self, img_id, coco):
        """Process a single image and its annotations"""
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id, 
                                catIds=[self.dog_category_id], 
                                iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        
        # Pre-process annotations
        processed_anns = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(x + w, img_info["width"])
            y2 = min(y + h, img_info["height"])
            processed_anns.append([x1, y1, x2, y2])
            
        return img_id, processed_anns, img_info

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.img_info[img_id]
        
        # Load and process image
        image_path = os.path.join(self.images_dir, img_info["file_name"])
        img = Image.open(image_path).convert("RGB")
        orig_width, orig_height = img.size
        orig_size = (orig_height, orig_width)
        
        # Get pre-processed annotations
        boxes = self.annotations[img_id]
        
        # Convert to tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)  # dog label = 1
        
        img = self.transform(img)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "orig_size": orig_size
        }
        
        return img, target

    @staticmethod
    def get_dataset_stats(data_root):
        """Collect statistics about the dataset"""
        # Try to load from local cache directory
        cache_dir = os.path.join(".", "cache")
        stats_cache = os.path.join(cache_dir, "dataset_stats.json")
        
        if os.path.exists(stats_cache):
            with open(stats_cache, 'r') as f:
                return json.load(f)
        
        stats = {}
        train_ann_file = os.path.join(data_root, "annotations", f"instances_{config.TRAIN_SET}.json")
        val_ann_file = os.path.join(data_root, "annotations", f"instances_{config.VAL_SET}.json")
        
        train_coco = COCO(train_ann_file)
        val_coco = COCO(val_ann_file)
        
        dog_category_id = 18
        
        train_dog_img_ids = train_coco.getImgIds(catIds=[dog_category_id])
        val_dog_img_ids = val_coco.getImgIds(catIds=[dog_category_id])
        
        train_target_count = int(len(train_dog_img_ids) * config.DATA_FRACTION)
        val_target_count = int(len(val_dog_img_ids) * config.DATA_FRACTION)
        
        stats['data_fraction'] = config.DATA_FRACTION
        stats['train_with_dogs'] = train_target_count
        stats['train_without_dogs'] = train_target_count
        stats['val_with_dogs'] = val_target_count
        stats['val_without_dogs'] = val_target_count
        stats['train_total'] = train_target_count * 2
        stats['val_total'] = val_target_count * 2
        
        # Cache the stats in local directory
        os.makedirs(cache_dir, exist_ok=True)
        with open(stats_cache, 'w') as f:
            json.dump(stats, f)
        
        return stats

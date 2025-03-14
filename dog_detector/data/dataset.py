"""
Dog detection dataset module.

This module defines the core dataset class for loading and processing dog detection data
from the COCO dataset.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiohttp
import asyncio
import time
from .cache import extract_zip

from config import DATA_ROOT

class DogDetectionDataset(Dataset):
    """
    Dataset for dog detection using COCO.
    
    Loads images and bounding box annotations for dog detection from COCO dataset.
    """
    
    COCO_CATEGORY_ID = 18  # Dog category ID in COCO dataset
    CHUNK_SIZE = 1024 * 1024  # 1MB chunks
    MAX_WORKERS = 8  # Number of parallel downloads
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    
    def __init__(self, root=DATA_ROOT, split='train', transform=None, download=True, load_all_splits=True):
        """
        Initialize the COCO dog detection dataset.
        
        Args:
            root: Root directory for the dataset
            split: 'train' or 'val'
            transform: Optional transform to be applied to images and boxes
            download: Whether to download the dataset if not found
            load_all_splits: If True, loads images from all splits (train/val)
        """
        self.root = root
        self.split = 'val' if split == 'validation' else split
        self.transform = transform
        self.load_all_splits = load_all_splits
        
        # Create COCO directories if they don't exist
        self.coco_dir = os.path.join(root, 'coco')
        os.makedirs(self.coco_dir, exist_ok=True)
        
        # Set up file paths
        year = '2017'
        self.image_dir = os.path.join(self.coco_dir, f'{split}{year}')
        
        if download:
            if load_all_splits:
                self._download_coco('train', year)
                self._download_coco('val', year)
            else:
                self._download_coco(split, year)
        
        # Initialize COCO api for each split
        self.split_data = {}
        if load_all_splits:
            splits = ['train', 'val']
        else:
            splits = [self.split]
            
        for s in splits:
            ann_file = os.path.join(self.coco_dir, f'annotations/instances_{s}{year}.json')
            coco = COCO(ann_file)
            # Get all image ids containing dogs for this split
            cat_ids = coco.getCatIds(catNms=['dog'])
            img_ids = coco.getImgIds(catIds=cat_ids)
            self.split_data[s] = {
                'coco': coco,
                'img_ids': img_ids,
                'image_dir': os.path.join(self.coco_dir, f'{s}{year}')
            }
            
        # Combine all image IDs if loading all splits
        if load_all_splits:
            self.all_img_data = []
            for s, data in self.split_data.items():
                for img_id in data['img_ids']:
                    self.all_img_data.append({
                        'split': s,
                        'img_id': img_id,
                        'coco': data['coco'],
                        'image_dir': data['image_dir']
                    })
            print(f"Found total of {len(self.all_img_data)} images containing dogs across all splits")
        else:
            print(f"Found {len(self.split_data[split]['img_ids'])} images containing dogs in {split} set")

    async def _download_file_async(self, session, url, destination):
        """Download a file asynchronously with retry logic"""
        for attempt in range(self.MAX_RETRIES):
            try:
                async with session.get(url) as response:
                    response.raise_for_status()
                    with open(destination, 'wb') as f:
                        async for chunk in response.content.iter_chunked(self.CHUNK_SIZE):
                            f.write(chunk)
                return True
            except Exception as e:
                if attempt == self.MAX_RETRIES - 1:
                    print(f"\nFailed to download after {self.MAX_RETRIES} attempts: {url}")
                    return False
                await asyncio.sleep(self.RETRY_DELAY * (attempt + 1))
        return False

    async def _download_batch_async(self, urls_and_paths):
        """Download a batch of files concurrently"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for url, path in urls_and_paths:
                task = asyncio.create_task(self._download_file_async(session, url, path))
                tasks.append(task)
            return await asyncio.gather(*tasks)

    def _download_coco(self, split, year):
        """Download only dog-related images from COCO dataset"""
        from tqdm import tqdm

        BASE_URL = 'http://images.cocodataset.org/'
        
        # First, download only annotations
        ann_url = f'{BASE_URL}annotations/annotations_trainval{year}.zip'
        ann_filename = 'annotations.zip'
        ann_zip_path = os.path.join(self.coco_dir, ann_filename)
        
        # Download and extract annotations if they don't exist
        if not os.path.exists(os.path.join(self.coco_dir, 'annotations')):
            print("\nDownloading COCO annotations...")
            asyncio.run(self._download_batch_async([(ann_url, ann_zip_path)]))
            print("Extracting annotations...")
            extract_zip(ann_zip_path, self.coco_dir)
        
        # Initialize COCO API with annotations
        ann_file = os.path.join(self.coco_dir, f'annotations/instances_{split}{year}.json')
        temp_coco = COCO(ann_file)
        
        # Get image IDs containing dogs
        print("\nIdentifying images containing dogs...")
        cat_ids = temp_coco.getCatIds(catNms=['dog'])
        dog_img_ids = temp_coco.getImgIds(catIds=cat_ids)
        total_dog_images = len(dog_img_ids)
        print(f"Found {total_dog_images} images containing dogs in {split} set")
        
        # Create directory for images if it doesn't exist
        os.makedirs(os.path.join(self.coco_dir, f'{split}{year}'), exist_ok=True)
        
        # Count how many images we need to download
        existing_images = 0
        to_download = []
        print("\nChecking existing files...")
        for img_id in dog_img_ids:
            img_info = temp_coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.coco_dir, f'{split}{year}', img_info['file_name'])
            if os.path.exists(img_path):
                existing_images += 1
            else:
                img_url = f'{BASE_URL}{split}{year}/{img_info["file_name"]}'
                to_download.append((img_url, img_path))
        
        print(f"\nFound {existing_images} already downloaded images")
        print(f"Need to download {len(to_download)} new images")
        
        # Download missing dog-related images in parallel batches
        if to_download:
            batch_size = 50  # Number of images to download in each batch
            total_batches = (len(to_download) + batch_size - 1) // batch_size
            
            with tqdm(total=len(to_download), desc="Downloading dog images") as pbar:
                for i in range(0, len(to_download), batch_size):
                    batch = to_download[i:i + batch_size]
                    results = asyncio.run(self._download_batch_async(batch))
                    successful = sum(1 for r in results if r)
                    pbar.update(successful)
        
        print("\nDownload completed!")
        print(f"Total dog images available: {total_dog_images}")
        print(f"- Previously downloaded: {existing_images}")
        print(f"- Newly downloaded: {len(to_download)}")

    def __getitem__(self, index):
        """Get a single image and its annotations with proper normalization"""
        if self.load_all_splits:
            img_data = self.all_img_data[index]
            img_id = img_data['img_id']
            coco = img_data['coco']
            image_dir = img_data['image_dir']
        else:
            img_id = self.split_data[self.split]['img_ids'][index]
            coco = self.split_data[self.split]['coco']
            image_dir = self.split_data[self.split]['image_dir']
        
        # Load image
        img_info = coco.loadImgs(img_id)[0]
        image = Image.open(os.path.join(image_dir, img_info['file_name'])).convert('RGB')
        width, height = image.size
        
        # Load annotations
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[self.COCO_CATEGORY_ID])
        anns = coco.loadAnns(ann_ids)
        
        # Convert COCO boxes to normalized coordinates
        boxes = []
        valid_boxes = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            # Normalize coordinates
            x1 = x / width
            y1 = y / height
            x2 = (x + w) / width
            y2 = (y + h) / height
            
            # Validate box coordinates
            if (x2 > x1 and y2 > y1 and 
                x1 >= 0 and y1 >= 0 and 
                x2 <= 1 and y2 <= 1):
                boxes.append([x1, y1, x2, y2])
                valid_boxes.append(True)
            else:
                valid_boxes.append(False)
        
        if not boxes:
            # Return a valid empty target if no valid boxes
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.long)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            # Only create labels for valid boxes
            labels = torch.ones(len(boxes), dtype=torch.long)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'valid_boxes': torch.tensor(valid_boxes) if valid_boxes else torch.zeros(0, dtype=torch.bool)
        }
        
        # Apply transforms if any
        if self.transform is not None:
            image = np.array(image)
            transformed = self.transform(
                image=image, 
                bboxes=boxes.numpy(),
                labels=labels.numpy()
            )
            image = transformed['image']
            target['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            target['labels'] = torch.tensor(transformed['labels'], dtype=torch.long)
        
        return image, target

    def __len__(self):
        """Return the number of images in the dataset"""
        if self.load_all_splits:
            return len(self.all_img_data)
        return len(self.split_data[self.split]['img_ids'])
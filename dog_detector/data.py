from concurrent.futures import ThreadPoolExecutor
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
    A dataset for dog detection using the COCO 2017 dataset.
    Downloads only images containing dogs or people to save space.
    """

    def __init__(self, data_root, set_name, transform=None):
        self.data_root = data_root
        self.set_name = set_name
        self.transform = transform
        self.categories_to_download = {
            'dog': config.DOG_CATEGORY_ID,
            'person': 1  # Used only for download filtering
        }

        # Create cache directory in the project root
        cache_dir = os.path.join(os.path.dirname(
            os.path.dirname(__file__)), "cache")
        os.makedirs(cache_dir, exist_ok=True)

        # Cache files paths
        self.cache_file = os.path.join(
            cache_dir, f"{set_name}_processed_data.pkl")
        self.stats_cache = os.path.join(cache_dir, "dataset_stats.json")

        # Load or initialize dataset
        if os.path.exists(self.cache_file):
            self._load_from_cache()
            # Check if DATA_SET_TO_USE has changed
            if self.stats.get('data_set_to_use') != config.DOG_USAGE_RATIO:
                print(
                    f"DATA_SET_TO_USE changed from {self.stats.get('data_set_to_use')} to {config.DOG_USAGE_RATIO}, reinitializing dataset...")
                self._initialize_dataset()
        else:
            self._initialize_dataset()

        # Set up image directory path
        self.images_dir = os.path.join(data_root, set_name)

        # Set up default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(config.IMAGE_SIZE, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.MEAN, std=config.STD)
            ])

    def _load_from_cache(self):
        """Load dataset information from cache"""
        print(f"Loading processed data from cache for {self.set_name}")
        with open(self.cache_file, 'rb') as f:
            cached_data = pickle.load(f)
            self.img_ids = cached_data['img_ids']
            self.annotations = cached_data['annotations']
            self.img_info = cached_data['img_info']
            self.stats = cached_data.get('stats', {})

    def _initialize_dataset(self):
        """Initialize dataset with parallel processing and caching"""
        # Set fixed seed for reproducibility and consistency between train/val splits
        random.seed(42)

        ann_file = os.path.join(
            self.data_root, "annotations", f"instances_{self.set_name}.json")
        coco = COCO(ann_file)

        # Get image IDs containing either dogs or persons
        dog_category_id = self.categories_to_download['dog']
        person_category_id = self.categories_to_download['person']

        # Get total available images
        all_dog_img_ids = list(coco.getImgIds(catIds=[dog_category_id]))
        person_img_ids = set(coco.getImgIds(catIds=[person_category_id]))
        person_only_img_ids = list(person_img_ids - set(all_dog_img_ids))

        # Store total number of available dog images
        total_available_dogs = len(all_dog_img_ids)
        selected_dog_imgs = list(random.sample(list(all_dog_img_ids), total_dog_images_to_use))

        # Calculate exact numbers for train/val split
        train_dogs_target = int(total_dog_images_to_use * config.TRAIN_VAL_SPLIT)
        val_dogs_target = total_dog_images_to_use - train_dogs_target  # Use remainder to avoid rounding errors

        # Split dogs into train and val sets
        if self.set_name == config.TRAIN_SET:
            dog_img_ids = set(selected_dog_imgs[:train_dogs_target])
        else:  # Validation set
            dog_img_ids = set(selected_dog_imgs[train_dogs_target:])

        # Get balanced number of person-only images for this split
        target_person_images = len(dog_img_ids)  # Same number as dogs for balance
        person_only_img_ids = set(random.sample(list(person_only_img_ids), target_person_images))

        print(f"Selected {len(dog_img_ids)} images with dogs and {len(person_only_img_ids)} images without dogs for {self.set_name}")

        # Combine and shuffle
        target_img_ids = list(dog_img_ids | person_only_img_ids)
        random.shuffle(target_img_ids)

        # Set up image directory path
        if self.set_name == config.TRAIN_SET:
            images_dir = os.path.join(self.data_root, "train2017")
        else:
            images_dir = os.path.join(self.data_root, "val2017")

        # Pre-fetch all annotations and image info in parallel
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._process_image, img_id, coco)
                       for img_id in target_img_ids]

            ann_counts = {}
            annotations = {}
            img_info = {}

            for future in futures:
                img_id, img_anns, img_data = future.result()
                # Include all images - those with dog annotations will have them,
                # those without will have empty annotations
                image_path = os.path.join(images_dir, img_data["file_name"])
                if os.path.exists(image_path):
                    ann_counts[img_id] = len(img_anns)
                    annotations[img_id] = img_anns
                    img_info[img_id] = img_data
                else:
                    pass
                    # print(f"Warning: Image {img_data['file_name']} not found, skipping...")

        if not annotations:
            raise RuntimeError(
                f"No valid images found in {images_dir}. Please ensure the dataset is downloaded correctly.")

        # Store all image IDs - they're already balanced and shuffled
        self.img_ids = list(annotations.keys())

        # Store processed data
        self.annotations = annotations
        self.img_info = img_info

        # Cache the processed data
        cache_data = {
            'img_ids': self.img_ids,
            'annotations': self.annotations,
            'img_info': self.img_info,
            'stats': {
                'with_dogs': len(dog_img_ids),
                'without_dogs': len(person_only_img_ids),
                'total_available_dogs': total_available_dogs,
                'dog_usage_ratio': config.DOG_USAGE_RATIO,
                'data_set_to_use': config.DOG_USAGE_RATIO
            }
        }
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

    def _process_image(self, img_id, coco):
        """Process a single image and its annotations - only process dog annotations"""
        img_info = coco.loadImgs(img_id)[0]
        # Get all dog annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id,
                                 # Only get dog annotations
                                 catIds=[config.DOG_CATEGORY_ID],
                                 iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        # Pre-process annotations - only for dogs
        processed_anns = []
        for ann in anns:
            if ann["category_id"] == config.DOG_CATEGORY_ID:  # Double check it's a dog
                x, y, w, h = ann["bbox"]
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(x + w, img_info["width"])
                y2 = min(y + h, img_info["height"])
                # Label is always 1 for dog
                processed_anns.append([x1, y1, x2, y2, 1])

        return img_id, processed_anns, img_info

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.img_info[img_id]

        # Load and process image
        image_path = os.path.join(self.images_dir, img_info["file_name"])
        try:
            img = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, IOError):
            print(f"Error loading image {image_path}, removing from cache...")
            # Remove this image from the dataset
            self.img_ids.pop(idx)
            # Force regeneration of cache to maintain balance
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
            # Try the next image
            return self.__getitem__(idx % len(self.img_ids))

            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)  # Force regeneration of cache
            # Try the next image instead
            return self.__getitem__(idx % len(self.img_ids))

        orig_width, orig_height = img.size
        orig_size = (orig_height, orig_width)

        # Get pre-processed annotations
        annotations = self.annotations[img_id]

        if annotations:
            # Split boxes and labels
            boxes = torch.tensor([ann[:4]
                                 for ann in annotations], dtype=torch.float32)
            labels = torch.tensor([ann[4]
                                  for ann in annotations], dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

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

        # Load cached stats from train and val datasets directly
        train_cache = os.path.join(cache_dir, "train2017_processed_data.pkl")
        val_cache = os.path.join(cache_dir, "val2017_processed_data.pkl")

        if os.path.exists(train_cache) and os.path.exists(val_cache):
            with open(train_cache, 'rb') as f:
                train_data = pickle.load(f)
            with open(val_cache, 'rb') as f:
                val_data = pickle.load(f)

            stats = {
                'total_available_dogs': train_data['stats'].get('total_available_dogs', 0),
                'dog_usage_ratio': config.DOG_USAGE_RATIO,
                'train_with_dogs': train_data['stats']['with_dogs'],
                'train_without_dogs': train_data['stats']['without_dogs'],
                'val_with_dogs': val_data['stats']['with_dogs'],
                'val_without_dogs': val_data['stats']['without_dogs']
            }

            # Cache the stats
            os.makedirs(cache_dir, exist_ok=True)
            with open(stats_cache, 'w') as f:
                json.dump(stats, f)

            return stats

        # If no cache exists, compute full dataset stats
        stats = {}
        train_ann_file = os.path.join(
            data_root, "annotations", f"instances_{config.TRAIN_SET}.json")
        val_ann_file = os.path.join(
            data_root, "annotations", f"instances_{config.VAL_SET}.json")

        train_coco = COCO(train_ann_file)
        val_coco = COCO(val_ann_file)

        # Get image IDs for dogs and people separately
        dog_category_id = 18
        person_category_id = 1

        # Get train set statistics
        train_dog_imgs = set(train_coco.getImgIds(catIds=[dog_category_id]))
        train_person_imgs = set(
            train_coco.getImgIds(catIds=[person_category_id]))
        train_person_only = train_person_imgs - train_dog_imgs

        # Get validation set statistics
        val_dog_imgs = set(val_coco.getImgIds(catIds=[dog_category_id]))
        val_person_imgs = set(val_coco.getImgIds(catIds=[person_category_id]))
        val_person_only = val_person_imgs - val_dog_imgs

        # Calculate balanced counts
        train_target = int(min(len(train_dog_imgs), len(
            train_person_only)) * config.DATA_FRACTION)
        val_target = int(min(len(val_dog_imgs), len(
            val_person_only)) * config.DATA_FRACTION)

        # Store balanced stats
        stats['data_fraction'] = config.DATA_FRACTION
        stats['train_with_dogs'] = train_target
        stats['train_without_dogs'] = train_target
        stats['val_with_dogs'] = val_target
        stats['val_without_dogs'] = val_target

        # Cache the stats
        os.makedirs(cache_dir, exist_ok=True)
        with open(stats_cache, 'w') as f:
            json.dump(stats, f)

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

    # Create training and validation datasets separately
    train_dataset = CocoDogsDataset(root, config.TRAIN_SET)
    val_dataset = CocoDogsDataset(root, config.VAL_SET)

    # Create data loaders directly from the balanced datasets

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

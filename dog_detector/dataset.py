#dog_detector/dataset.py
import os
import torch
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from dog_detector.config import config

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

        ann_file = os.path.join(data_root, "annotations", f"instances_{set_name}.json")
        self.coco = COCO(ann_file)

        if set_name == config.TRAIN_SET:
            self.images_dir = os.path.join(data_root, "train2017")
        elif set_name == config.VAL_SET:
            self.images_dir = os.path.join(data_root, "val2017")
        else:
            raise ValueError("set_name must be either 'train2017' or 'val2017'")

        # COCO category id for dog is 18
        self.dog_category_id = 18

        # Get all image IDs with dogs
        self.dog_img_ids = self.coco.getImgIds(catIds=[self.dog_category_id])
        
        # Get annotation counts for images with dogs
        ann_counts = {}
        for img_id in self.dog_img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=[self.dog_category_id], iscrowd=False)
            ann_counts[img_id] = len(ann_ids)
        
        # Get all image IDs without dogs
        all_img_ids = self.coco.getImgIds()
        self.non_dog_img_ids = list(set(all_img_ids) - set(self.dog_img_ids))
        
        # First apply data fraction to dog images, ensuring we keep images with multiple dogs
        num_dog_images = len(self.dog_img_ids)
        target_dog_images = int(num_dog_images * config.DATA_FRACTION)
        
        # Sort images by number of dogs (descending) to prioritize multi-dog images
        sorted_dog_ids = sorted(ann_counts.keys(), key=lambda k: ann_counts[k], reverse=True)
        sampled_dog_ids = sorted_dog_ids[:target_dog_images]
        
        # Sample equal number of non-dog images
        random.seed(42)  # For reproducibility
        sampled_non_dog_ids = random.sample(self.non_dog_img_ids, target_dog_images)
        
        # Combine the sampled IDs
        self.img_ids = sampled_dog_ids + sampled_non_dog_ids
        
        # Shuffle the final list of image IDs
        random.shuffle(self.img_ids)

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(config.IMAGE_SIZE, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.MEAN, std=config.STD)
            ])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        image_path = os.path.join(self.images_dir, img_info["file_name"])
        img = Image.open(image_path).convert("RGB")
        orig_width, orig_height = img.size
        
        # Store original dimensions for proper box scaling during training
        orig_size = (orig_height, orig_width)

        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=[self.dog_category_id], iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(x + w, orig_width)
            y2 = min(y + h, orig_height)
            boxes.append([x1, y1, x2, y2])
        
        # Convert to tensor, if no boxes were found (non-dog image), use empty tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)  # dog label = 1

        img = self.transform(img)
        
        # Include original image size in the target dict for correct box scaling
        target = {
            "boxes": boxes, 
            "labels": labels, 
            "image_id": torch.tensor([img_id]),
            "orig_size": orig_size
        }
        
        return img, target

    @staticmethod
    def get_dataset_stats(data_root):
        """
        Collect statistics about the dataset
        """
        stats = {}
        
        # Create COCO API objects for both train and val sets
        train_ann_file = os.path.join(data_root, "annotations", f"instances_{config.TRAIN_SET}.json")
        val_ann_file = os.path.join(data_root, "annotations", f"instances_{config.VAL_SET}.json")
        
        train_coco = COCO(train_ann_file)
        val_coco = COCO(val_ann_file)
        
        # COCO category id for dog is 18
        dog_category_id = 18
        
        # Get image counts
        train_dog_img_ids = train_coco.getImgIds(catIds=[dog_category_id])
        val_dog_img_ids = val_coco.getImgIds(catIds=[dog_category_id])
        
        # Calculate dataset sizes based on fraction of dog images
        train_target_count = int(len(train_dog_img_ids) * config.DATA_FRACTION)
        val_target_count = int(len(val_dog_img_ids) * config.DATA_FRACTION)
        
        # Compile stats
        stats['data_fraction'] = config.DATA_FRACTION
        stats['train_with_dogs'] = train_target_count
        stats['train_without_dogs'] = train_target_count
        stats['val_with_dogs'] = val_target_count
        stats['val_without_dogs'] = val_target_count
        stats['train_total'] = train_target_count * 2
        stats['val_total'] = val_target_count * 2
        
        return stats

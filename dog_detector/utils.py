#dog_detector/utils.py
import os
import zipfile
import torch
from pycocotools.coco import COCO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image
import config
import random
import urllib3
import time
from urllib3.util import Retry
import socket

# Create a global connection pool for reuse - aggressive but stable settings
http = urllib3.PoolManager(
    maxsize=100,  # Increased to 100 concurrent connections
    retries=Retry(
        total=2,  # Minimal retries
        backoff_factor=0.1,
        status_forcelist=[500, 502, 503, 504]
    ),
    timeout=urllib3.Timeout(connect=5, read=20)
)

def compute_iou(boxes1, boxes2):
    """
    Compute IoU between boxes1 (N,4) and boxes2 (M,4).
    Returns Tensor of shape (N, M).
    """
    N = boxes1.size(0)
    M = boxes2.size(0)
    if N == 0 or M == 0:
        return torch.zeros((N, M))
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])).unsqueeze(1)
    area2 = ((boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])).unsqueeze(0)
    union = area1 + area2 - inter
    iou = inter / union
    return iou


def assign_anchors_to_image(anchors, gt_boxes, pos_iou_thresh=0.5, neg_iou_thresh=0.4):
    """
    Assign each anchor a classification target and a regression target.
    Args:
        anchors: (N, 4) tensor of anchor boxes
        gt_boxes: (M, 4) tensor of ground truth boxes
        pos_iou_thresh: IoU threshold for positive samples
        neg_iou_thresh: IoU threshold for negative samples
    Returns:
        cls_targets: (N,) tensor with values {1: positive, 0: negative, -1: ignore}
        reg_targets: (N, 4) tensor of regression targets
        pos_mask: (N,) boolean tensor indicating positive anchors
    """
    num_anchors = anchors.size(0)
    device = anchors.device
    
    # Initialize targets
    cls_targets = -torch.ones((num_anchors,), dtype=torch.int64, device=device)
    reg_targets = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)

    if gt_boxes.numel() == 0:
        # No ground truth boxes, all anchors are negative
        cls_targets[:] = 0
        pos_mask = torch.zeros_like(cls_targets, dtype=torch.bool)
        return cls_targets, reg_targets, pos_mask

    # Compute IoU between anchors and GT boxes
    ious = compute_iou(anchors, gt_boxes)  # Shape: (num_anchors, num_gt)
    
    # For each anchor, find the best matching GT box
    max_iou, max_idx = ious.max(dim=1)
    
    # Assign positive/negative labels based on IoU thresholds
    pos_mask = max_iou >= pos_iou_thresh
    neg_mask = max_iou < neg_iou_thresh
    cls_targets[pos_mask] = 1
    cls_targets[neg_mask] = 0
    
    # For each GT box, ensure the best matching anchor is positive
    if gt_boxes.size(0) > 0:  # Only if there are GT boxes
        gt_best_anchors = ious.max(dim=0)[1]  # Best anchor for each GT
        cls_targets[gt_best_anchors] = 1
        pos_mask[gt_best_anchors] = True
    
    # Compute regression targets for positive anchors
    if pos_mask.any():
        matched_gt_boxes = gt_boxes[max_idx[pos_mask]]
        pos_anchors = anchors[pos_mask]
        
        # Convert both boxes to center format
        pos_anchor_w = pos_anchors[:, 2] - pos_anchors[:, 0]
        pos_anchor_h = pos_anchors[:, 3] - pos_anchors[:, 1]
        pos_anchor_cx = pos_anchors[:, 0] + 0.5 * pos_anchor_w
        pos_anchor_cy = pos_anchors[:, 1] + 0.5 * pos_anchor_h
        
        gt_w = matched_gt_boxes[:, 2] - matched_gt_boxes[:, 0]
        gt_h = matched_gt_boxes[:, 3] - matched_gt_boxes[:, 1]
        gt_cx = matched_gt_boxes[:, 0] + 0.5 * gt_w
        gt_cy = matched_gt_boxes[:, 1] + 0.5 * gt_h
        
        # Compute regression targets
        reg_targets[pos_mask, 0] = (gt_cx - pos_anchor_cx) / pos_anchor_w
        reg_targets[pos_mask, 1] = (gt_cy - pos_anchor_cy) / pos_anchor_h
        reg_targets[pos_mask, 2] = torch.log(gt_w / pos_anchor_w)
        reg_targets[pos_mask, 3] = torch.log(gt_h / pos_anchor_h)

    return cls_targets, reg_targets, pos_mask


def download_file_torch(url, dest_path, max_retries=2):
    """Optimized download with minimal overhead"""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return True
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            response = http.request('GET', url, preload_content=False)
            
            if response.status != 200:
                raise Exception(f"HTTP error {response.status}")
                
            with open(dest_path, 'wb') as out_file:
                while True:
                    data = response.read(65536)  # Increased to 64KB chunks
                    if not data:
                        break
                    out_file.write(data)
            
            response.release_conn()
            return True
            
        except Exception as e:
            if attempt == max_retries - 1:
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                return False
            
            time.sleep(0.1)  # Minimal delay between retries
            continue
    return False

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

def download_single_image(args):
    """Download a single image with verification and proper connection handling"""
    img_info, target_dir, is_train = args
    file_name = img_info['file_name']
    target_path = os.path.join(target_dir, file_name)
    
    subset = "train2017" if is_train else "val2017"
    url = f"http://images.cocodataset.org/{subset}/{file_name}"
    
    if download_file_torch(url, target_path):
        return f"Downloaded {file_name}"
    else:
        return f"Failed to download {file_name}"

def download_coco_dataset(data_root):
    """Download COCO dataset with maximum stable performance"""
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    ann_dir = os.path.join(data_root, "annotations")
    
    # First download and extract annotations
    if not os.path.exists(ann_dir):
        ann_zip = os.path.join(data_root, "annotations_trainval2017.zip")
        if not download_file_torch(ann_url, ann_zip):
            raise RuntimeError("Failed to download annotations")
        extract_zip(ann_zip, data_root)
        os.remove(ann_zip)
    else:
        print("Annotations directory exists; skipping annotations download.")

    # Initialize COCO API
    train_ann_file = os.path.join(ann_dir, "instances_train2017.json")
    val_ann_file = os.path.join(ann_dir, "instances_val2017.json")
    train_coco = COCO(train_ann_file)
    val_coco = COCO(val_ann_file)

    # Setup directories
    train_dir = os.path.join(data_root, "train2017")
    val_dir = os.path.join(data_root, "val2017")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    def get_balanced_image_ids(coco, category_ids):
        dog_imgs = set(coco.getImgIds(catIds=[18]))
        person_imgs = set(coco.getImgIds(catIds=[1]))
        person_only_imgs = person_imgs - dog_imgs
        
        target_size = int(min(len(dog_imgs), len(person_only_imgs)) * config.DATA_FRACTION)
        dog_imgs = set(random.sample(list(dog_imgs), target_size))
        person_only_imgs = set(random.sample(list(person_only_imgs), target_size))
        
        return list(dog_imgs | person_only_imgs)

    # Get image IDs and prepare download tasks
    train_img_ids = get_balanced_image_ids(train_coco, [18, 1])
    val_img_ids = get_balanced_image_ids(val_coco, [18, 1])
    
    train_tasks = [(train_coco.loadImgs(img_id)[0], train_dir, True) for img_id in train_img_ids]
    val_tasks = [(val_coco.loadImgs(img_id)[0], val_dir, False) for img_id in val_img_ids]
    all_tasks = train_tasks + val_tasks

    # Maximum stable performance settings for 8GB RAM
    import psutil
    total_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB
    cpu_count = psutil.cpu_count(logical=True)
    
    # Aggressive but stable worker count based on available memory
    max_workers = min(cpu_count * 4, int(total_memory * 2))  # 2 workers per GB of RAM
    batch_size = min(80, int(total_memory * 10))  # 10 downloads per GB of RAM
    
    print(f"\nStarting optimized download with {max_workers} workers")
    print(f"Downloading {len(all_tasks)} images in batches of {batch_size}")
    
    completed = 0
    failed = 0
    total = len(all_tasks)
    
    for i in range(0, len(all_tasks), batch_size):
        batch = all_tasks[i:i + batch_size]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(download_single_image, task) for task in batch]
            
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                if "Failed" in result:
                    failed += 1
                
                if completed % 20 == 0:
                    success_rate = ((completed - failed) / completed * 100) if completed > 0 else 0
                    print(f"\rProgress: {completed}/{total} ({success_rate:.1f}% success rate)", end="", flush=True)
        
        # Minimal cleanup
        gc.collect()
        time.sleep(0.2)  # Minimal delay between batches
    
    print("\nDownload phase completed!")
    
    # Quick verification only for completely missing files
    missing_train = quick_verify_downloads(train_img_ids, train_coco, train_dir)
    missing_val = quick_verify_downloads(val_img_ids, val_coco, val_dir)
    
    if missing_train or missing_val:
        print(f"\nRetrying {len(missing_train) + len(missing_val)} missing files...")
        retry_downloads(missing_train, "train2017", train_dir)
        retry_downloads(missing_val, "val2017", val_dir)

def quick_verify_downloads(img_ids, coco, target_dir):
    """Quick verification that just checks file existence and size"""
    missing = []
    for img_id in img_ids:
        file_name = coco.loadImgs(img_id)[0]['file_name']
        path = os.path.join(target_dir, file_name)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            missing.append(file_name)
    return missing

def retry_downloads(missing_files, subset, target_dir):
    """Minimal retry logic for failed downloads"""
    if not missing_files:
        return
        
    print(f"\nRetrying {len(missing_files)} files from {subset}...")
    for file_name in missing_files:
        url = f"http://images.cocodataset.org/{subset}/{file_name}"
        target_path = os.path.join(target_dir, file_name)
        download_file_torch(url, target_path)

def verify_dataset_integrity(data_root):
    """Verify dataset exists and contains necessary files"""
    train_dir = os.path.join(data_root, "train2017")
    val_dir = os.path.join(data_root, "val2017")
    ann_dir = os.path.join(data_root, "annotations")
    
    if not all(os.path.exists(d) for d in [train_dir, val_dir, ann_dir]):
        return False
        
    required_annotations = [
        "instances_train2017.json",
        "instances_val2017.json"
    ]
    
    if not all(os.path.exists(os.path.join(ann_dir, f)) for f in required_annotations):
        return False
        
    min_images = 100
    train_images = [f for f in os.listdir(train_dir) if f.endswith('.jpg')]
    val_images = [f for f in os.listdir(val_dir) if f.endswith('.jpg')]
    
    if len(train_images) < min_images or len(val_images) < min_images:
        return False
        
    return True

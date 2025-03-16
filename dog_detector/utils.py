#dog_detector/utils.py
import os
import zipfile
import torch
from pycocotools.coco import COCO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

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
    Regression targets are computed using the parameterization:
        t_x = (gt_cx - a_cx) / a_w
        t_y = (gt_cy - a_cy) / a_h
        t_w = log(gt_w / a_w)
        t_h = log(gt_h / a_h)
    Returns:
      - cls_targets: Tensor of shape (num_anchors,) with values {1: dog, 0: background, -1: ignore}
      - reg_targets: Tensor of shape (num_anchors, 4)
      - pos_mask: Boolean mask for positive anchors.
    """
    num_anchors = anchors.size(0)
    cls_targets = -torch.ones((num_anchors,), dtype=torch.int64, device=anchors.device)
    reg_targets = torch.zeros((num_anchors, 4), dtype=torch.float32, device=anchors.device)

    if gt_boxes.numel() == 0:
        cls_targets[:] = 0
        pos_mask = cls_targets == 1
        return cls_targets, reg_targets, pos_mask

    # Compute IoU between each anchor and ground truth box
    ious = compute_iou(anchors, gt_boxes)
    max_iou, max_idx = ious.max(dim=1)

    # Assign positive and negative labels based on IoU thresholds
    pos_inds = max_iou >= pos_iou_thresh
    cls_targets[pos_inds] = 1
    neg_inds = max_iou < neg_iou_thresh
    cls_targets[neg_inds] = 0

    # For positive anchors, compute the regression targets using the standard parameterization
    if pos_inds.sum() > 0:
        assigned_gt_boxes = gt_boxes[max_idx[pos_inds]]
        pos_anchors = anchors[pos_inds]

        # Compute anchor widths, heights, and centers
        anchor_x1 = pos_anchors[:, 0]
        anchor_y1 = pos_anchors[:, 1]
        anchor_x2 = pos_anchors[:, 2]
        anchor_y2 = pos_anchors[:, 3]
        anchor_w = anchor_x2 - anchor_x1
        anchor_h = anchor_y2 - anchor_y1
        anchor_cx = anchor_x1 + 0.5 * anchor_w
        anchor_cy = anchor_y1 + 0.5 * anchor_h

        # Compute ground truth widths, heights, and centers
        gt_x1 = assigned_gt_boxes[:, 0]
        gt_y1 = assigned_gt_boxes[:, 1]
        gt_x2 = assigned_gt_boxes[:, 2]
        gt_y2 = assigned_gt_boxes[:, 3]
        gt_w = gt_x2 - gt_x1
        gt_h = gt_y2 - gt_y1
        gt_cx = gt_x1 + 0.5 * gt_w
        gt_cy = gt_y1 + 0.5 * gt_h

        # Compute the regression targets
        t_x = (gt_cx - anchor_cx) / anchor_w
        t_y = (gt_cy - anchor_cy) / anchor_h
        t_w = torch.log(gt_w / anchor_w)
        t_h = torch.log(gt_h / anchor_h)

        targets = torch.stack([t_x, t_y, t_w, t_h], dim=1)
        reg_targets[pos_inds] = targets

    pos_mask = cls_targets == 1
    return cls_targets, reg_targets, pos_mask


def download_file_torch(url, dest_path, max_retries=3):
    if os.path.exists(dest_path):
        if os.path.getsize(dest_path) > 0:  # Check if file is not empty
            return
        else:
            os.remove(dest_path)  # Remove empty file to retry download
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            torch.hub.download_url_to_file(url, dest_path)
            # Verify the file was downloaded successfully
            if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
                return
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to download {url} after {max_retries} attempts: {e}")
                raise
            print(f"Attempt {attempt + 1} failed, retrying...")

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

def download_coco_dataset(data_root):
    """Download only COCO annotations and images containing dogs or people in parallel"""
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    ann_dir = os.path.join(data_root, "annotations")
    
    # First download and extract annotations as we need them to know which images to download
    if not os.path.exists(ann_dir):
        ann_zip = os.path.join(data_root, "annotations_trainval2017.zip")
        download_file_torch(ann_url, ann_zip)
        extract_zip(ann_zip, data_root)
        os.remove(ann_zip)
    else:
        print("Annotations directory exists; skipping annotations download.")

    # Initialize COCO API for both train and val sets
    train_ann_file = os.path.join(ann_dir, "instances_train2017.json")
    val_ann_file = os.path.join(ann_dir, "instances_val2017.json")
    train_coco = COCO(train_ann_file)
    val_coco = COCO(val_ann_file)

    # Target categories
    categories = {'dog': 18, 'person': 1}
    category_ids = list(categories.values())

    # Create image directories
    train_dir = os.path.join(data_root, "train2017")
    val_dir = os.path.join(data_root, "val2017")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    def download_single_image(args):
        img_info, target_dir, is_train = args
        file_name = img_info['file_name']
        target_path = os.path.join(target_dir, file_name)
        
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            return f"Skipped {file_name} (already exists)"
            
        subset = "train2017" if is_train else "val2017"
        url = f"http://images.cocodataset.org/{subset}/{file_name}"
        try:
            download_file_torch(url, target_path)
            # Verify downloaded file
            if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
                return f"Downloaded {file_name}"
            else:
                return f"Failed to download {file_name}: File is empty"
        except Exception as e:
            if os.path.exists(target_path):
                os.remove(target_path)  # Remove potentially corrupt file
            return f"Failed to download {file_name}: {e}"

    # Prepare download tasks for both train and val sets
    train_img_ids = train_coco.getImgIds(catIds=category_ids)
    val_img_ids = val_coco.getImgIds(catIds=category_ids)
    
    print(f"Found {len(train_img_ids)} training images and {len(val_img_ids)} validation images")
    
    train_tasks = [(train_coco.loadImgs(img_id)[0], train_dir, True) for img_id in train_img_ids]
    val_tasks = [(val_coco.loadImgs(img_id)[0], val_dir, False) for img_id in val_img_ids]
    all_tasks = train_tasks + val_tasks
    
    # Use ThreadPoolExecutor for parallel downloads with progress bar
    max_workers = min(32, len(all_tasks))  # Limit max concurrent downloads
    print(f"Starting parallel download with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_single_image, task) for task in all_tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
            result = future.result()
            if "Failed" in result:
                print(f"\n{result}")  # Print failures on new line

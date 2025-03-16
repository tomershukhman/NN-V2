#dog_detector/utils.py
import os
import zipfile
import torch

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


def download_file_torch(url, dest_path):
    if os.path.exists(dest_path):
        print(f"{dest_path} exists. Skipping download.")
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Downloading {url} to {dest_path}...")
    torch.hub.download_url_to_file(url, dest_path)
    print("Download complete.")

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

def download_coco_dataset(data_root):
    train_url = "http://images.cocodataset.org/zips/train2017.zip"
    val_url = "http://images.cocodataset.org/zips/val2017.zip"
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    train_dir = os.path.join(data_root, "train2017")
    val_dir = os.path.join(data_root, "val2017")
    ann_dir = os.path.join(data_root, "annotations")

    if not os.path.exists(train_dir):
        train_zip = os.path.join(data_root, "train2017.zip")
        download_file_torch(train_url, train_zip)
        extract_zip(train_zip, data_root)
        os.remove(train_zip)
    else:
        print("Train directory exists; skipping train download.")

    if not os.path.exists(val_dir):
        val_zip = os.path.join(data_root, "val2017.zip")
        download_file_torch(val_url, val_zip)
        extract_zip(val_zip, data_root)
        os.remove(val_zip)
    else:
        print("Validation directory exists; skipping val download.")

    if not os.path.exists(ann_dir):
        ann_zip = os.path.join(data_root, "annotations_trainval2017.zip")
        download_file_torch(ann_url, ann_zip)
        extract_zip(ann_zip, data_root)
        os.remove(ann_zip)
    else:
        print("Annotations directory exists; skipping annotations download.")

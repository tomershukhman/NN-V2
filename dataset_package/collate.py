"""
Collate functions for batching dataset items.
"""
import torch
import random
import logging

logger = logging.getLogger('dog_detector')

def collate_fn(batch):
    """
    Custom collate function to handle variable number of bounding boxes per image.
    Ensures boxes are properly formatted for both model training and visualization.
    Makes sure all images are consistently sized to 224x224.
    
    Args:
        batch: A list of tuples (image, target)
        
    Returns:
        tuple: (images, num_objects, all_bboxes)
    """
    images = []
    num_objects = []
    all_bboxes = []
    valid_batch = []
    
    # Standard image size required for the model
    target_size = (224, 224)
    
    for i, (img, target) in enumerate(batch):
        # Skip any invalid samples (e.g., images that failed to load)
        if img.shape[0] != 3 or img.isnan().any() or len(target['boxes']) == 0:
            continue
            
        # Always ensure every image is 224x224, regardless of original size
        if img.shape[1] != target_size[0] or img.shape[2] != target_size[1]:
            # Resize images that don't match the target size
            import torch.nn.functional as F
            img = F.interpolate(img.unsqueeze(0), size=target_size, 
                               mode='bilinear', align_corners=False).squeeze(0)
        
        valid_batch.append((img, target))
        images.append(img)
        boxes = target['boxes']
        num_objects.append(len(boxes))
        
        # Check if boxes are normalized and swap coordinates if needed
        if len(boxes) > 0:
            is_normalized = True
            for i in range(len(boxes)):
                for j in range(4):
                    if boxes[i, j] < 0.0 or boxes[i, j] > 1.0:
                        is_normalized = False
                        break
                if not is_normalized:
                    break
            if not is_normalized:
                logger.warning(f"Found unnormalized boxes in collate_fn: {boxes}")
                boxes = torch.clamp(boxes, min=0.0, max=1.0)
            
            # Fix box coordinates if needed
            for i in range(len(boxes)):
                if boxes[i, 0] > boxes[i, 2]:
                    boxes[i, 0], boxes[i, 2] = boxes[i, 2], boxes[i, 0]
                if boxes[i, 1] > boxes[i, 3]:
                    boxes[i, 1], boxes[i, 3] = boxes[i, 3], boxes[i, 1]
                    
                # Ensure minimum box size to prevent degenerate boxes
                if boxes[i, 2] - boxes[i, 0] < 0.01:
                    boxes[i, 2] = min(1.0, boxes[i, 0] + 0.01)
                if boxes[i, 3] - boxes[i, 1] < 0.01:
                    boxes[i, 3] = min(1.0, boxes[i, 1] + 0.01)
        
        all_bboxes.append(boxes)
    
    # If we have no valid samples, create a dummy batch
    if len(valid_batch) == 0:
        logger.warning("Empty batch after filtering invalid samples")
        dummy_img = torch.zeros((3, target_size[0], target_size[1]), dtype=torch.float32)
        dummy_boxes = torch.zeros((0, 4), dtype=torch.float32)
        return torch.stack([dummy_img]), torch.tensor([0]), [dummy_boxes]
    
    # Stack the tensors - we don't need to check sizes anymore since we've already
    # resized everything to the target size
    if len(images) > 0:
        images = torch.stack(images)
        num_objects = torch.tensor(num_objects)
        
        # Periodically log batch statistics (at debug level only and much less frequently)
        if logger.isEnabledFor(logging.DEBUG) and random.random() < 0.01:  # Log roughly 1% of batches
            means = images.view(images.size(0), images.size(1), -1).mean(dim=2).mean(dim=0)
            stds = images.view(images.size(0), images.size(1), -1).std(dim=2).mean(dim=0)
            logger.debug(f"Batch stats: {len(images)} images, objects per image: {num_objects.tolist()}")
            logger.debug(f"Image channel means: {means.tolist()}, stds: {stds.tolist()}")
            
            # Log multi-object percentage
            multi_object_count = sum(1 for n in num_objects if n > 1)
            if len(num_objects) > 0:
                logger.debug(f"Multi-object percentage in batch: {multi_object_count / len(num_objects):.1%}")
        
        return images, num_objects, all_bboxes
    
    # Fallback for empty batch (should not happen often)
    dummy_img = torch.zeros((3, target_size[0], target_size[1]), dtype=torch.float32)
    dummy_boxes = torch.zeros((0, 4), dtype=torch.float32)
    return torch.stack([dummy_img]), torch.tensor([0]), [dummy_boxes]
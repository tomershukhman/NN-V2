import torch
from tqdm import tqdm
import os
import time
from datetime import datetime
from config import (
    DEVICE, LEARNING_RATE, NUM_EPOCHS, OUTPUT_ROOT,
    WEIGHT_DECAY, DATA_ROOT, DOG_USAGE_RATIO, TRAIN_VAL_SPLIT,
    CONFIDENCE_THRESHOLD, IOU_THRESHOLD
)
from dog_detector.data import get_data_loaders, CocoDogsDataset
from dog_detector.model.model import get_model
from dog_detector.model.losses import DetectionLoss
from dog_detector.visualization.tensorboard_logger import VisualizationLogger
from dog_detector.utils import compute_iou

def train(data_root=None, download=True, batch_size=None):
    """Train the dog detection model"""
    start_time = time.time()
    
    # Create output directories
    os.makedirs(os.path.join(OUTPUT_ROOT, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, 'tensorboard'), exist_ok=True)

    # Initialize loggers
    vis_logger = VisualizationLogger(os.path.join(OUTPUT_ROOT, 'tensorboard'))

    # Get data loaders
    train_loader, val_loader = get_data_loaders(
        root=data_root,
        download=download,
        batch_size=batch_size
    )
    
    if data_root is None:
        data_root = DATA_ROOT
        
    # Get and display dataset statistics
    stats = CocoDogsDataset.get_dataset_stats(data_root)
    
    if stats:
        print("\n" + "="*80)
        print("📊 DATASET STATISTICS 📊")
        print("="*80)
        print(f"🐕 Dog images:")
        print(f"  - Training:   {stats.get('train_with_dogs', 0)} images")
        print(f"  - Validation: {stats.get('val_with_dogs', 0)} images")
        print(f"  - Total:      {stats.get('total_with_dogs', 0)} images")
        print(f"\n👤 Person-only images (no dogs):")
        print(f"  - Training:   {stats.get('train_without_dogs', 0)} images")
        print(f"  - Validation: {stats.get('val_without_dogs', 0)} images") 
        print(f"  - Total:      {stats.get('total_without_dogs', 0)} images")
        print(f"\n📈 Dataset configuration:")
        print(f"  - Total available dog images:     {stats.get('total_available_dogs', 0)}")
        print(f"  - Total available person images:  {stats.get('total_available_persons', 0)}")
        print(f"  - Dog usage ratio:                {stats.get('dog_usage_ratio', DOG_USAGE_RATIO)}")
        print(f"  - Train/val split:                {stats.get('train_val_split', TRAIN_VAL_SPLIT)}")
        print(f"  - Total dataset size:             {stats.get('total_images', 0)} images")
        print("="*80 + "\n")
        
        # Also log to tensorboard
        vis_logger.log_metrics({
            'dataset/total_available_dogs': stats.get('total_available_dogs', 0),
            'dataset/train_with_dogs': stats.get('train_with_dogs', 0),
            'dataset/train_without_dogs': stats.get('train_without_dogs', 0),
            'dataset/val_with_dogs': stats.get('val_with_dogs', 0),
            'dataset/val_without_dogs': stats.get('val_without_dogs', 0),
            'dataset/dog_usage_ratio': stats.get('dog_usage_ratio', DOG_USAGE_RATIO),
            'dataset/train_val_split': stats.get('train_val_split', TRAIN_VAL_SPLIT),
            'dataset/total_with_dogs': stats.get('total_with_dogs', 0),
            'dataset/total_without_dogs': stats.get('total_without_dogs', 0),
            'dataset/total_images': stats.get('total_images', 0)
        }, 0, 'stats')

    # Get model and criterion
    model = get_model(DEVICE)
    criterion = DetectionLoss().to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Training loop
    best_val_loss = float('inf')
    best_f1_score = 0.0
    
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, epoch)
        
        # Validate with detailed metrics
        val_metrics = validate_epoch(model, val_loader, criterion, epoch)
        
        # Calculate epoch duration
        epoch_duration = time.time() - epoch_start_time
        
        # Display metrics summary
        vis_logger.display_metrics_summary(train_metrics, val_metrics, epoch, epoch_duration)
        
        # Save best model (by validation loss)
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['total_loss'],
            }, os.path.join(OUTPUT_ROOT, 'checkpoints', 'best_model_loss.pth'))
            print(f'✨ New best model saved (val_loss: {val_metrics["total_loss"]:.4f})')
        
        # Also save best model by F1 score
        if val_metrics.get('f1_score', 0) > best_f1_score:
            best_f1_score = val_metrics.get('f1_score', 0)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1_score': best_f1_score,
                'precision': val_metrics.get('precision', 0),
                'recall': val_metrics.get('recall', 0),
            }, os.path.join(OUTPUT_ROOT, 'checkpoints', 'best_model_f1.pth'))
            print(f'✨ New best model saved (F1 score: {best_f1_score:.4f})')

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['total_loss'],
                'f1_score': val_metrics.get('f1_score', 0),
            }, os.path.join(OUTPUT_ROOT, 'checkpoints', f'checkpoint_epoch_{epoch+1}.pth'))
            print(f'📦 Checkpoint saved for epoch {epoch+1}')

    # Log training completion statistics
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "="*80)
    print(f"🎉 TRAINING COMPLETED!")
    print("="*80)
    print(f"⏱️  Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")
    print(f"🎯 Best F1 score: {best_f1_score:.4f}")
    print(f"💾 Models saved to: {os.path.join(OUTPUT_ROOT, 'checkpoints')}")
    print(f"📈 Logs saved to: {os.path.join(OUTPUT_ROOT, 'tensorboard')}")
    print("="*80)
    
    vis_logger.close()


def train_epoch(model, train_loader, criterion, optimizer, epoch):
    """Train model for one epoch and return metrics"""
    model.train()
    train_loss = 0
    cls_loss_total = 0
    reg_loss_total = 0
    train_steps = 0

    train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]')
    for images, targets in train_bar:
        # Prepare batch
        images = torch.stack([img.to(DEVICE) for img in images])
        targets = [{k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                    for k, v in t.items()} for t in targets]

        # Forward pass
        predictions = model(images)
        loss = criterion(predictions, targets)
        
        # Extract individual loss components if available
        if isinstance(loss, dict):
            cls_loss = loss.get('cls_loss', 0)
            reg_loss = loss.get('reg_loss', 0)
            total_loss = loss.get('total_loss', loss)  # Fallback to combined loss
        else:
            # Approximate individual losses if not provided
            total_loss = loss
            cls_loss = 0
            reg_loss = 0

        # Backward pass
        optimizer.zero_grad()
        if isinstance(total_loss, torch.Tensor):
            total_loss.backward()
        else:
            loss.backward()  # Fallback to original loss object
        optimizer.step()

        # Update metrics
        if isinstance(total_loss, torch.Tensor):
            train_loss += total_loss.item()
        else:
            train_loss += loss.item()
            
        cls_loss_total += cls_loss if isinstance(cls_loss, (int, float)) else (cls_loss.item() if isinstance(cls_loss, torch.Tensor) else 0)
        reg_loss_total += reg_loss if isinstance(reg_loss, (int, float)) else (reg_loss.item() if isinstance(reg_loss, torch.Tensor) else 0)
        train_steps += 1
        
        # Update progress bar
        train_bar.set_postfix({
            'loss': train_loss / train_steps,
            'cls_loss': cls_loss_total / train_steps,
            'reg_loss': reg_loss_total / train_steps
        })

    # Calculate average metrics
    metrics = {
        'total_loss': train_loss / (train_steps if train_steps > 0 else 1),
        'cls_loss': cls_loss_total / (train_steps if train_steps > 0 else 1),
        'reg_loss': reg_loss_total / (train_steps if train_steps > 0 else 1)
    }
    
    return metrics


def validate_epoch(model, val_loader, criterion, epoch):
    """Validate the model and compute performance metrics"""
    model.eval()
    val_loss = 0
    cls_loss_total = 0
    reg_loss_total = 0
    val_steps = 0

    # Detection metrics
    all_pred_boxes = []
    all_pred_scores = []
    all_gt_boxes = []
    true_positives = 0
    false_positives = 0
    total_gt_boxes = 0
    iou_scores = []
    
    val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Val]')
    
    with torch.no_grad():
        for images, targets in val_bar:
            # Prepare batch
            images = torch.stack([img.to(DEVICE) for img in images])
            targets = [{k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                       for k, v in t.items()} for t in targets]

            # Forward pass
            predictions = model(images)
            loss = criterion(predictions, targets)
            
            # Extract individual loss components if available
            if isinstance(loss, dict):
                cls_loss = loss.get('cls_loss', 0)
                reg_loss = loss.get('reg_loss', 0)
                total_loss = loss.get('total_loss', loss)  # Fallback to combined loss
            else:
                # Approximate individual losses if not provided
                total_loss = loss
                cls_loss = 0
                reg_loss = 0
            
            # Update loss metrics
            if isinstance(total_loss, torch.Tensor):
                val_loss += total_loss.item()
            else:
                val_loss += loss.item()
                
            cls_loss_total += cls_loss if isinstance(cls_loss, (int, float)) else (cls_loss.item() if isinstance(cls_loss, torch.Tensor) else 0)
            reg_loss_total += reg_loss if isinstance(reg_loss, (int, float)) else (reg_loss.item() if isinstance(reg_loss, torch.Tensor) else 0)
            val_steps += 1
            
            # Generate feature map sizes for anchor generation - FIXED ERROR HERE
            if isinstance(predictions, tuple) and len(predictions) >= 2:
                cls_output, reg_output = predictions[:2]
                
                # Safely get feature map dimensions
                if len(cls_output.shape) == 5:  # [B, num_classes, num_anchors, H, W]
                    _, _, _, feat_h, feat_w = cls_output.shape
                elif len(cls_output.shape) == 4:  # [B, C, H, W]
                    _, _, feat_h, feat_w = cls_output.shape
                else:
                    # Fallback: can't determine feature map size directly
                    # Instead, run backbone to get feature map size
                    with torch.no_grad():
                        features = model.backbone(images)
                        _, _, feat_h, feat_w = features.shape
            else:
                # If predictions format is unexpected, we can't proceed
                print(f"Warning: Unexpected predictions format: {type(predictions)}")
                continue  # Skip this batch
            
            # Generate anchors
            try:
                anchors = model.generate_anchors((feat_h, feat_w), DEVICE)
            
                # Post-process outputs to get bounding boxes and scores
                boxes, scores = model.post_process(cls_output, reg_output, anchors)
                
                # Calculate detection metrics
                for i, (pred_boxes, pred_scores, target) in enumerate(zip(boxes, scores, targets)):
                    gt_boxes = target["boxes"]
                    total_gt_boxes += len(gt_boxes)
                    
                    # Store for calculating avg IOU
                    if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                        iou_matrix = compute_iou(pred_boxes, gt_boxes)
                        max_iou_per_pred, _ = iou_matrix.max(dim=1)
                        
                        # Add IoU scores to list
                        for iou in max_iou_per_pred:
                            iou_scores.append(iou.item())
                            
                        # Count true and false positives
                        for j, score in enumerate(pred_scores):
                            if score > CONFIDENCE_THRESHOLD:  # Changed from CONF_THRESHOLD to CONFIDENCE_THRESHOLD
                                if max_iou_per_pred[j] > IOU_THRESHOLD:  # IoU threshold
                                    true_positives += 1
                                else:
                                    false_positives += 1
                    elif len(pred_boxes) > 0:
                        # All predictions are false positives if there are no gt boxes
                        false_positives += len(pred_scores[pred_scores > CONFIDENCE_THRESHOLD])  # Changed from CONF_THRESHOLD to CONFIDENCE_THRESHOLD
            except Exception as e:
                print(f"Warning: Error during post-processing: {str(e)}")
                continue  # Skip this batch on error

    # Calculate average metrics
    avg_val_loss = val_loss / (val_steps if val_steps > 0 else 1)
    avg_cls_loss = cls_loss_total / (val_steps if val_steps > 0 else 1)
    avg_reg_loss = reg_loss_total / (val_steps if val_steps > 0 else 1)
    
    # Compute average number of predictions per image
    pred_counts = [len(boxes) for boxes in all_pred_boxes]
    avg_pred_count = sum(pred_counts) / len(pred_counts) if pred_counts else 0
    
    # Calculate confidence scores
    try:
        if all_pred_scores and any(len(scores) > 0 for scores in all_pred_scores):
            all_scores = torch.cat([scores for scores in all_pred_scores if len(scores) > 0], dim=0).cpu().numpy()
            avg_confidence = all_scores.mean() if len(all_scores) > 0 else 0
        else:
            avg_confidence = 0
    except Exception:
        avg_confidence = 0
    
    # Calculate mean IoU
    mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0
    
    # Compute precision, recall, and F1 score
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0
        
    if total_gt_boxes > 0:
        recall = true_positives / total_gt_boxes
    else:
        recall = 0
        
    if precision + recall > 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0
    
    # Return all metrics
    metrics = {
        'total_loss': avg_val_loss,
        'cls_loss': avg_cls_loss,
        'reg_loss': avg_reg_loss,
        'mean_pred_count': avg_pred_count,
        'mean_confidence': avg_confidence,
        'mean_iou': mean_iou,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'total_gt_boxes': total_gt_boxes
    }
    
    return metrics

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
        print(
            f"  - Total available dog images:     {stats.get('total_available_dogs', 0)}")
        print(
            f"  - Total available person images:  {stats.get('total_available_persons', 0)}")
        print(
            f"  - Dog usage ratio:                {stats.get('dog_usage_ratio', DOG_USAGE_RATIO)}")
        print(
            f"  - Train/val split:                {stats.get('train_val_split', TRAIN_VAL_SPLIT)}")
        print(
            f"  - Total dataset size:             {stats.get('total_images', 0)} images")
        print("="*80 + "\n")

        # Also log to tensorboard and CSV
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
    # Pass model instance to DetectionLoss
    criterion = DetectionLoss(model).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Training loop
    best_val_loss = float('inf')
    best_f1_score = 0.0

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, epoch)

        # Validate with detailed metrics
        val_metrics = validate_epoch(
            model, val_loader, criterion, epoch, vis_logger, log_images=True)

        # Calculate epoch duration
        epoch_duration = time.time() - epoch_start_time

        # Display metrics summary
        vis_logger.display_metrics_summary(
            train_metrics, val_metrics, epoch, epoch_duration)

        # Log metrics to CSV with epoch number
        train_metrics['epoch'] = epoch
        val_metrics['epoch'] = epoch
        # Will log to both tensorboard and CSV
        vis_logger.log_metrics(train_metrics, epoch, 'train')
        # Will log to both tensorboard and CSV
        vis_logger.log_metrics(val_metrics, epoch, 'val')

        # Save best model (by validation loss)
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['total_loss'],
            }, os.path.join(OUTPUT_ROOT, 'checkpoints', 'best_model_loss.pth'))
            print(
                f'✨ New best model saved (val_loss: {val_metrics["total_loss"]:.4f})')

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
    print(
        f"⏱️  Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")
    print(f"🎯 Best F1 score: {best_f1_score:.4f}")
    print(f"💾 Models saved to: {os.path.join(OUTPUT_ROOT, 'checkpoints')}")
    print(f"📈 Logs saved to: {os.path.join(OUTPUT_ROOT, 'tensorboard')}")
    print("="*80)

    # Log final metrics
    vis_logger.log_metrics({
        'training/best_val_loss': best_val_loss,
        'training/best_f1_score': best_f1_score,
        'training/total_time_hours': hours + minutes/60 + seconds/3600
    }, NUM_EPOCHS, 'final')

    vis_logger.close()


def train_epoch(model, train_loader, criterion, optimizer, epoch):
    """Train model for one epoch and return metrics"""
    model.train()
    train_loss = 0
    cls_loss_total = 0
    reg_loss_total = 0
    train_steps = 0

    train_bar = tqdm(
        train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]')
    for images, targets in train_bar:
        # Prepare batch
        images = torch.stack([img.to(DEVICE) for img in images])
        targets = [{k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                    for k, v in t.items()} for t in targets]

        # Forward pass - now returns anchors too
        cls_output, reg_output, anchors = model(images)
        # Package outputs for criterion
        predictions = (cls_output, reg_output, anchors)
        loss_dict = criterion(predictions, targets)

        # Extract individual loss components
        cls_loss = loss_dict['cls_loss']
        reg_loss = loss_dict['reg_loss']
        total_loss = loss_dict['total_loss']

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Update metrics
        train_loss += total_loss.item()
        cls_loss_total += cls_loss.item()
        reg_loss_total += reg_loss.item()
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


def validate_epoch(model, val_loader, criterion, epoch, vis_logger, log_images=False):
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

            # Forward pass - now returns anchors also
            cls_output, reg_output, anchors = model(images)
            predictions = (cls_output, reg_output, anchors)
            loss_dict = criterion(predictions, targets)

            # Update loss metrics - handle dictionary return type
            val_loss += loss_dict['total_loss'].item()
            cls_loss_total += loss_dict['cls_loss'].item()
            reg_loss_total += loss_dict['reg_loss'].item()
            val_steps += 1

            # Post-process outputs to get bounding boxes and scores using the same anchors
            boxes, scores = model.post_process(cls_output, reg_output, anchors)

            # Log validation images with predictions if requested
            if log_images:
                vis_logger.log_images(
                    images, boxes, scores, epoch, prefix='val')

            # Calculate detection metrics
            for i, (pred_boxes, pred_scores, target) in enumerate(zip(boxes, scores, targets)):
                gt_boxes = target["boxes"]
                # This will be 0 for negative samples
                total_gt_boxes += len(gt_boxes)

                # For positive samples, compute IoU and count TP/FP
                if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                    iou_matrix = compute_iou(pred_boxes, gt_boxes)
                    max_iou_per_pred, _ = iou_matrix.max(dim=1)

                    # Add IoU scores to list
                    for iou in max_iou_per_pred:
                        iou_scores.append(iou.item())

                    # Count true and false positives
                    for j, score in enumerate(pred_scores):
                        if score > CONFIDENCE_THRESHOLD:
                            if max_iou_per_pred[j] > IOU_THRESHOLD:
                                true_positives += 1
                            else:
                                false_positives += 1
                else:
                    # For negative samples or no predictions, any prediction above threshold is a false positive
                    false_positives += len(
                        pred_scores[pred_scores > CONFIDENCE_THRESHOLD])

                # Store predictions for later analysis
                all_pred_boxes.extend(pred_boxes)
                all_pred_scores.extend(pred_scores)
                if len(gt_boxes) > 0:
                    all_gt_boxes.extend(gt_boxes)

    # Calculate average metrics
    avg_val_loss = val_loss / val_steps
    avg_cls_loss = cls_loss_total / val_steps
    avg_reg_loss = reg_loss_total / val_steps

    # Calculate mean IoU only if we have any valid IoU scores
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

    # Calculate average predictions per image
    pred_counts = [len(boxes) for boxes in all_pred_boxes]
    avg_pred_count = sum(pred_counts) / len(pred_counts) if pred_counts else 0

    # Calculate mean confidence score
    if all_pred_scores:
        mean_confidence = sum(s.item()
                              for s in all_pred_scores) / len(all_pred_scores)
    else:
        mean_confidence = 0

    # Return all metrics
    metrics = {
        'total_loss': avg_val_loss,
        'cls_loss': avg_cls_loss,
        'reg_loss': avg_reg_loss,
        'mean_pred_count': avg_pred_count,
        'mean_confidence': mean_confidence,
        'mean_iou': mean_iou,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'total_gt_boxes': total_gt_boxes
    }

    return metrics

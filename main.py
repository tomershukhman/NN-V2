import argparse
import csv
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dog_detector.config import config
from dog_detector.dataset import CocoDogsDataset
from dog_detector.model import DogDetector
from dog_detector.train import train_one_epoch, evaluate
from dog_detector.utils import download_coco_dataset

# Define a top-level collate function so that images are stacked into a batch tensor.
def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, targets

def log_metrics_to_csv(csv_path, epoch, train_metrics, eval_metrics):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow([
                "epoch", 
                "stage",
                "loss",
                "cls_loss",
                "reg_loss",
                "mean_pred_count", 
                "mean_confidence", 
                "mean_iou",
                "true_positives", 
                "false_positives", 
                "total_gt_boxes",
                "precision", 
                "recall", 
                "f1_score"
            ])
        
        # Log training metrics
        writer.writerow([
            epoch,
            "train",
            train_metrics["loss"],
            train_metrics["cls_loss"],
            train_metrics["reg_loss"],
            "N/A",  # mean_pred_count not available in training
            "N/A",  # mean_confidence not available in training
            "N/A",  # mean_iou not available in training
            "N/A",  # true_positives not available in training
            "N/A",  # false_positives not available in training
            "N/A",  # total_gt_boxes not available in training
            "N/A",  # precision not available in training
            "N/A",  # recall not available in training
            "N/A"   # f1_score not available in training
        ])
        
        # Log validation metrics
        writer.writerow([
            epoch,
            "val",
            eval_metrics["loss"],
            eval_metrics["cls_loss"],
            eval_metrics["reg_loss"],
            eval_metrics["mean_pred_count"],
            eval_metrics["mean_confidence"],
            eval_metrics["mean_iou"],
            eval_metrics["true_positives"],
            eval_metrics["false_positives"],
            eval_metrics["total_gt_boxes"],
            eval_metrics["precision"],
            eval_metrics["recall"],
            eval_metrics["f1_score"]
        ])

def display_dataset_stats(data_root):
    """Display statistics about the dataset"""
    stats = CocoDogsDataset.get_dataset_stats(data_root)
    
    print("\n" + "="*60)
    print("DATASET STATISTICS:")
    print(f"Data fraction being used: {stats['data_fraction']:.2f}")
    print("\nTRAINING SET:")
    print(f"- Images with dogs: {stats['train_with_dogs']}")
    print(f"- Images without dogs: {stats['train_without_dogs']}")
    print(f"- Total images: {stats['train_with_dogs'] + stats['train_without_dogs']}")
    
    print("\nVALIDATION SET:")
    print(f"- Images with dogs: {stats['val_with_dogs']}")
    print(f"- Images without dogs: {stats['val_without_dogs']}")
    print(f"- Total images: {stats['val_with_dogs'] + stats['val_without_dogs']}")
    print("="*60 + "\n")
    
    return stats

def main():
    download_coco_dataset(config.DATA_ROOT)
    
    # Display dataset statistics
    display_dataset_stats(config.DATA_ROOT)
    
    writer = SummaryWriter()
    
    # Check for available devices in order of preference: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    train_dataset = CocoDogsDataset(data_root=config.DATA_ROOT, set_name=config.TRAIN_SET)
    val_dataset = CocoDogsDataset(data_root=config.DATA_ROOT, set_name=config.VAL_SET)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS, collate_fn=collate_fn)
    
    model = DogDetector(num_classes=config.NUM_CLASSES, pretrained=config.PRETRAINED)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Initialize best model tracking
    best_iou = 0.0
    best_epoch = 0
    
    csv_log_path = "metrics_log.csv"
    for epoch in range(1, config.NUM_EPOCHS + 1):
        start_time = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch, writer)
        print(f"Epoch {epoch} training loss: {train_metrics['loss']:.4f} in {time.time()-start_time:.1f}s")
        
        # Run evaluation with image logging only when we might have a new best model
        eval_metrics = evaluate(model, val_loader, device, epoch, writer, log_images=True)
        writer.add_scalar("Epoch/TrainLoss", train_metrics["loss"], epoch)
        log_metrics_to_csv(csv_log_path, epoch, train_metrics, eval_metrics)
        
        # Track best model
        current_iou = eval_metrics["mean_iou"]
        if current_iou > best_iou:
            best_iou = current_iou
            best_epoch = epoch
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"New best model saved at epoch {epoch} with mean IoU: {best_iou:.4f}")
    
    print(f"\nTraining completed. Best model was from epoch {best_epoch} with mean IoU: {best_iou:.4f}")
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dog Detector Training using COCO2017")
    main()

import torch
from tqdm import tqdm
import os
from config import (
    DEVICE, LEARNING_RATE, NUM_EPOCHS,
    OUTPUT_ROOT
)
from dataset import get_data_loaders
from model import get_model
from losses import DetectionLoss
from visualization import VisualizationLogger

def train():
    # Create save directories
    checkpoints_dir = os.path.join(OUTPUT_ROOT, 'checkpoints')
    tensorboard_dir = os.path.join(OUTPUT_ROOT, 'tensorboard')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # Initialize visualization logger
    vis_logger = VisualizationLogger(tensorboard_dir)
    
    # Get model and criterion
    model = get_model(DEVICE)
    criterion = DetectionLoss().to(DEVICE)
    
    # Setup optimizer with weight decay
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders()
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        
        # Train
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc='Training')
        
        for batch_idx, (images, _, boxes) in enumerate(progress_bar):
            images = torch.stack([image.to(DEVICE) for image in images])
            targets = []
            for i, boxes_per_image in enumerate(boxes):
                target = {
                    'boxes': boxes_per_image.to(DEVICE),
                    'labels': torch.ones((len(boxes_per_image),), dtype=torch.int64, device=DEVICE)
                }
                targets.append(target)
            
            predictions = model(images, targets)
            loss_dict = criterion(predictions, targets)
            losses = loss_dict['total_loss']
            
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            
            # Log metrics
            vis_logger.log_train_metrics(loss_dict, epoch, len(train_loader) + batch_idx)
            
            # Log images with predictions periodically
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    model.eval()
                    inference_preds = model(images, None)
                    model.train()
                    vis_logger.log_images('Train', images, inference_preds, targets, epoch, len(train_loader) + batch_idx)
            
            total_loss += losses.item()
            progress_bar.set_postfix({
                'loss': losses.item(),
                'conf_loss': loss_dict['conf_loss'],
                'bbox_loss': loss_dict['bbox_loss']
            })
        
        train_loss = total_loss / len(train_loader)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        model.eval()
        val_loss = 0
        progress_bar = tqdm(val_loader, desc='Validating')
        
        with torch.no_grad():
            for batch_idx, (images, _, boxes) in enumerate(progress_bar):
                images = torch.stack([image.to(DEVICE) for image in images])
                targets = []
                for i, boxes_per_image in enumerate(boxes):
                    target = {
                        'boxes': boxes_per_image.to(DEVICE),
                        'labels': torch.ones((len(boxes_per_image),), dtype=torch.int64, device=DEVICE)
                    }
                    targets.append(target)
                
                # Get both training mode predictions for loss calculation
                model.train()
                train_predictions = model(images, targets)
                loss_dict = criterion(train_predictions, targets)
                losses = loss_dict['total_loss']
                
                # Switch back to eval mode for visualization predictions
                model.eval()
                inference_preds = model(images, None)
                
                # Log validation images periodically
                if batch_idx % 50 == 0:
                    vis_logger.log_images('Val', images, inference_preds, targets, epoch, len(val_loader) + batch_idx)
                
                val_loss += losses.item()
                progress_bar.set_postfix({'val_loss': losses.item()})
        
        val_loss = val_loss / len(val_loader)
        print(f"Val Loss: {val_loss:.4f}")
        
        # Log epoch metrics
        vis_logger.log_epoch_metrics(
            train_loss, 
            val_loss, 
            optimizer.param_groups[0]['lr'], 
            epoch
        )
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Save best model and implement early stopping
        if val_loss < best_val_loss:
            epochs_without_improvement = 0
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(checkpoints_dir, 'best_model.pth'))
            print("Saved new best model")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 10:
                print("Early stopping triggered")
                break
    
    vis_logger.close()

if __name__ == "__main__":
    train()
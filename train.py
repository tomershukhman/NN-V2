import os
import torch
from config import (
    DEVICE, LEARNING_RATE, NUM_EPOCHS,
    OUTPUT_ROOT, LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE, LR_SCHEDULER_MIN_LR
)
from data_manager import get_data_loaders
from model import get_model
from losses import DetectionLoss
from visualization import VisualizationLogger
from training.trainer import Trainer

def main():
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

    # Setup optimizer with increased weight decay for better regularization
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=0.05  # Increased from 0.01
    )

    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        min_lr=LR_SCHEDULER_MIN_LR,
        verbose=True
    )

    # Get data loaders
    train_loader, val_loader = get_data_loaders()
    
    # Create trainer instance with additional parameters
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        visualization_logger=vis_logger,
        checkpoints_dir=checkpoints_dir
    )
    
    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()
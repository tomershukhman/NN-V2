import torch
from tqdm import tqdm
import os
from config import (
    DEVICE, LEARNING_RATE, NUM_EPOCHS, OUTPUT_ROOT,
    WEIGHT_DECAY, DATA_ROOT, DOG_USAGE_RATIO, TRAIN_VAL_SPLIT
)
from dog_detector.data import get_data_loaders, CocoDogsDataset
from dog_detector.model.model import get_model
from dog_detector.model.losses import DetectionLoss
from dog_detector.visualization.visualization import VisualizationLogger



def train(data_root=None, download=True, batch_size=None):
    """Train the dog detection model"""
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
        
    # Log dataset statistics to tensorboard only, without printing them again
    stats = CocoDogsDataset.get_dataset_stats(data_root)
    
    if stats:
        vis_logger.log_metrics({
            'dataset/total_available_dogs': stats.get('total_available_dogs', 0),
            'dataset/train_with_dogs': stats.get('train_with_dogs', 0),
            'dataset/train_without_dogs': stats.get('train_without_dogs', 0),
            'dataset/val_with_dogs': stats.get('val_with_dogs', 0),
            'dataset/val_without_dogs': stats.get('val_without_dogs', 0),
            'dataset/dog_usage_ratio': stats.get('dog_usage_ratio', DOG_USAGE_RATIO)
        }, 0, 'stats')

    # Get model and criterion
    model = get_model(DEVICE)
    criterion = DetectionLoss().to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        train_loss = 0
        train_steps = 0

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
        for images, targets in train_bar:
            # Prepare batch
            images = torch.stack([img.to(DEVICE) for img in images])
            targets = [{k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                        for k, v in t.items()} for t in targets]

            # Forward pass
            predictions = model(images)
            loss = criterion(predictions, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            train_loss += loss.item()
            train_steps += 1
            train_bar.set_postfix({'loss': train_loss / train_steps})

        # Validate
        model.eval()
        val_loss = 0
        val_steps = 0

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc='Validating'):
                images = torch.stack([img.to(DEVICE) for img in images])
                targets = [{k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                           for k, v in t.items()} for t in targets]

                predictions = model(images)
                loss = criterion(predictions, targets)
                val_loss += loss.item()
                val_steps += 1

        # Calculate epoch metrics
        train_loss = train_loss / train_steps
        val_loss = val_loss / val_steps

        # Log metrics
        print(f'\nEpoch {epoch+1}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')

        vis_logger.log_metrics(
            {'total_loss': train_loss}, epoch, 'train')
        vis_logger.log_metrics({'total_loss': val_loss}, epoch, 'val')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(OUTPUT_ROOT, 'checkpoints', 'best_model.pth'))
            print(f'✨ New best model saved (val_loss: {val_loss:.4f})')

        print('-' * 80)

    vis_logger.close()
    print('\nTraining completed!')

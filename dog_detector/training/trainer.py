import torch
from tqdm import tqdm
import os
from config import (
    DEVICE, LEARNING_RATE, NUM_EPOCHS, OUTPUT_ROOT,
    WEIGHT_DECAY
)
from dog_detector.data import get_data_loaders
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

    # Calculate dataset statistics
    train_with_dogs = 0
    train_without_dogs = 0
    val_with_dogs = 0
    val_without_dogs = 0

    print('\nCalculating dataset statistics...')
    for images, targets in tqdm(train_loader, desc='Analyzing training set'):
        for target in targets:
            if len(target['boxes']) > 0:
                train_with_dogs += 1
            else:
                train_without_dogs += 1

    for images, targets in tqdm(val_loader, desc='Analyzing validation set'):
        for target in targets:
            if len(target['boxes']) > 0:
                val_with_dogs += 1
            else:
                val_without_dogs += 1

    print('\nDataset Statistics:')
    print('Training set:')
    print(f'  Images with dogs: {train_with_dogs}')
    print(f'  Images without dogs: {train_without_dogs}')
    print('Validation set:')
    print(f'  Images with dogs: {val_with_dogs}')
    print(f'  Images without dogs: {val_without_dogs}\n')

    # Log dataset statistics to tensorboard
    vis_logger.log_metrics({
        'dataset/train_with_dogs': train_with_dogs,
        'dataset/train_without_dogs': train_without_dogs,
        'dataset/val_with_dogs': val_with_dogs,
        'dataset/val_without_dogs': val_without_dogs
    }, 0, 'stats')

    # Get model and criterion
    model = get_model(DEVICE)
    criterion = DetectionLoss().to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Get data loaders
    train_loader, val_loader = get_data_loaders(
        root=data_root,
        download=download,
        batch_size=batch_size
    )

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

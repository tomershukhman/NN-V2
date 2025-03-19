import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from data_manager.open_images_manager import OpenImagesV7Manager
import numpy as np

def denormalize(tensor):
    """Denormalize the tensor back to image format"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def show_dataset_samples(dataset, title, num_samples=16):
    """Display a grid of images from the dataset"""
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    fig.suptitle(title, fontsize=16)
    
    # Get random indices
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for idx, ax in zip(indices, axes.flat):
        img, label = dataset[idx]
        
        # Denormalize and convert to numpy
        img = denormalize(img)
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'{"Dog" if label == 1 else "Not Dog"}')
    
    plt.tight_layout()
    return fig

def main():
    # Initialize the dataset manager
    manager = OpenImagesV7Manager()
    
    # Create basic transforms without augmentation for visualization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get datasets with the same transform for both train and val
    train_dataset, val_dataset = manager.get_datasets(
        train_transform=transform,
        val_transform=transform
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create and show the plots
    train_fig = show_dataset_samples(train_dataset, "Training Dataset Samples")
    val_fig = show_dataset_samples(val_dataset, "Validation Dataset Samples")
    
    plt.show()

if __name__ == "__main__":
    main()
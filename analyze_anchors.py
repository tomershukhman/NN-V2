import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dog_detector.model.model import DogDetector
from dog_detector.data import CocoDogsDataset
import config
import torchvision.transforms as transforms
from PIL import Image
import random

def visualize_anchors(model, image_tensor, num_anchors_to_show=100, save_path=None):
    """
    Visualizes a subset of anchors on an image
    
    Args:
        model: The DogDetector model
        image_tensor: Image tensor of shape [C, H, W]
        num_anchors_to_show: Number of anchors to visualize (randomly sampled)
        save_path: Path to save the visualization, if None will show instead
    """
    # Convert image tensor to CPU then numpy for visualization
    img = image_tensor.cpu().permute(1, 2, 0).numpy()
    
    # Denormalize image
    for i in range(3):
        img[:, :, i] = img[:, :, i] * config.STD[i] + config.MEAN[i]
    img = np.clip(img, 0, 1)
    
    # Forward pass to generate anchors
    with torch.no_grad():
        model.eval()
        # Add batch dimension
        batch = image_tensor.unsqueeze(0)
        
        # Forward pass to generate anchors (prints debug info)
        cls_output, reg_output, anchors = model(batch)
    
    # Sample random anchors to visualize
    if len(anchors) > num_anchors_to_show:
        indices = np.random.choice(len(anchors), num_anchors_to_show, replace=False)
        sampled_anchors = anchors[indices].cpu().numpy()
    else:
        sampled_anchors = anchors.cpu().numpy()
    
    # Create plot with multiple anchor colors based on size
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    
    # Group anchors by approximate area for color coding
    small_anchors = []
    medium_anchors = []
    large_anchors = []
    
    for anchor in sampled_anchors:
        x1, y1, x2, y2 = anchor
        area = (x2 - x1) * (y2 - y1)
        
        if area < 3000:  # Small anchors
            small_anchors.append(anchor)
        elif area < 15000:  # Medium anchors
            medium_anchors.append(anchor)
        else:  # Large anchors
            large_anchors.append(anchor)
    
    # Plot anchors by group
    for anchor in small_anchors:
        x1, y1, x2, y2 = anchor
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, 
                                linewidth=1, edgecolor='blue', facecolor='none', alpha=0.3)
        plt.gca().add_patch(rect)
    
    for anchor in medium_anchors:
        x1, y1, x2, y2 = anchor
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, 
                                linewidth=1, edgecolor='green', facecolor='none', alpha=0.3)
        plt.gca().add_patch(rect)
        
    for anchor in large_anchors:
        x1, y1, x2, y2 = anchor
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, 
                                linewidth=1, edgecolor='red', facecolor='none', alpha=0.3)
        plt.gca().add_patch(rect)
    
    # Add legend
    plt.gca().add_patch(patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='blue', 
                                         facecolor='none', label=f'Small ({len(small_anchors)})'))
    plt.gca().add_patch(patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='green', 
                                         facecolor='none', label=f'Medium ({len(medium_anchors)})'))
    plt.gca().add_patch(patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='red', 
                                         facecolor='none', label=f'Large ({len(large_anchors)})'))
    
    plt.legend(loc='upper right')
    plt.title(f'Sample of {num_anchors_to_show} anchors out of {len(anchors)} total')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    plt.close()


def analyze_anchor_distribution(anchors):
    """
    Analyzes the distribution of anchor sizes and aspect ratios
    """
    # Calculate widths, heights, areas and aspect ratios
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    areas = widths * heights
    aspect_ratios = widths / heights
    
    # Print statistics
    print("\n===== ANCHOR DISTRIBUTION ANALYSIS =====")
    print(f"Total anchors: {len(anchors)}")
    print("\nWidth statistics:")
    print(f"  Min: {widths.min().item():.1f}")
    print(f"  Max: {widths.max().item():.1f}")
    print(f"  Mean: {widths.mean().item():.1f}")
    print(f"  Median: {widths.median().item():.1f}")
    
    print("\nHeight statistics:")
    print(f"  Min: {heights.min().item():.1f}")
    print(f"  Max: {heights.max().item():.1f}")
    print(f"  Mean: {heights.mean().item():.1f}")
    print(f"  Median: {heights.median().item():.1f}")
    
    print("\nArea statistics:")
    print(f"  Min: {areas.min().item():.1f}")
    print(f"  Max: {areas.max().item():.1f}")
    print(f"  Mean: {areas.mean().item():.1f}")
    print(f"  Median: {areas.median().item():.1f}")
    
    print("\nAspect ratio statistics (width/height):")
    print(f"  Min: {aspect_ratios.min().item():.3f}")
    print(f"  Max: {aspect_ratios.max().item():.3f}")
    print(f"  Mean: {aspect_ratios.mean().item():.3f}")
    print(f"  Values: {sorted([round(ar.item(), 3) for ar in aspect_ratios.unique()])}")
    
    # Calculate anchor counts by size buckets
    small_count = (areas < 3000).sum().item()
    medium_count = ((areas >= 3000) & (areas < 15000)).sum().item()
    large_count = (areas >= 15000).sum().item()
    
    print(f"\nAnchor sizes:")
    print(f"  Small (<3000 px²): {small_count} ({small_count/len(areas)*100:.1f}%)")
    print(f"  Medium (3000-15000 px²): {medium_count} ({medium_count/len(areas)*100:.1f}%)")
    print(f"  Large (>15000 px²): {large_count} ({large_count/len(areas)*100:.1f}%)")
    
    return {
        'widths': widths,
        'heights': heights,
        'areas': areas, 
        'aspect_ratios': aspect_ratios
    }


def analyze_anchor_coverage(model, gt_boxes, image_tensor=None):
    """
    Analyzes how well the anchors cover the ground truth boxes
    
    Args:
        model: The DogDetector model
        gt_boxes: Ground truth boxes tensor [N, 4]
        image_tensor: Optional image tensor for visualization
    """
    device = gt_boxes.device
    
    # Generate anchors by running a forward pass
    with torch.no_grad():
        # Use either provided image or create a dummy one
        if image_tensor is None:
            dummy_input = torch.zeros(1, 3, config.IMAGE_SIZE[1], config.IMAGE_SIZE[0], device=device)
            cls_output, reg_output, anchors = model(dummy_input)
        else:
            # Add batch dimension
            batch = image_tensor.unsqueeze(0).to(device)
            cls_output, reg_output, anchors = model(batch)
    
    # Calculate IoUs between all anchors and ground truth boxes
    def compute_iou(boxes1, boxes2):
        """Compute IoU between two sets of boxes"""
        # Calculate intersection coordinates
        x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
        y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
        x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
        y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
        
        # Calculate intersection area
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Calculate areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # Calculate union
        union = area1[:, None] + area2[None, :] - intersection
        
        # Calculate IoU
        iou = intersection / torch.clamp(union, min=1e-6)
        return iou
    
    # Skip if no ground truth boxes
    if len(gt_boxes) == 0:
        print("No ground truth boxes provided for analysis.")
        return
    
    # Calculate IoUs
    iou_matrix = compute_iou(anchors, gt_boxes)
    
    # For each GT box, find the highest IoU with any anchor
    max_ious, _ = iou_matrix.max(dim=0)
    
    # For each anchor, find the highest IoU with any GT box
    best_anchor_ious, _ = iou_matrix.max(dim=1)
    
    # Analyze coverage
    print("\n===== ANCHOR COVERAGE ANALYSIS =====")
    print(f"Ground truth boxes: {len(gt_boxes)}")
    print(f"Total anchors: {len(anchors)}")
    
    # Stats on GT coverage
    print("\nGround truth coverage stats (max IoU with any anchor):")
    print(f"  Min IoU: {max_ious.min().item():.3f}")
    print(f"  Max IoU: {max_ious.max().item():.3f}")
    print(f"  Mean IoU: {max_ious.mean().item():.3f}")
    print(f"  GT boxes with IoU > 0.5: {(max_ious > 0.5).sum().item()}/{len(max_ious)} ({(max_ious > 0.5).sum().item()/len(max_ious)*100:.1f}%)")
    print(f"  GT boxes with IoU > 0.7: {(max_ious > 0.7).sum().item()}/{len(max_ious)} ({(max_ious > 0.7).sum().item()/len(max_ious)*100:.1f}%)")
    
    # Stats on anchor quality 
    print("\nAnchor quality stats (IoU with GT boxes):")
    print(f"  Anchors with IoU > 0.3: {(best_anchor_ious > 0.3).sum().item()}/{len(best_anchor_ious)} ({(best_anchor_ious > 0.3).sum().item()/len(best_anchor_ious)*100:.3f}%)")
    print(f"  Anchors with IoU > 0.5: {(best_anchor_ious > 0.5).sum().item()}/{len(best_anchor_ious)} ({(best_anchor_ious > 0.5).sum().item()/len(best_anchor_ious)*100:.3f}%)")
    print(f"  Anchors with IoU > 0.7: {(best_anchor_ious > 0.7).sum().item()}/{len(best_anchor_ious)} ({(best_anchor_ious > 0.7).sum().item()/len(best_anchor_ious)*100:.3f}%)")
    
    return max_ious, best_anchor_ious


def plot_gt_size_distribution(gt_boxes, dataset_name="Dataset", save_path=None):
    """
    Plot distribution of ground truth box sizes
    """
    if len(gt_boxes) == 0:
        print("No ground truth boxes to analyze.")
        return
    
    # Calculate width, height, area and aspect ratio
    widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    areas = widths * heights
    aspect_ratios = widths / heights
    
    # Convert to numpy
    widths = widths.cpu().numpy()
    heights = heights.cpu().numpy()
    areas = areas.cpu().numpy()
    aspect_ratios = aspect_ratios.cpu().numpy()
    
    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot width distribution
    axs[0, 0].hist(widths, bins=30, alpha=0.7)
    axs[0, 0].set_title('Width Distribution')
    axs[0, 0].axvline(x=np.median(widths), color='r', linestyle='--', label=f'Median: {np.median(widths):.1f}')
    axs[0, 0].axvline(x=np.mean(widths), color='g', linestyle='--', label=f'Mean: {np.mean(widths):.1f}')
    axs[0, 0].legend()
    
    # Plot height distribution
    axs[0, 1].hist(heights, bins=30, alpha=0.7)
    axs[0, 1].set_title('Height Distribution')
    axs[0, 1].axvline(x=np.median(heights), color='r', linestyle='--', label=f'Median: {np.median(heights):.1f}')
    axs[0, 1].axvline(x=np.mean(heights), color='g', linestyle='--', label=f'Mean: {np.mean(heights):.1f}')
    axs[0, 1].legend()
    
    # Plot area distribution
    axs[1, 0].hist(areas, bins=30, alpha=0.7)
    axs[1, 0].set_title('Area Distribution')
    axs[1, 0].axvline(x=np.median(areas), color='r', linestyle='--', label=f'Median: {np.median(areas):.1f}')
    axs[1, 0].axvline(x=np.mean(areas), color='g', linestyle='--', label=f'Mean: {np.mean(areas):.1f}')
    axs[1, 0].legend()
    
    # Plot aspect ratio distribution
    axs[1, 1].hist(aspect_ratios, bins=30, alpha=0.7)
    axs[1, 1].set_title('Aspect Ratio Distribution (width/height)')
    axs[1, 1].axvline(x=np.median(aspect_ratios), color='r', linestyle='--', label=f'Median: {np.median(aspect_ratios):.2f}')
    axs[1, 1].axvline(x=np.mean(aspect_ratios), color='g', linestyle='--', label=f'Mean: {np.mean(aspect_ratios):.2f}')
    axs[1, 1].legend()
    
    plt.suptitle(f'Ground Truth Box Statistics - {dataset_name} (n={len(gt_boxes)})')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()
    
    # Print statistics
    print(f"\n===== GROUND TRUTH BOX STATISTICS ({dataset_name}) =====")
    print(f"Number of boxes: {len(gt_boxes)}")
    print("\nWidth statistics:")
    print(f"  Min: {np.min(widths):.1f}")
    print(f"  Max: {np.max(widths):.1f}")
    print(f"  Mean: {np.mean(widths):.1f}")
    print(f"  Median: {np.median(widths):.1f}")
    
    print("\nHeight statistics:")
    print(f"  Min: {np.min(heights):.1f}")
    print(f"  Max: {np.max(heights):.1f}")
    print(f"  Mean: {np.mean(heights):.1f}")
    print(f"  Median: {np.median(heights):.1f}")
    
    print("\nArea statistics:")
    print(f"  Min: {np.min(areas):.1f}")
    print(f"  Max: {np.max(areas):.1f}")
    print(f"  Mean: {np.mean(areas):.1f}")
    print(f"  Median: {np.median(areas):.1f}")
    
    print("\nAspect ratio statistics (width/height):")
    print(f"  Min: {np.min(aspect_ratios):.3f}")
    print(f"  Max: {np.max(aspect_ratios):.3f}")
    print(f"  Mean: {np.mean(aspect_ratios):.3f}")
    print(f"  Median: {np.median(aspect_ratios):.3f}")


def main():
    # Create output directory for visualizations
    output_dir = os.path.join("outputs", "anchor_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize device
    device = torch.device(config.DEVICE)
    
    # Initialize model
    print("Initializing model...")
    model = DogDetector()
    model.to(device)
    model.eval()
    
    # Try to load model weights if available (optional)
    best_model_path = os.path.join(config.OUTPUT_ROOT, 'checkpoints', 'best_model_f1.pth')
    if os.path.exists(best_model_path):
        try:
            print(f"Loading model weights from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Could not load model weights: {e}")
    
    # Load a few validation images for analysis
    print("Loading dataset...")
    val_dataset = CocoDogsDataset(config.DATA_ROOT, is_train=False)
    
    # Find images with dogs (we want GT boxes for analysis)
    dog_images = []
    dog_targets = []
    
    # Gather all ground truth boxes for analysis
    all_gt_boxes = []
    
    print("Finding images with dogs for analysis...")
    for i in range(min(100, len(val_dataset))):  # Check first 100 images
        try:
            image, target = val_dataset[i]
            if len(target['boxes']) > 0:  # Has at least one dog
                dog_images.append(image)
                dog_targets.append(target)
                all_gt_boxes.append(target['boxes'].to(device))
                
                if len(dog_images) >= 10:  # Get 10 images with dogs
                    break
        except Exception as e:
            print(f"Error loading image {i}: {e}")
    
    if not dog_images:
        print("No images with dogs found, using a sample image without dogs")
        # Just use the first image if no dogs found
        image, target = val_dataset[0]
        dog_images.append(image)
        dog_targets.append(target)
    
    # Concatenate all GT boxes
    all_gt_boxes = torch.cat(all_gt_boxes, dim=0) if all_gt_boxes else torch.zeros((0, 4), device=device)
    
    # Plot ground truth box size distribution
    if len(all_gt_boxes) > 0:
        plot_gt_size_distribution(
            all_gt_boxes, 
            "Validation Dogs", 
            save_path=os.path.join(output_dir, "gt_box_distribution.png")
        )

    # Generate and analyze anchors
    for i, (image, target) in enumerate(zip(dog_images, dog_targets)):
        if i >= 3:  # Only process first 3 images
            break
            
        # Move tensors to device
        image = image.to(device)
        target_boxes = target['boxes'].to(device)
        
        # Visualize anchors on image
        print(f"\n==== Analyzing anchors for image {i+1} ====")
        visualize_anchors(
            model, 
            image, 
            num_anchors_to_show=200,
            save_path=os.path.join(output_dir, f"anchors_image_{i+1}.png")
        )
        
        # Forward pass to generate anchors
        with torch.no_grad():
            cls_output, reg_output, anchors = model(image.unsqueeze(0))
            
        # Analyze anchor size distribution
        stats = analyze_anchor_distribution(anchors)
        
        # Analyze anchor coverage of ground truth boxes if there are any
        if len(target_boxes) > 0:
            analyze_anchor_coverage(model, target_boxes, image)
        
    print(f"\nAnalysis complete. Visualizations saved to {output_dir}")
    
    # Recommendations based on analysis
    print("\n===== RECOMMENDATIONS =====")
    print("Based on the analysis, check the following:")
    print("1. Verify if the anchor scales match the GT box size distribution")
    print("2. Make sure aspect ratios cover the variety in your dataset")
    print("3. Ensure anchors have good overlap with ground truth boxes (IoU)")
    print("4. Consider adding more anchors if coverage is poor")
    
    print(f"\nCurrent anchor scales: {config.ANCHOR_SCALES}")
    print(f"Current anchor ratios: {config.ANCHOR_RATIOS}")


if __name__ == "__main__":
    main()
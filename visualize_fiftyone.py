import fiftyone as fo
import os
from data_manager.open_images_manager import OpenImagesV7Manager

def visualize_datasets():
    # Initialize the dataset manager
    manager = OpenImagesV7Manager()
    
    # Get the splits
    splits = manager.get_dataset_splits()
    
    # Create a single FiftyOne dataset for both train and val
    dataset = fo.Dataset("dog_classification_dataset")
    
    # Add samples from both splits
    for split_name, samples in splits.items():
        for sample_info in samples:
            filepath = sample_info['filepath']
            if os.path.exists(filepath):
                sample = fo.Sample(filepath=filepath)
                sample.tags = ["dog"] if sample_info['label'] == 1 else ["non_dog"]
                # Add split information as a field
                sample["split"] = split_name
                dataset.add_sample(sample)
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(dataset)}")
    print("\nTraining Set:")
    train_view = dataset.match(F("split") == "train")
    print(f"Total: {len(train_view)}")
    print(f"Dogs: {len(train_view.match_tags('dog'))}")
    print(f"Non-dogs: {len(train_view.match_tags('non_dog'))}")
    
    print("\nValidation Set:")
    val_view = dataset.match(F("split") == "val")
    print(f"Total: {len(val_view)}")
    print(f"Dogs: {len(val_view.match_tags('dog'))}")
    print(f"Non-dogs: {len(val_view.match_tags('non_dog'))}")
    
    # Launch the FiftyOne App
    session = fo.launch_app(dataset)
    print("\nLaunched FiftyOne App. You can use the following filters:")
    print("1. Filter by split: Click 'Add stage' → 'Match' → split == 'train' or 'val'")
    print("2. Filter by class: Click on the tags in the sidebar")
    print("\nPress Ctrl+C to exit...")
    
    try:
        session.wait()
    except KeyboardInterrupt:
        pass
    
    # Cleanup
    dataset.delete()

if __name__ == "__main__":
    from fiftyone import F  # Import F here to use in field expressions
    visualize_datasets()
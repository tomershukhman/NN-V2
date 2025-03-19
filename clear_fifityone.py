import fiftyone as fo

# List all datasets
datasets = fo.list_datasets()

# Delete each dataset
for dataset_name in datasets:
    print(f"Deleting dataset: {dataset_name}")
    fo.delete_dataset(dataset_name)

print("All local FiftyOne datasets have been deleted.")

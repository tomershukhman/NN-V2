import argparse
import os
import config
from dog_detector.training.trainer import train
from dog_detector.utils import verify_dataset_integrity, download_coco_dataset
from config import DATA_ROOT

def main():
    # Check dataset integrity and download if needed
    if not verify_dataset_integrity(DATA_ROOT):
        print("Dataset incomplete or missing. Downloading COCO dataset...")
        download_coco_dataset(DATA_ROOT)
    
    # Start training
    train(data_root=DATA_ROOT, download=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dog Detector Training using COCO2017")
    args = parser.parse_args()
    main()

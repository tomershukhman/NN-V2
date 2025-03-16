import argparse
import os
import config
from dog_detector.training.trainer import train
from dog_detector.utils import verify_dataset_integrity, download_coco_dataset

def main():
    # Check dataset integrity and download if needed
    if not verify_dataset_integrity(config.DATA_ROOT):
        print("Dataset incomplete or missing. Downloading COCO dataset...")
        download_coco_dataset(config.DATA_ROOT)
    
    # Start training
    train(data_root=config.DATA_ROOT, download=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dog Detector Training using COCO2017")
    args = parser.parse_args()
    main()

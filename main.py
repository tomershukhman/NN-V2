#!/usr/bin/env python
"""
Dog Detector - Main entry point

This script provides an entry point to the dog detector application.
It allows training or using the dog detection model.
"""
import os
import argparse
from config import OUTPUT_ROOT

def parse_args():
    parser = argparse.ArgumentParser(description='Dog Detection model')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'predict'],
                      help='Mode to run: train, eval, or predict')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to model checkpoint for evaluation or prediction')
    parser.add_argument('--data_root', type=str, default=None,
                      help='Path to data directory (overrides config)')
    parser.add_argument('--no_download', action='store_true', 
                      help='Skip dataset download')
    parser.add_argument('--image_path', type=str, default=None,
                      help='Path to image for prediction')
    parser.add_argument('--batch_size', type=int, default=None, 
                      help='Batch size for training/evaluation')
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.mode == 'train':
        from dog_detector.training.trainer import train
        train(
            data_root=args.data_root,
            download=(not args.no_download),
            batch_size=args.batch_size
        )
    
    elif args.mode == 'eval':
        if not args.checkpoint:
            print("Error: Checkpoint required for evaluation mode")
            return
            
        from dog_detector.training.eval import evaluate
        evaluate(
            checkpoint_path=args.checkpoint,
            data_root=args.data_root,
            batch_size=args.batch_size
        )
    
    elif args.mode == 'predict':
        if not args.image_path:
            print("Error: Image path required for prediction mode")
            return
        if not args.checkpoint:
            # Look for best model in checkpoints directory
            checkpoint_dir = os.path.join(OUTPUT_ROOT, 'checkpoints')
            best_model = os.path.join(checkpoint_dir, 'best_f1_model.pth')
            if os.path.exists(best_model):
                args.checkpoint = best_model
            else:
                print("Error: No checkpoint specified and no best model found")
                return
                
        from dog_detector.utils.inference import predict_on_image
        predict_on_image(
            image_path=args.image_path,
            checkpoint_path=args.checkpoint
        )

if __name__ == "__main__":
    main()
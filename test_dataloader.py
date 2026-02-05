#!/usr/bin/env python3
"""
Test DataLoader to debug slow first batch issue.
"""
import time
import torch
import numpy as np
from utils import load_data, print_timing_stats

def test_dataloader():
    print("\n" + "="*80)
    print("TESTING DATALOADER")
    print("="*80 + "\n")
    
    dataset_dir = './data/data_ik/aura_k1'
    num_episodes = 900
    camera_names = ['rgb_main', 'rgb_left', 'rgb_under']  # 使用实际的相机名称
    batch_size = 2
    chunk_size = 20
    
    print(f"Dataset directory: {dataset_dir}")
    print(f"Number of episodes: {num_episodes}")
    print(f"Cameras: {camera_names}")
    print(f"Batch size: {batch_size}")
    print(f"Chunk size (num_action): {chunk_size}")
    
    # Load data using the same function as training
    print("\nLoading data...")
    load_start = time.time()
    train_dataloader, val_dataloader, norm_stats, is_sim = load_data(
        dataset_dir, 
        num_episodes, 
        camera_names, 
        batch_size, 
        batch_size,
        chunk_size=chunk_size,
        num_obs=1
    )
    load_end = time.time()
    
    print(f"\nData loading complete in {load_end - load_start:.2f}s")
    print(f"Train batches: {len(train_dataloader)}")
    print(f"Val batches: {len(val_dataloader)}")
    print(f"Is simulation: {is_sim}")
    print(f"\nData loading complete in {load_end - load_start:.2f}s")
    print(f"Train batches: {len(train_dataloader)}")
    print(f"Val batches: {len(val_dataloader)}")
    print(f"Is simulation: {is_sim}")
    
    # Test first few batches
    num_test_batches = 3
    print(f"\n{'='*80}")
    print(f"Testing first {num_test_batches} batches...")
    print(f"{'='*80}\n")
    
    overall_start = time.time()
    
    for batch_idx, data in enumerate(train_dataloader):
        if batch_idx >= num_test_batches:
            break
            
        batch_start = time.time()
        print(f"Batch {batch_idx+1}/{num_test_batches}:")
        
        # Unpack data
        image_data, qpos_data, action_data, is_pad = data
        
        batch_end = time.time()
        
        # Print shapes and timing
        print(f"  Image shape: {image_data.shape}")
        print(f"  Qpos shape: {qpos_data.shape}")
        print(f"  Action shape: {action_data.shape}")
        print(f"  Is pad shape: {is_pad.shape}")
        print(f"  Batch time: {batch_end - batch_start:.3f}s")
        if torch.cuda.is_available():
            print(f"  GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        print()
    
    overall_end = time.time()
    avg_time = (overall_end - overall_start) / num_test_batches
    print(f"{'='*80}")
    print(f"Total time for {num_test_batches} batches: {overall_end - overall_start:.2f}s")
    print(f"Average time per batch: {avg_time:.2f}s")
    print(f"{'='*80}\n")
    
    # Print timing statistics
    print("="*80)
    print("DETAILED TIMING STATISTICS")
    print("="*80)
    print_timing_stats()
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")

if __name__ == '__main__':
    test_dataloader()

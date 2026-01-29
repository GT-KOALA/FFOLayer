#!/usr/bin/env python
"""
Peak RSS Memory Benchmark: ffocp_eq vs cvxpylayer
Benchmarks memory usage across ydim from 100 to 1000.
"""

import os
import sys
import time
import gc

import numpy as np
import torch
import pandas as pd

from models import OptModel
from data import genData
from utils_synthetic import PeakRSS, bytes_to_gb

# Configuration
BATCH_SIZE = 8
SEED = 3
INPUT_DIM = 640
YDIM_LIST = list(range(100, 1001, 100))  # [100, 200, 300, ..., 1000]
NUM_SAMPLES = 2000
LEARNING_RATE = 0.001

# Hyperparameters for ffocp_eq
ALPHA = 100
DUAL_CUTOFF = 1e-3
SLACK_TOL = 1e-8
BACKWARD_EPS = 1e-3

# Device
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def cleanup():
    """Clean up memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def benchmark_method(method_name, ydim):
    """
    Benchmark a single method for initialization and first iteration peak RSS.
    """
    cleanup()
    set_seed(SEED)
    
    results = {'ydim': ydim, 'method': method_name}
    
    # Generate data
    train_loader, test_loader = genData(device, INPUT_DIM, ydim, NUM_SAMPLES, BATCH_SIZE)
    
    # Measure initialization peak RSS
    with PeakRSS() as m_init:
        model = OptModel(
            INPUT_DIM, ydim, 
            layer_type=method_name, 
            constraint_learnable=False, 
            batch_size=BATCH_SIZE, 
            device=device, 
            alpha=ALPHA, 
            dual_cutoff=DUAL_CUTOFF, 
            slack_tol=SLACK_TOL, 
            backward_eps=BACKWARD_EPS, 
            is_QP=True
        ).to(device)
    
    results['init_peak_rss_gb'] = bytes_to_gb(m_init.peak)
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0)
    loss_fn = torch.nn.MSELoss()
    
    # Get first batch
    x, y = next(iter(train_loader))
    
    model.train()
    
    # Measure forward pass peak RSS
    with PeakRSS() as m_fwd:
        start_fwd = time.time()
        z, y_pred = model(x)
        forward_time = time.time() - start_fwd
    
    results['forward_peak_rss_gb'] = bytes_to_gb(m_fwd.peak)
    results['forward_time'] = forward_time
    
    # Compute loss
    ts_loss = loss_fn(y_pred, y)
    df_loss = torch.mean(y * z)
    loss = df_loss
    
    # Measure backward pass peak RSS
    with PeakRSS() as m_bwd:
        start_bwd = time.time()
        loss.backward()
        backward_time = time.time() - start_bwd
    
    results['backward_peak_rss_gb'] = bytes_to_gb(m_bwd.peak)
    results['backward_time'] = backward_time
    results['total_iter_time'] = forward_time + backward_time
    
    # Cleanup
    del model, optimizer, train_loader, test_loader
    cleanup()
    
    return results


def main():
    print(f"Using device: {device}")
    print(f"ydim values to benchmark: {YDIM_LIST}")
    
    all_results = []

    for ydim in YDIM_LIST:
        print(f"\n{'='*60}")
        print(f"Benchmarking ydim={ydim}")
        print(f"{'='*60}")
        
        # Benchmark ffocp_eq
        print(f"  Running ffocp_eq...")
        ffocp_result = benchmark_method('ffocp_eq', ydim)
        all_results.append(ffocp_result)
        print(f"    Init RSS: {ffocp_result['init_peak_rss_gb']:.4f} GB, "
              f"Forward RSS: {ffocp_result['forward_peak_rss_gb']:.4f} GB, "
              f"Backward RSS: {ffocp_result['backward_peak_rss_gb']:.4f} GB")
        
        # Benchmark cvxpylayer
        print(f"  Running cvxpylayer...")
        cvxpy_result = benchmark_method('cvxpylayer', ydim)
        all_results.append(cvxpy_result)
        print(f"    Init RSS: {cvxpy_result['init_peak_rss_gb']:.4f} GB, "
              f"Forward RSS: {cvxpy_result['forward_peak_rss_gb']:.4f} GB, "
              f"Backward RSS: {cvxpy_result['backward_peak_rss_gb']:.4f} GB")

    print(f"\n{'='*60}")
    print("All benchmarks completed!")
    print(f"{'='*60}")
    
    # Save results to CSV
    df = pd.DataFrame(all_results)
    df.to_csv('peakrss_benchmark_results.csv', index=False)
    print(f"\nResults saved to 'peakrss_benchmark_results.csv'")
    
    return df


if __name__ == '__main__':
    main()
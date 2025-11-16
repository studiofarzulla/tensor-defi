#!/usr/bin/env python3
"""
Debug OOM Issue in CP Decomposition

Traces memory usage during CP-ALS to identify where 128GB RAM gets consumed
by 3MB of tensor data.
"""

import sys
import gc
import psutil
import pickle
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from tensor_ops.decomposition import TensorDecomposer

def get_memory_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024**2

def print_memory(label):
    """Print memory usage with label."""
    mem_mb = get_memory_mb()
    print(f"{label:40s} {mem_mb:10.2f} MB")
    return mem_mb

def find_memory_leak_pattern():
    """Test if memory leak is cumulative across decompositions."""
    print("\n" + "="*80)
    print("CUMULATIVE MEMORY LEAK TEST")
    print("="*80)
    
    data_dir = Path(__file__).parent.parent / 'data' / 'tensors'
    with open(data_dir / 'normalized_ohlcv_tensor.pkl', 'rb') as f:
        data = pickle.load(f)
        tensor = data['tensor']
    
    decomposer = TensorDecomposer(backend='numpy')
    
    print(f"Tensor shape: {tensor.shape}")
    mem_baseline = print_memory("Baseline after loading")
    
    # Run multiple decompositions at different ranks
    ranks = [3, 5, 10, 15, 20]
    
    for i, rank in enumerate(ranks):
        print(f"\n--- Iteration {i+1}: Rank {rank} ---")
        mem_before = print_memory(f"Before rank {rank}")
        
        try:
            result = decomposer.cp_decomposition(tensor, rank=rank, verbose=False)
            mem_after = print_memory(f"After rank {rank}")
            print(f"  Variance: {result.explained_variance:.4f}")
            print(f"  Memory delta: {mem_after - mem_before:.2f} MB")
            
            del result
            gc.collect()
            mem_after_gc = print_memory(f"After GC")
            print(f"  Memory vs baseline: +{mem_after_gc - mem_baseline:.2f} MB")
            
        except MemoryError as e:
            print(f"\nOOM at rank {rank}!")
            print_memory("At failure")
            break

if __name__ == "__main__":
    print("\n" + "="*80)
    print("TENSOR DECOMPOSITION OOM DEBUGGER")
    print("="*80)
    print(f"\nSystem RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print(f"Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    
    find_memory_leak_pattern()
    
    print("\n" + "="*80)
    print("DEBUGGING COMPLETE")
    print("="*80)

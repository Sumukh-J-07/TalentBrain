#!/usr/bin/env python3
"""
benchmark.py — Run the pipeline and measure time/memory performance.
"""
import os
import sys
import time
from pathlib import Path

# Fix OpenMP duplicate lib error and threading segfaults on Mac with PyTorch/FAISS/Tokenizers
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import run_pipeline
from src.utils import log_memory

def run_benchmark():
    start_time = time.perf_counter()
    start_mem = log_memory("benchmark_start")
    
    run_pipeline(
        candidates_path="data/candidates.jsonl",
        output_path="outputs/submission.csv",
        models_dir="models",
        use_cache=True  # Force full run
    )
    
    end_time = time.perf_counter()
    end_mem = log_memory("benchmark_end")
    
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    print(f"Total Time:  {end_time - start_time:.2f} seconds")
    print(f"Peak Memory: ~{end_mem:.0f} MB")
    print("="*50)

if __name__ == "__main__":
    run_benchmark()

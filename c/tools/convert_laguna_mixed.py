#!/usr/bin/env python3
"""
Convert Laguna S 2.1 model with mixed precision:
- int4 for shared experts, dense MLP, and attention (important tensors)
- int2 for routed experts (256 per layer, only activated for some tokens)

This reduces model size from ~56 GB (int4) to ~28 GB (int2 experts + int4 rest),
allowing more experts to fit in RAM cache.

Usage:
  python3 convert_laguna_mixed.py --repo poolside/Laguna-S-2.1 --outdir ./laguna_i4_mixed

The conversion streams shards one at a time: download → convert → delete → repeat.
Peak disk usage: ~10 GB (one shard + converted output)
"""
import os
import sys
import subprocess

def main():
    # Use the existing converter with mixed precision flags
    # ebits=4: int4 for most tensors (attention, shared experts, dense MLP)
    # xbits=2: int2 for routed experts (the 256 experts per layer)
    # io_bits=8: int8 for embeddings and lm_head
    cmd = [
        sys.executable, "convert_laguna.py",
        "--repo", "poolside/Laguna-S-2.1",
        "--outdir", "./laguna_i4_mixed",
        "--ebits", "4",       # int4 for most weights
        "--io-bits", "8",     # int8 for embeddings/lm_head
        "--xbits", "2",       # int2 for routed experts (256 per layer)
        "--n-layers", "48",
        "--min-free-gb", "10",
        "--group-size", "128",  # Better quality with group scaling
    ]
    
    print("Starting mixed-precision conversion:")
    print("  Routed experts: int2 (256 experts/layer, ~28 GB total)")
    print("  Shared experts: int4 (~0.4 GB)")
    print("  Attention: int4 (~1.5 GB)")
    print("  Dense MLP: int4 (~0.1 GB)")
    print("  Embeddings: int8 (~0.3 GB)")
    print("  Estimated total: ~30 GB")
    print("  With 20 GB RAM: ~110 experts cached per layer (vs 55 with int4)")
    print()
    
    # Run the converter
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()

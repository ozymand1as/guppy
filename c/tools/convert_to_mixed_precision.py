#!/usr/bin/env python3
"""Convert existing int4 model to mixed precision (int2 experts, int4 rest).

Uses vectorized numpy operations for fast dequantization and re-quantization.

Usage:
  python3 convert_to_mixed_precision.py --indir /Volumes/files/laguna_i4 --outdir /Volumes/files/laguna_i4_mixed --xbits 2
"""
import argparse
import os
import sys
import json
import glob
import shutil
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

def main():
    parser = argparse.ArgumentParser(description="Convert int4 model to mixed precision (int2 experts)")
    parser.add_argument("--indir", required=True, help="Input directory (int4 model)")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--xbits", type=int, default=2, help="Bits for routed experts (default: 2)")
    args = parser.parse_args()
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from convert_fp8_to_int4 import quant_int2
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # Copy config and tokenizer
    for fn in ["config.json", "tokenizer.json", "tokenizer_config.json",
               "generation_config.json", "chat_template.jinja"]:
        src = os.path.join(args.indir, fn)
        if os.path.exists(src):
            shutil.copy(src, args.outdir)
    
    # Process shards
    shards = sorted(glob.glob(os.path.join(args.indir, "out-*.safetensors")))
    print(f"Processing {len(shards)} shards...")
    
    shard_idx = 0
    out_tensors = {}
    out_size = 0
    max_shard_size = 5 * 1024 * 1024 * 1024  # 5 GB
    
    def save_shard():
        nonlocal shard_idx, out_tensors, out_size
        if not out_tensors:
            return
        out_path = os.path.join(args.outdir, f"out-{shard_idx:05d}.safetensors")
        save_file(out_tensors, out_path)
        print(f"  Shard {shard_idx}: {len(out_tensors)} tensors, {out_size/1e9:.2f} GB")
        shard_idx += 1
        out_tensors = {}
        out_size = 0
    
    def dequant_int4_vec(q4_data, qs_data, O, I):
        """Dequantize int4 weights to float32 using vectorized numpy."""
        rb = (I + 1) // 2
        # Reshape to [O, rb]
        q4 = q4_data.reshape(O, rb)
        # Extract low and high nibbles
        lo = (q4 & 0x0F).astype(np.int32) - 8
        hi = ((q4 >> 4) & 0x0F).astype(np.int32) - 8
        # Interleave to get [O, I]
        w = np.empty((O, I), dtype=np.float32)
        w[:, 0::2] = lo[:, :I//2].astype(np.float32) * qs_data[:, None]
        if I > 1:
            w[:, 1::2] = hi[:, :I//2].astype(np.float32) * qs_data[:, None]
        return w
    
    for shard_path in shards:
        print(f"Processing {os.path.basename(shard_path)}...")
        with safe_open(shard_path, framework='pt') as f:
            keys = list(f.keys())
            
            for key in keys:
                if key.endswith('.qs'):
                    continue
                
                tensor = f.get_tensor(key)
                data = tensor.numpy() if hasattr(tensor, 'numpy') else np.array(tensor)
                
                # Check if this is an expert weight that needs re-quantization
                is_expert = "mlp.experts." in key and "e_score_correction_bias" not in key
                
                if is_expert and data.dtype == np.uint8:
                    qs_key = key + ".qs"
                    if qs_key not in keys:
                        print(f"  Warning: no scale for {key}, copying as-is")
                        out_tensors[key] = data
                        out_size += data.nbytes
                        if out_size > max_shard_size:
                            save_shard()
                        continue
                    
                    qs_data = f.get_tensor(qs_key).numpy()
                    
                    # Determine tensor shape from key name
                    if "gate_proj" in key or "up_proj" in key:
                        O, I = 1024, 3072
                    elif "down_proj" in key:
                        O, I = 3072, 1024
                    else:
                        print(f"  Warning: unknown expert tensor {key}, copying as-is")
                        out_tensors[key] = data
                        out_tensors[qs_key] = qs_data
                        out_size += data.nbytes + qs_data.nbytes
                        if out_size > max_shard_size:
                            save_shard()
                        continue
                    
                    # Check if it's int4 or int8 by comparing sizes
                    expected_int4 = O * ((I + 1) // 2)
                    expected_int8 = O * I
                    
                    if len(data) == expected_int4:
                        # int4 - dequantize to float32, then re-quantize to int2
                        w = dequant_int4_vec(data, qs_data, O, I)
                        
                        # Re-quantize to int2
                        q, s = quant_int2(w, args.xbits)
                        out_tensors[key] = q
                        out_tensors[key + ".qs"] = s
                        out_size += w.nbytes
                        if out_size > max_shard_size:
                            save_shard()
                    elif len(data) == expected_int8:
                        # int8 - dequantize to float32, then re-quantize to int2
                        w = (data.astype(np.float32) - 128) * qs_data[:, None]
                        
                        q, s = quant_int2(w, args.xbits)
                        out_tensors[key] = q
                        out_tensors[key + ".qs"] = s
                        out_size += w.nbytes
                        if out_size > max_shard_size:
                            save_shard()
                    else:
                        print(f"  Warning: unexpected size for {key}: {len(data)} (int4: {expected_int4}, int8: {expected_int8})")
                        out_tensors[key] = data
                        out_tensors[qs_key] = qs_data
                        out_size += data.nbytes + qs_data.nbytes
                else:
                    # Keep as-is (int4, int8, or float32)
                    out_tensors[key] = data
                    if key.endswith('.weight') and not key.endswith('.qs'):
                        qs_key = key + ".qs"
                        if qs_key in keys:
                            out_tensors[qs_key] = f.get_tensor(qs_key).numpy()
                    out_size += data.nbytes
                    if out_size > max_shard_size:
                        save_shard()
    
    save_shard()
    
    total_size = sum(os.path.getsize(os.path.join(args.outdir, f)) 
                     for f in os.listdir(args.outdir) if f.endswith('.safetensors'))
    print(f"\nConversion complete!")
    print(f"  Output: {args.outdir}")
    print(f"  Shards: {shard_idx}")
    print(f"  Total size: {total_size/1e9:.1f} GB")
    print(f"  Expert bits: {args.xbits}, Other bits: 4 (unchanged)")

if __name__ == "__main__":
    main()

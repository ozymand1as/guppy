#!/usr/bin/env python3
"""Convert GGUF model to Colibri int2/int4 format with streaming.

Downloads a GGUF file, extracts weights, and converts them to the colibri
safetensors format. Uses streaming to minimize disk usage.

Usage:
  python3 convert_gguf_to_colibri.py --gguf_repo unsloth/Laguna-S-2.1-GGUF \
    --gguf_file Laguna-S-2.1-UD-IQ1_S.gguf \
    --outdir ./laguna_i4_from_gguf \
    --ebits 4 --xbits 2 --io-bits 8

The GGUF file is downloaded, converted, and deleted. Peak disk usage is
approximately 2x the GGUF file size (GGUF + converted output).
"""
import argparse
import os
import sys
import json
import shutil
import time

def main():
    parser = argparse.ArgumentParser(description="Convert GGUF model to Colibri format")
    parser.add_argument("--gguf_repo", required=True, help="HuggingFace repo ID")
    parser.add_argument("--gguf_file", required=True, help="GGUF filename")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--ebits", type=int, default=4, help="Bits for non-expert weights (default: 4)")
    parser.add_argument("--xbits", type=int, default=2, help="Bits for routed experts (default: 2)")
    parser.add_argument("--io-bits", type=int, default=8, help="Bits for embeddings/lm_head (default: 8)")
    parser.add_argument("--group-size", type=int, default=128, help="Group size for int4 (default: 128)")
    parser.add_argument("--min-free-gb", type=float, default=10.0, help="Minimum free disk space (default: 10)")
    args = parser.parse_args()
    
    from huggingface_hub import hf_hub_download
    from gguf import GGUFReader
    import numpy as np
    from safetensors.numpy import save_file
    
    # Add tools directory to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from convert_fp8_to_int4 import quant_int4, quant_int2, quant_int8, quant_int4_grouped
    from convert_laguna import classify, rename_out
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # Download config and tokenizer from main model repo
    print("[1/4] Downloading config and tokenizer...")
    config_path = hf_hub_download("poolside/Laguna-S-2.1", "config.json", repo_type="model")
    tokenizer_path = hf_hub_download("poolside/Laguna-S-2.1", "tokenizer.json", repo_type="model")
    shutil.copy(config_path, args.outdir)
    shutil.copy(tokenizer_path, args.outdir)
    
    try:
        chat_template = hf_hub_download("poolside/Laguna-S-2.1", "chat_template.jinja", repo_type="model")
        shutil.copy(chat_template, args.outdir)
    except:
        pass
    
    with open(config_path) as f:
        config = json.load(f)
    n_layers = config.get("num_hidden_layers", 48)
    
    # Download GGUF file with progress
    print(f"[2/4] Downloading GGUF file: {args.gguf_file}")
    gguf_path = hf_hub_download(
        args.gguf_repo, args.gguf_file, repo_type="model",
        local_dir=os.path.join(args.outdir, "_gguf_cache")
    )
    gguf_size = os.path.getsize(gguf_path)
    print(f"  Downloaded: {gguf_size/1e9:.1f} GB")
    
    # Load GGUF file
    print("[3/4] Converting tensors...")
    reader = GGUFReader(gguf_path)
    print(f"  Tensors: {len(reader.tensors)}")
    
    # Process tensors in chunks
    shard_idx = 0
    shard_tensors = {}
    shard_size = 0
    max_shard_size = 5 * 1024 * 1024 * 1024  # 5 GB per shard
    total_tensors = len(reader.tensors)
    
    def save_shard():
        nonlocal shard_idx, shard_tensors, shard_size
        if not shard_tensors:
            return
        out_path = os.path.join(args.outdir, f"out-{shard_idx:05d}.safetensors")
        save_file(shard_tensors, out_path)
        print(f"  Shard {shard_idx}: {len(shard_tensors)} tensors, {shard_size/1e9:.2f} GB")
        shard_idx += 1
        shard_tensors = {}
        shard_size = 0
    
    for i, tensor in enumerate(reader.tensors):
        name = tensor.name
        data = tensor.data  # numpy array
        
        if (i + 1) % 50 == 0:
            print(f"  Processing tensor {i+1}/{total_tensors}: {name}")
        
        # Skip non-weight tensors
        if "weight" not in name and "bias" not in name:
            continue
        
        # Classify the tensor
        kind = classify(name, n_layers)
        
        # Determine bit width
        if kind == "io":
            bits = args.io_bits
        elif kind == "x":
            bits = args.xbits
        elif kind == "f32":
            shard_tensors[name] = data.astype(np.float32)
            shard_size += data.nbytes
            if shard_size > max_shard_size:
                save_shard()
            continue
        else:
            bits = args.ebits
        
        # Quantize the tensor
        if data.ndim == 1:
            shard_tensors[name] = data.astype(np.float32)
        else:
            O, I = data.shape
            w = data.astype(np.float32)
            
            if bits == 8:
                q, s = quant_int8(w, 8)
            elif bits == 4:
                if args.group_size > 0:
                    q, s = quant_int4_grouped(w, 4, args.group_size)
                else:
                    q, s = quant_int4(w, 4)
            elif bits == 2:
                q, s = quant_int2(w, 2)
            else:
                raise ValueError(f"Unsupported bit width: {bits}")
            
            shard_tensors[name] = q
            shard_tensors[name + ".qs"] = s
        
        shard_size += data.nbytes
        if shard_size > max_shard_size:
            save_shard()
    
    save_shard()
    
    # Clean up GGUF file
    print(f"[4/4] Cleaning up...")
    print(f"  Removing GGUF file ({gguf_size/1e9:.1f} GB)...")
    os.remove(gguf_path)
    shutil.rmtree(os.path.join(args.outdir, "_gguf_cache"), ignore_errors=True)
    
    total_size = sum(os.path.getsize(os.path.join(args.outdir, f)) 
                     for f in os.listdir(args.outdir) if f.endswith('.safetensors'))
    print(f"\nConversion complete!")
    print(f"  Output: {args.outdir}")
    print(f"  Shards: {shard_idx}")
    print(f"  Total size: {total_size/1e9:.1f} GB")
    print(f"  Expert bits: {args.xbits}, Other bits: {args.ebits}, Embed bits: {args.io_bits}")

if __name__ == "__main__":
    main()

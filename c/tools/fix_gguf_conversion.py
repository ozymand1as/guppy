#!/usr/bin/env python3
"""Fix GGUF-converted model tensor names and split expert tensors.

The GGUF format stores all experts in single 3D tensors, but the colibri
engine expects separate 2D tensors per expert. This script fixes the naming
and splits the expert tensors.

Usage:
  python3 fix_gguf_conversion.py --indir ./laguna_i4_from_gguf --outdir ./laguna_i4_fixed
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
    parser = argparse.ArgumentParser(description="Fix GGUF-converted model")
    parser.add_argument("--indir", required=True, help="Input directory")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--ebits", type=int, default=4, help="Bits for non-expert weights")
    parser.add_argument("--xbits", type=int, default=2, help="Bits for routed experts")
    parser.add_argument("--io-bits", type=int, default=8, help="Bits for embeddings")
    parser.add_argument("--group-size", type=int, default=128, help="Group size for int4")
    args = parser.parse_args()
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from convert_fp8_to_int4 import quant_int4, quant_int2, quant_int8, quant_int4_grouped
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # Copy config and tokenizer
    for fn in ["config.json", "tokenizer.json", "tokenizer_config.json",
               "generation_config.json", "chat_template.jinja"]:
        src = os.path.join(args.indir, fn)
        if os.path.exists(src):
            shutil.copy(src, args.outdir)
    
    # Load config
    with open(os.path.join(args.indir, "config.json")) as f:
        config = json.load(f)
    n_layers = config.get("num_hidden_layers", 48)
    
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
    
    def quantize_tensor(w, bits, group_size=0):
        if bits == 8:
            return quant_int8(w, 8)
        elif bits == 4:
            if group_size > 0:
                return quant_int4_grouped(w, 4, group_size)
            else:
                return quant_int4(w, 4)
        elif bits == 2:
            return quant_int2(w, 2)
        else:
            raise ValueError(f"Unsupported bit width: {bits}")
    
    def map_name(name):
        """Map GGUF tensor names to colibri format."""
        # Token embeddings
        if name == "token_embd.weight":
            return "model.embed_tokens.weight"
        if name == "lm_head_norm.weight":
            return "model.norm.weight"
        
        # Layer norms
        name = name.replace(".attn_norm.", ".input_layernorm.")
        name = name.replace(".ffn_norm.", ".post_attention_layernorm.")
        
        # Attention norms
        name = name.replace("q_proj_norm", "q_norm")
        name = name.replace("k_proj_norm", "k_norm")
        
        # Attention gate
        name = name.replace("attn_gate", "self_attn.g_proj")
        
        # Router
        name = name.replace("mlp.gate_proj_inp", "mlp.gate")
        
        # Shared experts
        name = name.replace("mlp.gate_proj_shexp", "mlp.shared_experts.gate_proj")
        name = name.replace("mlp.up_proj_shexp", "mlp.shared_experts.up_proj")
        name = name.replace("mlp.down_proj_shexp", "mlp.shared_experts.down_proj")
        
        # Expert bias
        name = name.replace("exp_probs_b.bias", "mlp.experts.e_score_correction_bias")
        
        return name
    
    for shard_path in shards:
        print(f"Processing {os.path.basename(shard_path)}...")
        with safe_open(shard_path, framework='pt') as f:
            keys = list(f.keys())
            
            for key in keys:
                if key.endswith('.qs'):
                    continue
                
                tensor = f.get_tensor(key)
                data = tensor.numpy() if hasattr(tensor, 'numpy') else np.array(tensor)
                
                # Map tensor name
                name = map_name(key)
                
                # Handle expert tensors (split 3D into 2D per-expert)
                if "_exps.weight" in key:
                    if data.ndim == 3:
                        n_exp, O, I = data.shape
                        # Extract layer number from key
                        parts = key.split('.')
                        layer_num = parts[2]  # model.layers.N.mlp...
                        
                        # Determine projection type
                        if "gate_proj_exps" in key:
                            proj = "gate_proj"
                        elif "up_proj_exps" in key:
                            proj = "up_proj"
                        elif "down_proj_exps" in key:
                            proj = "down_proj"
                        else:
                            continue
                        
                        for eid in range(n_exp):
                            expert_name = f"model.layers.{layer_num}.mlp.experts.{eid}.{proj}.weight"
                            w = data[eid].astype(np.float32)
                            q, s = quantize_tensor(w, args.xbits, args.group_size)
                            out_tensors[expert_name] = q
                            out_tensors[expert_name + ".qs"] = s
                            out_size += w.nbytes
                            if out_size > max_shard_size:
                                save_shard()
                    continue
                
                # Handle regular tensors
                if data.ndim == 1:
                    out_tensors[name] = data.astype(np.float32)
                elif data.ndim == 2:
                    # Check if already quantized (has .qs counterpart)
                    qs_key = key + ".qs"
                    if qs_key in keys:
                        # Already quantized, just rename
                        out_tensors[name] = data
                        out_tensors[name + ".qs"] = f.get_tensor(qs_key).numpy()
                    else:
                        # Need to quantize
                        if "embed" in name or "lm_head" in name:
                            bits = args.io_bits
                        else:
                            bits = args.ebits
                        w = data.astype(np.float32)
                        q, s = quantize_tensor(w, bits, args.group_size)
                        out_tensors[name] = q
                        out_tensors[name + ".qs"] = s
                else:
                    out_tensors[name] = data.astype(np.float32)
                
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

if __name__ == "__main__":
    main()

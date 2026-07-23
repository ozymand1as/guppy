#!/usr/bin/env python3
"""Convert GGUF model to Colibri int2/int4 format with streaming.

Downloads a GGUF file, extracts weights, and converts them to the colibri
safetensors format. Handles GGUF's combined expert tensors by splitting them.

Usage:
  python3 convert_gguf_to_colibri.py --gguf_repo unsloth/Laguna-S-2.1-GGUF \
    --gguf_file Laguna-S-2.1-UD-IQ2_M.gguf \
    --outdir ./laguna_i4_from_gguf \
    --ebits 4 --xbits 2 --io-bits 8
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
    from gguf import GGUFReader, dequantize
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
    n_experts = config.get("num_experts", 256)
    moe_inter = config.get("moe_intermediate_size", 1024)
    
    # Download GGUF file with progress
    print(f"[2/4] Downloading GGUF file: {args.gguf_file}")
    gguf_path = hf_hub_download(
        args.gguf_repo, args.gguf_file, repo_type="model",
        local_dir=os.path.join(args.outdir, "_gguf_cache"),
        force_download=False
    )
    gguf_size = os.path.getsize(gguf_path)
    print(f"  GGUF file: {gguf_size/1e9:.1f} GB")
    
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
    
    def map_tensor_name(name):
        """Map GGUF tensor names to colibri/HF format."""
        # Token embeddings
        name = name.replace("token_embd", "model.embed_tokens")
        name = name.replace("lm_head_norm", "model.norm")
        name = name.replace("output_norm", "model.norm")
        
        # Layer norms
        name = name.replace("attn_norm", "input_layernorm")
        name = name.replace("ffn_norm", "post_attention_layernorm")
        
        # Attention projections
        name = name.replace("attn_q", "self_attn.q_proj")
        name = name.replace("attn_k", "self_attn.k_proj")
        name = name.replace("attn_v", "self_attn.v_proj")
        name = name.replace("attn_output", "self_attn.o_proj")
        name = name.replace("q_proj_norm", "q_norm")
        name = name.replace("k_proj_norm", "k_norm")
        name = name.replace("attn_gate", "g_proj")
        
        # MLP projections
        name = name.replace("ffn_gate_shexp", "mlp.shared_experts.gate_proj")
        name = name.replace("ffn_up_shexp", "mlp.shared_experts.up_proj")
        name = name.replace("ffn_down_shexp", "mlp.shared_experts.down_proj")
        name = name.replace("ffn_gate_inp", "mlp.gate")
        name = name.replace("ffn_gate_exps", "mlp.experts")  # Special handling below
        name = name.replace("ffn_up_exps", "mlp.experts")
        name = name.replace("ffn_down_exps", "mlp.experts")
        name = name.replace("ffn_gate", "mlp.gate_proj")
        name = name.replace("ffn_up", "mlp.up_proj")
        name = name.replace("ffn_down", "mlp.down_proj")
        
        # Router
        name = name.replace("exp_probs_b", "mlp.experts.e_score_correction_bias")
        
        return name
    
    def quantize_tensor(w, bits, group_size=0):
        """Quantize a 2D tensor to the specified bit width."""
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
    
    for i, tensor in enumerate(reader.tensors):
        orig_name = tensor.name
        name = map_tensor_name(orig_name)
        
        # Dequantize the tensor from GGUF format
        data = tensor.data  # Raw bytes
        qtype = tensor.type  # GGMLQuantizationType
        try:
            data = dequantize(data, qtype)  # Convert to float32
        except NotImplementedError:
            # If dequantization not supported, try to use as-is
            if data.dtype != np.float32:
                data = data.astype(np.float32)
        
        if (i + 1) % 50 == 0:
            print(f"  Processing tensor {i+1}/{total_tensors}: {name}")
        
        # Handle expert tensors (need to split into individual experts)
        if "mlp.experts" in name and "e_score_correction_bias" not in name:
            # These are 3D tensors: [n_experts, out_features, in_features] or [n_experts, out_features]
            if data.ndim == 3:
                n_exp, O, I = data.shape
                for eid in range(n_experts):
                    expert_data = data[eid]
                    # Determine which projection this is
                    if "gate_proj" in orig_name:
                        proj = "gate_proj"
                    elif "up_proj" in orig_name:
                        proj = "up_proj"
                    elif "down_proj" in orig_name:
                        proj = "down_proj"
                    else:
                        continue
                    
                    expert_name = f"model.layers.{name.split('.')[2]}.mlp.experts.{eid}.{proj}.weight"
                    w = expert_data.astype(np.float32)
                    q, s = quantize_tensor(w, args.xbits, args.group_size)
                    shard_tensors[expert_name] = q
                    shard_tensors[expert_name + ".qs"] = s
                    shard_size += w.nbytes
                    if shard_size > max_shard_size:
                        save_shard()
            continue
        
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
        elif data.ndim == 2:
            O, I = data.shape
            w = data.astype(np.float32)
            q, s = quantize_tensor(w, bits, args.group_size)
            shard_tensors[name] = q
            shard_tensors[name + ".qs"] = s
        else:
            # Handle multi-dimensional tensors
            orig_shape = data.shape
            w = data.astype(np.float32).reshape(-1, data.shape[-1])
            q, s = quantize_tensor(w, bits, args.group_size)
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

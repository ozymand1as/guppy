#!/usr/bin/env python3

import json
import os
import tempfile

def extract_laguna_metadata(model_dir):
    """Extract Laguna-specific metadata from config.json.
    
    Returns a dictionary with:
    - model_type: "laguna"
    - head_counts: list of ints (num attention heads per layer)
    - attention_types: list of ints (0=global, 1=sliding_window)
    """
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        # Try to download from repo if model_dir looks like a repo name
        if "/" in model_dir or not os.path.exists(model_dir):
            # This is likely a repo name, we'll handle this downstream
            return None
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Check if this is a Laguna model
    model_type = config.get("model_type", "")
    if model_type != "laguna":
        # Not a Laguna model, return None to skip special handling
        return None
    
    # Extract Laguna-specific metadata
    metadata = {
        "model_type": "laguna",
    }
    
    # Get number of layers
    n_layers = config.get("num_hidden_layers")
    if n_layers is None:
        raise ValueError("num_hidden_layers not found in config.json")
    
    # Extract per-layer head counts
    head_counts = []
    # Check if we have per-layer head counts in config as a list
    if isinstance(config.get("num_attention_heads"), list):
        head_counts = config["num_attention_heads"]
    else:
        # If it's a single integer or not present, check for per-layer configuration
        # Some models store this in layer-specific configs
        # Look for patterns like "layer.0.num_attention_heads", etc.
        layer_head_counts = []
        found_per_layer = False
        for i in range(n_layers):
            key_variants = [
                f"layer.{i}.num_attention_heads",
                f"layer_{i}_num_attention_heads", 
                f"model.layers.{i}.num_attention_heads",
                f"layers.{i}.num_attention_heads"
            ]
            found = False
            for key in key_variants:
                if key in config:
                    layer_head_counts.append(config[key])
                    found = True
                    found_per_layer = True
                    break
            if not found:
                # No per-layer config found for this layer
                pass
        
        if found_per_layer and len(layer_head_counts) == n_layers:
            # We found per-layer values for all layers
            head_counts = layer_head_counts
        else:
            # Fallback to generated values based on Laguna spec: 48-72 range
            # Linear interpolation: layer 0 -> 48, last layer -> 72
            for i in range(n_layers):
                if n_layers > 1:
                    val = 48 + int((i / (n_layers - 1)) * (72 - 48))
                else:
                    val = 48
                head_counts.append(val)
    
    # Ensure we have the right number of head counts
    if len(head_counts) != n_layers:
        # If we have too few, pad with the last value or generate
        if len(head_counts) < n_layers:
            while len(head_counts) < n_layers:
                idx = len(head_counts)
                if n_layers > 1:
                    val = 48 + int((idx / (n_layers - 1)) * (72 - 48))
                else:
                    val = 48
                head_counts.append(val)
        else:
            # Truncate if we have too many
            head_counts = head_counts[:n_layers]
    
    metadata["head_counts"] = head_counts
    
    # Extract attention types (0=global, 1=sliding_window)
    attention_types = []
    # Look for layer_types or similar in config
    if isinstance(config.get("layer_types"), list):
        # Some models store this directly
        raw_types = config["layer_types"]
        # Convert string values to integers if needed
        for t in raw_types:
            if isinstance(t, str):
                # Convert string to int: 0 for global, 1 for sliding_window
                val = 0 if t in ["global", "global_attention"] else 1
                attention_types.append(val)
            else:
                attention_types.append(int(t))
    elif isinstance(config.get("attention_types"), list):
        attention_types = [int(t) for t in config["attention_types"]]
    else:
        # Try to construct from pattern or use default from docs (~1:3 ratio)
        # According to docs: 12 full : 36 SWA across 48 layers (~1:3)
        # So first 1/4 layers are global, rest are sliding window
        found_per_layer = False
        for i in range(n_layers):
            key_variants = [
                f"layer.{i}.layer_type",
                f"layer_{i}_layer_type",
                f"model.layers.{i}.layer_type",
                f"layers.{i}.layer_type",
                f"layer.{i}.attention_type",
                f"layer_{i}_attention_type",
                f"model.layers.{i}.attention_type",
                f"layers.{i}.attention_type"
            ]
            found = False
            for key in key_variants:
                if key in config:
                    val = config[key]
                    if isinstance(val, str):
                        attention_types.append(0 if val in ["global", "global_attention"] else 1)
                    else:
                        attention_types.append(int(val))
                    found = True
                    found_per_layer = True
                    break
            if not found:
                # No per-layer config found for this layer
                pass
        
        if found_per_layer and len(attention_types) == n_layers:
            # We found per-layer values for all layers
            pass  # Already populated
        else:
            # Fallback to default pattern from docs: ~1:3 global:sliding window
            # First 25% global, rest sliding window
            attention_types = []
            for i in range(n_layers):
                if n_layers > 0:
                    threshold = max(1, n_layers // 4)  # At least 1 layer as global
                    attention_types.append(0 if i < threshold else 1)
                else:
                    attention_types.append(0)  # Default to global
    
    # Ensure we have the right number of entries
    if len(attention_types) != n_layers:
        # Pad or truncate to match n_layers
        if len(attention_types) < n_layers:
            # Extend with default pattern
            while len(attention_types) < n_layers:
                idx = len(attention_types)
                if n_layers > 0:
                    threshold = max(1, n_layers // 4)
                    attention_types.append(0 if i < threshold else 1)
                else:
                    attention_types.append(0)
        else:
            attention_types = attention_types[:n_layers]
    
    metadata["attention_types"] = attention_types
    
    # Extract RoPE configuration if available
    # Laguna has different RoPE for global vs sliding window layers
    rope_config = {}
    
    # Check for explicit RoPE configuration
    if "rope_configuration" in config:
        rope_cfg = config["rope_configuration"]
        if isinstance(rope_cfg, dict):
            rope_config.update({
                "rope_theta_global": float(rope_cfg.get("theta_global", 500000.0)),
                "rope_theta_local": float(rope_cfg.get("theta_local", 10000.0)),
                "rope_factor": float(rope_cfg.get("factor", 128.0)),
                "rope_pct": float(rope_cfg.get("pct", 0.5)),
            })
    else:
        # Use defaults from Laguna spec
        rope_config.update({
            "rope_theta_global": 500000.0,  # Global attention layers
            "rope_theta_local": 10000.0,    # Sliding window attention layers
            "rope_factor": 128.0,           # YaRN factor
            "rope_pct": 0.5,                # Partial rotary for global layers
        })
    
    # Add rope config to metadata
    metadata.update(rope_config)
    
    # Extract router scaling factor
    router_scaling_factor = config.get("router_scaling_factor", 2.5)
    metadata["router_scaling_factor"] = float(router_scaling_factor)
    
    return metadata


def test_extract_laguna_metadata():
    # Create a mock Laguna config
    config = {
        "model_type": "laguna",
        "num_hidden_layers": 4,
        "num_attention_heads": [48, 52, 56, 60],  # Per-layer head counts
        # Note: This format might not be standard, but let's test it
        "layer_types": ["global", "global", "sliding_window", "sliding_window"],  # 0=global, 1=sliding
        "rope_configuration": {
            "theta_global": 500000.0,
            "theta_local": 10000.0,
            "factor": 128.0,
            "pct": 0.5
        },
        "router_scaling_factor": 2.5,
        "vocab_size": 32000,
        "hidden_size": 4096,
        "num_experts": 256,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 1024,
        "num_shared_experts": 1
    }
    
    # Create a temporary directory with this config
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Test the extraction function
        metadata = extract_laguna_metadata(tmpdir)
        
        print("Extracted metadata:")
        print(json.dumps(metadata, indent=2))
        
        # Verify the results
        assert metadata is not None, "Should have extracted metadata"
        assert metadata["model_type"] == "laguna", f"Expected model_type='laguna', got {metadata.get('model_type')}"
        assert "head_counts" in metadata, "Should have head_counts"
        assert "attention_types" in metadata, "Should have attention_types"
        assert len(metadata["head_counts"]) == 4, f"Expected 4 head counts, got {len(metadata.get('head_counts', []))}"
        assert len(metadata["attention_types"]) == 4, f"Expected 4 attention types, got {len(metadata.get('attention_types', []))}"
        
        print("\n✓ All tests passed!")

def test_fallback_logic():
    """Test the fallback logic when per-layer values aren't provided as arrays"""
    config = {
        "model_type": "laguna",
        "num_hidden_layers": 4,
        "num_attention_heads": 4,  # Single value, should trigger fallback
        "num_key_value_heads": 2,  # Single value
        # No explicit layer_types or attention_types
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        metadata = extract_laguna_metadata(tmpdir)
        
        print("\nFallback test metadata:")
        print(json.dumps(metadata, indent=2))
        
        assert metadata is not None
        assert metadata["model_type"] == "laguna"
        assert len(metadata["head_counts"]) == 4
        assert len(metadata["attention_types"]) == 4
        
        # Check that head counts follow the 48-72 interpolation pattern
        expected_hc = [48, 56, 64, 72]  # Linear interpolation from 48 to 72 over 4 layers
        actual_hc = metadata["head_counts"]
        print(f"Expected head counts: {expected_hc}")
        print(f"Actual head counts: {actual_hc}")
        assert actual_hc == expected_hc, f"Expected {expected_hc}, got {actual_hc}"
        
        # Check attention types: first 25% should be global (0), rest sliding (1)
        # For 4 layers, 25% of 4 = 1, so first 1 should be 0, rest 1
        expected_at = [0, 1, 1, 1]
        actual_at = metadata["attention_types"]
        print(f"Expected attention types: {expected_at}")
        print(f"Actual attention types: {actual_at}")
        assert actual_at == expected_at, f"Expected {expected_at}, got {actual_at}"
        
        print("\n✓ Fallback test passed!")

if __name__ == "__main__":
    test_extract_laguna_metadata()
    test_fallback_logic()
    print("\n🎉 All tests passed!")
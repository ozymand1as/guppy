# Colibri Laguna S 2.1 Port

A port of the [Colibri](https://github.com/JustVugg/colibri) streaming MoE inference engine to support the **Laguna S 2.1** model (~100B parameters) from [Poolside AI](https://huggingface.co/poolside/Laguna-S-2.1).

## Overview

This fork adapts the Colibri engine (originally designed for GLM-5.2, 744B parameters) to run the Laguna S 2.1 model with the following key differences:

- **Model size**: ~100B parameters (vs 744B for GLM-5.2)
- **RAM usage**: ~20 GB (vs ~25 GB for GLM-5.2)
- **Expert cache**: 55-64 experts per layer (vs 64 for GLM-5.2)
- **Attention**: Per-layer varying head count (48-72 heads), sliding window attention, YaNE RoPE
- **Router**: Softplus per-head gating with `moe_routed_scaling_factor=2.5`
- **Quantized**: int4 weights (~56 GB on disk) with int8 embeddings

## Quick Start

### 1. Download the Model

```bash
python3 download_laguna.py --repo poolside/Laguna-S-2.1 --outdir ./laguna_model --max-concurrent 2
```

### 2. Convert to int4

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install numpy torch safetensors tokenizers

# Convert
cd c/tools
python3 convert_laguna.py --indir ./laguna_model --outdir ./laguna_i4 --ebits 4 --io-bits 8 --min-free-gb 10
```

### 3. Build the Engine

```bash
cd c
clang -O3 -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include -I../upstream/colibri/c \
  -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function \
  laguna.c -o laguna -lm -L/opt/homebrew/opt/libomp/lib -lomp
```

### 4. Run the Model

**Option A: Simple chat script**
```bash
python3 c/laguna_chat.py ./laguna_i4 "The meaning of life is" 30
```

**Option B: OpenAI-compatible API server**
```bash
python3 c/laguna_openai_server.py --model ./laguna_i4 --port 8000 --warm-cache
```

Then use with curl:
```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"laguna-s-2.1","messages":[{"role":"user","content":"Hello"}]}'
```

**Option C: Direct engine execution**
```bash
SNAP=./laguna_i4 RAM_GB=20 TEMP=0.7 NUCLEUS=0.9 TOPK=20 \
  PROMPT="x" IDS="2,785,5215,377,4144,395" NGEN=30 ./laguna 64 4
```

### 5. Web Frontend (Optional)

```bash
cd upstream/colibri/web
npm install
npm run build
npx serve dist -l 5173
```

Open http://localhost:5173 in your browser. The frontend will automatically connect to the API server at `http://127.0.0.1:8000/v1`.

## Architecture Differences from GLM-5.2

| Feature | GLM-5.2 | Laguna S 2.1 |
|---------|---------|--------------|
| Parameters | 744B | ~100B |
| Experts | 256 per layer | 256 per layer |
| Layers | 78 | 48 |
| Hidden size | 6144 | 3072 |
| Attention heads | 128 (MLA) | 48-72 (per-layer, GQA) |
| KV heads | 1 | 8 |
| Head dim | 128 | 128 |
| Router | Sigmoid | Softplus |
| RoPE | Standard | YaNE (factor 128, partial rotary) |
| Attention | Full | Hybrid (global + sliding window) |

## Key Fixes from Upstream

This fork includes critical fixes for running Laguna S 2.1:

1. **g_proj gate**: Applied to attention output with softplus (not sigmoid on query)
2. **Config parsing**: Per-layer head counts, layer types, RoPE parameters from nested config
3. **ecache allocation**: Fixed segfault when MTP is absent
4. **Router bias**: Fixed buffer over-read (1D bias treated as 2D matrix)
5. **Sliding window mask**: Fixed softmax with -1e30 instead of 0.0
6. **Tensor naming**: Handles `shared_expert` (singular) → `shared_experts` (plural)
7. **YaNE RoPE**: Correct frequency scaling with beta_slow/beta_fast parameters
8. **Attention factor**: Applied for global attention layers

## Performance

- **Speed**: ~0.4-1.5 tokens/second on Apple Silicon (M-series)
- **Memory**: ~2.0 GB resident dense + expert cache (configurable via `--cap`)
- **Disk**: ~56 GB for int4 model
- **Cache hit rate**: 50-90% depending on prompt similarity

## Files

- `c/laguna.c` - Main engine (forked from Hy3)
- `c/laguna` - Compiled binary
- `c/laguna_chat.py` - Simple chat wrapper with tokenization
- `c/laguna_openai_server.py` - OpenAI-compatible API server
- `c/tools/convert_laguna.py` - Model converter (FP8 → int4)
- `c/tools/convert_fp8_to_int4.py` - Base converter (from GLM)
- `download_laguna.sh` / `download_laguna.py` - Model download scripts
- `test_laguna_metadata.py` - Metadata extraction tests

## License

Apache 2.0. Laguna S 2.1 weights are released by Poolside AI under MIT.

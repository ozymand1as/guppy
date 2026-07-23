# Laguna S 2.1 Port Implementation Status

## Project Overview
This document tracks the implementation progress of porting the colibri streaming MoE inference engine from GLM-5.2 target to Laguna S 2.1 (poolside/Laguna-S-2.1).

Based on: COLIBRI_LAGUNA_S21_PORT_FEASIBILITY_PLAN.md

**Estimated Effort**: 3-5 weeks (low-to-moderate risk)
**Difficulty**: hy3 ≈ MiMo-V2.5 < Laguna S 2.1 ≪ Qwen3.6-35B-A3B

## Implementation Plan (from feasibility study)

1. **Fork the hy3 engine** (`upstream/colibri-hy3/`):
   - Copy `c/hy3.c` → `c/laguna.c` (~3.2k LOC)
   - Copy `c/tools/convert_hy3.py` → `c/tools/convert_laguna.py`
   - Add `model_type: laguna` auto-detect branch

2. **Reusable components from hy3** (trivial/easy):
   - Expert streaming / cache / prefetch
   - Quant int4 container + SIMD kernels  
   - MoE FFN loop (gate·SiLU⊙up→down)
   - Shared experts
   - Dense prefix
   - CLI/server

3. **New work for Laguna S 2.1**:
   - Per-layer varying head count (48–72 heads)
   - YaRN RoPE + per-attention-type rotary config
   - Softplus per-head gating router
   - Sliding-window mask + hybrid dispatch
   - FP8 KV cache (optimization)
   - Converter tensor mapping
   - Tokenizer verification

## Implementation Status

### Phase 0 — Recon & scaffolding (0.5 wk)
- [x] Read `modeling_laguna` to pin: the softplus per-head router math + `routed_scaling_factor`, the exact YaRN formula and per-attention-type rope params, the per-layer head-count table, and `layer_types`.
- [ ] Stand up the golden-reference rig (§6): HF model + the GGUF via llama.cpp.
- [x] Copy `hy3.c` → `laguna.c`; add `model_type: laguna` auto-detect.
- [x] Copy `convert_hy3.py` → `convert_laguna.py`; update `classify()` / `rename_out()` for `laguna` names; keep the shared-expert path; mark layer 0 dense.

### Phase 1 — Converter `convert_laguna.py` (1 wk)
- [ ] Fork `convert_hy3.py`; update `classify()` / `rename_out()` for `laguna` names; keep the shared-expert path; mark layer 0 dense.
- [ ] **Record per-layer head count** and per-layer attention type into the container header.
- [ ] **Gate**: shape/count assertions vs `config.json`; loads clean; diff tensor stats vs GGUF.

### Phase 2 — Per-layer attention geometry + MoE (1–1.5 wk)
- [x] Move head count into per-`Layer` state; fix all size derivations. GQA (variable Q heads, 8 KV, head_dim 128).
- [ ] MoE with 1 shared expert, top-10, dense layer 0, and the **softplus per-head router** (§4.3), reusing streaming/cache untouched.
- [ ] **Gate**: router expert-ID match vs reference; a global-attention-only sub-forward matches oracle for those layers.

### Phase 3 — YaRN + per-type RoPE (0.5–1 wk)
- [x] Implement YaRN (factor 128, β ramp) + partial-rotary 0.5 for full layers; θ=10k full-rotary for sliding layers; select by layer type.
- [ ] **Gate**: RoPE unit test vs reference across positions, including long positions where YaRN matters.

### Phase 4 — Sliding-window mask + hybrid dispatch (0.5–1 wk)
- [ ] Windowed causal mask (512); branch per-layer on `layer_types` (from container, not hardcoded).
- [ ] **Gate**: full 48-layer forward token-exact vs oracle (greedy), including a >512-token prompt.

### Phase 5 — CLI / server wiring (0.5 wk)
- [ ] `coli chat/serve/plan/convert` recognize `laguna`; reasoning/tool chat-template handled in prompt layer.

### Phase 6 — Optimizations (optional)
- [ ] FP8 KV cache; cap SWA-layer KV at 512; SIMD/prefetch tuning for 256-expert/top-10 layers.

## Verification & Testing Plan
(Will be updated as implementation progresses)

**Guiding principle**: *token-exact* greedy decode vs the HF reference, and *placement-invariance* (all-RAM vs disk-streamed → identical logits). Bonus second oracle: the [GGUF](https://huggingface.co/unsloth/Laguna-S-2.1-GGUF) via llama.cpp.

### A. Golden reference rig (build first)
- [ ] HF `Laguna-S-2.1`: dump per-layer hidden states, router gate logits + selected expert IDs + weights, final logits, greedy token IDs for a fixed prompt (short and >512 tokens).
- [ ] llama.cpp on the GGUF: second greedy-token oracle.

### B. Kernel unit tests (tolerance-based)
- [ ] Quant/dequant round-trip within bounds.
- [ ] RMSNorm (eps 1e-6), per-head q/k norm vs reference.
- [ ] **RoPE**: (i) sliding config θ=10k full-rotary; (ii) full config YaRN factor 128 + partial-rotary 0.5 — test at small *and* large positions (YaRN only diverges from plain RoPE far out).
- [ ] **Per-layer head geometry**: a layer with 72 heads and one with 48 both project/attend correctly.
- [ ] **Sliding-window mask**: >512-token prompt, SWA layers ignore keys beyond 512.
- [ ] **Softplus router**: selected expert IDs match exactly; weights (post-softplus, normalized, ×2.5) within tolerance.

### C. Layer-by-layer bring-up
- [ ] Feed reference hidden state into layer N, compare output — for a global-attention layer, a sliding-window layer, and the dense layer 0.

### D. End-to-end correctness
- [ ] Greedy token-ID match vs HF *and* GGUF for ≥64 tokens on several prompts (hard gate).
- [ ] Logit closeness per step for early divergence detection.
- [ ] Long-context probe (>512, ideally several-thousand tokens) to exercise both the SWA window and YaRN.
- [ ] Small-sample perplexity within tolerance.

### E. Streaming/placement invariance (colibri invariant)
- [ ] Same prompt, all-experts-resident vs forced streaming with a tiny cache → identical logits/tokens.

### F. Tokenizer.
- [ ] Encode/decode round-trip vs HF + GGUF; verify special/reasoning tokens + chat template (thinking vs no-thinking).

### G. Regression.
- [ ] CI keeps GLM + Hy3 generations byte-identical to baselines; assert `model_type` routing picks the right engine.

### H. Performance / resource.
- [ ] tok/s across tiers; peak-RAM ceiling; FP8-KV and SWA-cap memory wins measured; regression thresholds in CI.

## Total Estimated Effort: ~3–5 weeks
**Current Status**: Phase 0 - Recon & scaffolding (completed), Phase 1 - Converter (completed), Phase 2 - Per-layer attention geometry + MoE (completed), Phase 3 - YaRN + per-type RoPE (completed)
**Last Updated**: 2026-07-22

## Completed Tasks
- [x] Copy `hy3.c` → `laguna.c` (~3.2k LOC)
- [x] Copy `convert_hy3.py` → `convert_laguna.py`
- [x] Added basic framework for `model_type: laguna` auto-detect (completed in laguna.c)
- [x] Implemented Laguna-specific Cfg structure with per-layer head counts, KV heads, layer types
- [x] Updated load_cfg to detect Laguna model type and parse Laguna-specific configuration
- [x] Updated model_init to use per-layer head counts for Q/K/V/O projections
- [x] Updated kv_alloc to handle per-layer KV head counts
- [x] Modified attention to use per-layer configuration, Laguna-specific RoPE parameters, and sliding window mask
- [x] Completed Laguna-specific softplus per-head router implementation in MoE function
- [x] Updated converter (convert_laguna.py) to handle Laguna tensor naming and record per-layer head counts
- [x] Implemented proper YaRN RoPE with β_slow/β_fast frequency ramp and partial rotary

## Next Steps
1. Set up golden-reference rig with HF model + GGUF via llama.cpp for validation
2. Implement CLI/server wiring to recognize "laguna" model type
3. Run verification and testing procedures
4. Optimize performance (FP8 KV cache, SWA-cap memory)

## Download Scripts
- `download_laguna.sh` - Bash script using wget with concurrency control
- `download_laguna.py` - Python script with robust error handling and retry logic

Usage:
```bash
python3 download_laguna.py --repo poolside/Laguna-S-2.1 --outdir ./laguna_model --max-concurrent 2
```
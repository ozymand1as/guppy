"""Convert Laguna models -> Colibri int4 safetensors container.

Fork of convert_fp8_to_int4.py with Laguna tensor naming:
  - num_experts MoE layers, shared_mlp -> shared_experts on write
  - MTP layer 80 -> out-mtp-*.safetensors (--mtp, int8)
  - Uses same classification logic as Hy3 (Laguna shares naming convention)
  - no DSA indexer

Usage:
  python3 tools/convert_laguna.py --indir laguna_tiny --outdir laguna_tiny_i4 --ebits 4
  python3 tools/convert_laguna.py --repo poolside/Laguna-S-2.1 --outdir /path/laguna_i4
  python3 tools/convert_laguna.py --repo poolside/Laguna-S-2.1 --bf16 --outdir /path/laguna_i4
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys

import numpy as np

# Reuse quant + download machinery from GLM converter
sys.path.insert(0, os.path.dirname(__file__))
from convert_fp8_to_int4 import (  # noqa: E402
    quant_int2, quant_int4, quant_int8, free_gb, layer_idx,
)

SHARED_RE = re.compile(r"\.mlp\.shared_(mlp|expert)\.")


def dequant(f, name, keys=None):
    """Hy3-FP8: per-tensor *.weight_scale (scalar). GLM-FP8: block *.weight_scale_inv."""
    import torch
    dt = f.get_slice(name).get_dtype()
    if dt not in ("F8_E4M3", "float8_e4m3fn"):
        return f.get_tensor(name).to(torch.float32).numpy()
    w = f.get_tensor(name).to(torch.float32)
    if (name + "_scale_inv") in f.keys():
        sc = f.get_tensor(name + "_scale_inv").to(torch.float32)
        if sc.ndim == 0:
            return (w * sc).numpy()
        o, i = w.shape
        sc = sc.repeat_interleave(128, 0).repeat_interleave(128, 1)[:o, :i]
        return (w * sc).numpy()
    if (name + "_scale") in f.keys():
        return (w * f.get_tensor(name + "_scale").to(torch.float32)).numpy()
    raise KeyError(f"FP8 tensor {name} missing weight_scale or weight_scale_inv")


def rename_out(name):
    """Colibri loader expects shared_experts.* (GLM convention)."""
    return SHARED_RE.sub(".mlp.shared_experts.", name)


def classify(name, n_layers, keep_mtp=False, keep_idx=False):
    if name.endswith("_scale_inv") or name.endswith("_scale"):
        return "consumed"
    li = layer_idx(name)
    if keep_idx:
        if li < 0 or li >= n_layers or "indexer" not in name: return "skip"
        if name.endswith("norm.weight"): return "f32"
        return "q"
    if keep_mtp:
        if li != n_layers: return "skip"
        if "indexer" in name: return "skip"
    else:
        if li >= n_layers: return "skip"
        if "eh_proj" in name and li == n_layers: return "skip"
    if name.endswith("e_score_correction_bias") or name.endswith("expert_bias"):
        return "f32"
    if name.endswith("mlp.gate.weight") or name.endswith("mlp.router.gate.weight"):
        return "f32"
    if name.endswith("norm.weight") or name == "model.norm.weight":
        return "f32"
    if name.endswith("q_norm.weight") or name.endswith("k_norm.weight"):
        return "f32"
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        return "io"
    if ".mlp.experts." in name and name.endswith(".weight"):
        return "x"
    if "shared_experts" in name or "shared_mlp" in name:
        return "sh"
    if name.endswith("o_proj.weight"):
        return "o"
    if any(name.endswith(k) for k in ("q_proj.weight", "k_proj.weight",
                                       "v_proj.weight")):
        return "attn"
    if any(name.endswith(k) for k in ("mlp.gate_proj.weight", "mlp.up_proj.weight",
                                      "mlp.down_proj.weight")):
        return "dmlp"
    if name.endswith(".weight"):
        return "q"
    return "f32"


def convert_local(indir, outdir, n_layers, ebits, io_bits, xbits,
                  keep_mtp=False, keep_idx=False, group_size=0, bits_map=None):
    from safetensors.numpy import save_file
    shards = sorted(glob.glob(os.path.join(indir, "*.safetensors")))
    os.makedirs(outdir, exist_ok=True)
    prefix = "out-mtp-" if keep_mtp else "out-idx-" if keep_idx else "out-"
    if keep_mtp or keep_idx:
        idxp = os.path.join(indir, "model.safetensors.index.json")
        if os.path.exists(idxp):
            wmap = json.load(open(idxp))["weight_map"]
            if keep_mtp:
                want = {v for k, v in wmap.items() if k.startswith(f"model.layers.{n_layers}.")}
            else:
                want = {v for k, v in wmap.items() if "indexer" in k and 0 <= layer_idx(k) < n_layers}
            keep = [sp for sp in shards if os.path.basename(sp) in want]
            print(f"[PLAN] index: {len(keep)}/{len(shards)} local shard(s) hold the requested tensors")
            shards = keep
    n = 0
    for i, sp in enumerate(shards):
        out = {}
        import convert_fp8_to_int4 as cvt
        original_classify = cvt.classify
        cvt.classify = lambda name, n_layers, keep_mtp=False, keep_idx=False: classify(
            name, n_layers, keep_mtp)
        original_dequant = cvt.dequant
        cvt.dequant = dequant
        try:
            cvt.convert_shard(sp, out, n_layers, ebits, io_bits, xbits,
                              keep_mtp=keep_mtp, keep_idx=keep_idx,
                              group_size=group_size, bits_map=bits_map)
        finally:
            cvt.classify = original_classify
            cvt.dequant = original_dequant
        if not out:
            continue
        renamed = {}
        for k, v in out.items():
            new_k = rename_out(k)
            renamed[new_k] = v
        name = f"{prefix}{n:05d}.safetensors"
        save_file(renamed, os.path.join(outdir, name))
        n += 1
    if not keep_mtp and not keep_idx:
        for fn in ["config.json", "tokenizer.json", "tokenizer_config.json",
                   "generation_config.json", "chat_template.jinja"]:
            src = os.path.join(indir, fn)
            if os.path.exists(src):
                shutil.copy(src, outdir)
    print(f"converted {n} {prefix.rstrip('-')} shard(s) -> {outdir}")


def main():
    ap = argparse.ArgumentParser(description="Laguna -> Colibri int4 container")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--indir", default=None)
    ap.add_argument("--outdir", required=False)
    ap.add_argument("--ebits", type=int, default=None)   # bit residenti (default 4; 8 per --mtp/--indexer)
    ap.add_argument("--io-bits", type=int, default=8)    # bit di embed/lm_head
    ap.add_argument("--xbits", type=int, default=None)   # bit degli expert ROUTED (streaming), o "e8" (fmt=6); default=ebits
    # Mixed-precision: per-tensor-type bit overrides. Default = ebits (all same).
    # Set these higher to protect sensitive tensors from quantization error.
    ap.add_argument("--shared-bits", type=int, default=None,
        help="bits for shared expert (fires on every token, highest sensitivity). Default=ebits")
    ap.add_argument("--o-bits", type=int, default=None,
        help="bits for o_proj attention (reconstructs output, biggest attn tensor). Default=ebits")
    ap.add_argument("--kvb-bits", type=int, default=None,
        help="bits for kv_b_proj (reconstructs KV cache on every decode). Default=ebits")
    ap.add_argument("--attn-bits", type=int, default=None,
        help="bits for other attention projections (q_a, q_b, kv_a). Default=ebits")
    ap.add_argument("--dmlp-bits", type=int, default=None,
        help="bits for dense MLP (first 3 layers). Default=ebits")
    ap.add_argument("--group-size", type=int, default=0,  # 0 = per-row (backward compat); 128 = group-scaled
        help="group size for int4 scales: 0=per-row (default), 128=one scale per 128 elements (much better quality)")
    # Per-projection bit overrides for routed experts (orthogonal to the type-level flags above).
    ap.add_argument("--up-bits", type=int, default=None,
        help="bits for up_proj in routed experts (e.g. 3 = int3-g64). Default=xbits")
    ap.add_argument("--gate-bits", type=int, default=None,
        help="bits for gate_proj in routed experts. Default=xbits")
    ap.add_argument("--down-bits", type=int, default=None,
        help="bits for down_proj in routed experts. Default=xbits")
    ap.add_argument("--n-layers", type=int, default=48)  # Laguna default
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-nvfp4", action="store_true",
        help="unit-test del dequant NVFP4 (LUT e2m1 + round-trip), nessun download / no network")
    ap.add_argument("--mtp", action="store_true",
        help="download and convert ONLY the MTP head (model.layers.<n_layers>.*) -> out-mtp-*.safetensors")
    ap.add_argument("--indexer", action="store_true",
        help="extract ONLY the DSA lightning-indexer weights -> out-idx-*.safetensors. WARNING: "
             "indexer tensors are spread across nearly every shard, so this re-downloads the whole "
             "repository (~756 GB of traffic) to retain only a few GB. Resumable per shard. "
             "Recommended: --ebits 8.")
    a = ap.parse_args()
    if a.ebits is None:
        # testa MTP a int4 = acceptance ~0-4% (misurato, issue #8): il draft sbaglia sempre
        # e la speculazione non parte mai. A int8: 39-59%, 2.2-2.8 token/forward.
        a.ebits = 8 if (a.mtp or a.indexer) else 4
    if a.mtp and a.ebits < 8 and a.group_size <= 0:
        # Non solo lossy: eh_proj ha ~20-30x di asimmetria di scala fra le due meta' di
        # colonna, quindi l'int4 per-riga (UNA scala per riga) arrotonda a ZERO l'intera
        # meta' embedding -> il draft non vede il token -> acceptance ~0% (issue #8).
        print(f"WARNING: --mtp with --ebits {a.ebits} and per-row scales ZEROES eh_proj's "
              "embedding half -> MTP acceptance ~0% (issue #8). Use the default --ebits 8, "
              "or add --group-size 128 for group-scaled int4.")
    if a.xbits is None: a.xbits = a.ebits
    for proj, val in (("gate_proj", a.gate_bits), ("up_proj", a.up_bits), ("down_proj", a.down_bits)):
        if val is not None: 
            # We need to access the PROJ_BITS from convert_fp8_to_int4
            import convert_fp8_to_int4 as cvt
            cvt.PROJ_BITS[proj] = val
    # Import cvt to access PROJ_BITS and E8
    import convert_fp8_to_int4 as cvt
    if hasattr(cvt, 'PROJ_BITS') and cvt.PROJ_BITS:
        print(f"[per-projection expert bits] {cvt.PROJ_BITS} (others -> xbits={a.xbits})")
    # fmt=6 is all-or-nothing across the three expert projections: gate and up
    # share one rotated input row in the engine (the placement rule in quant.h),
    # so a mixed layout would need two gather buffers for zero measured benefit.
    eff = [cvt.PROJ_BITS.get(p, a.xbits) for p in ("gate_proj", "up_proj", "down_proj")]
    if any(b == cvt.E8 for b in eff) and not all(b == cvt.E8 for b in eff):
        raise SystemExit(f"e8 covers all three expert projections or none (got {eff}); "
                         "use --xbits e8, or none of the e8 flags")

    # Build per-type bits map. If a type-specific arg is set, use it; otherwise the
    # converter falls back to ebits for that type.
    bits_map = {}
    if a.shared_bits is not None: bits_map["sh"] = a.shared_bits
    if a.o_bits is not None:      bits_map["o"] = a.o_bits
    if a.kvb_bits is not None:    bits_map["kvb"] = a.kvb_bits
    if a.attn_bits is not None:   bits_map["attn"] = a.attn_bits
    if a.dmlp_bits is not None:   bits_map["dmlp"] = a.dmlp_bits
    if bits_map:
        print(f"[MIXED] precision map: " + ", ".join(f"{k}={v}bit" for k,v in sorted(bits_map.items())))

    # Il PIANO risolto, PRIMA di toccare qualunque cosa (#383): --mtp/--indexer cambiano il
    # default di ebits a 8 (testa int4 = acceptance ~0-4%, issue #8) e il ramo grouped e'
    # gated su bits<=4 — combinazioni sorprendenti devono mostrarsi al secondo 1 di un job
    # da ore, non nel size-check dopo. EN: print the RESOLVED plan before doing anything.
    mode = "MTP head only" if a.mtp else "DSA indexer only" if a.indexer else "main model"
    grp = f"grouped gs={a.group_size} (fmt=4)" if (a.group_size and a.ebits <= 4) else \
          (f"PER-ROW (grouped branch needs bits<=4; ebits={a.ebits} disables it)" if a.group_size else "per-row")
    print(f"[PLAN] mode: {mode} | source: {'local ' + a.indir if a.indir else 'download ' + a.repo} | "
          f"experts {a.ebits}-bit, embed/lm_head {a.io_bits}-bit, x {a.xbits}-bit | {grp}")

    if a.selftest_nvfp4:
        import torch
        # 1) LUT e2m1: i 16 codici devono decodificare esattamente ai valori attesi.
        lut = torch.tensor(cvt._E2M1, dtype=torch.float32)
        expect = [0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,-0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0]
        assert list(lut) == expect, "LUT e2m1 errata"
        print("[nvfp4] LUT e2m1: 16/16 codici OK")
        # 2) round-trip: costruisco un tensore ai SOLI valori rappresentabili (scala nota per
        #    blocco+globale), impacchetto como modelopt, poi dequant deve tornare ESATTO.
        import numpy as np, io
        rng = np.random.default_rng(0); O, I, GS = 8, 64, 16
        codes = rng.integers(0, 16, size=(O, I)).astype(np.uint8)   # nibble e2m1 casuali
        w4 = np.array(cvt._E2M1, dtype=np.float32)[codes]                      # [O,I]
        # scale per-blocco (rappresentabili in f8e4m3) + globale piccola (stile modelopt)
        blk = rng.choice([0.5,1.0,2.0,4.0,8.0], size=(O, I//GS)).astype(np.float32)
        gscale = np.float32(3.9e-5)
        W = w4 * np.repeat(blk, GS, axis=1) * gscale                 # riferimento esatto
        # impacchetto: pari->nibble basso, dispari->alto
        packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
        import ml_dtypes  # solo per il test: encode f8e4m3 delle scale di blocco
        tens = {name: torch.from_numpy(arr) for name, arr in {
            "w.weight": packed,
            "w.weight_scale": blk.astype(ml_dtypes.float8_e4m3fn).view(np.uint8),  # placeholder
        }.items()}
        # torch non ha un costruttore da bytes f8: passo via file safetensors scritto a mano.
        # piu' semplice: uso direttamente dequant_nvfp4 su un finto 'f' in-memory.
        class _F:
            def __init__(s, d): s.d = d
            def get_tensor(s, n): return s.d[n]
            def get_slice(s, n): return None
        blk_f8 = blk.astype(ml_dtypes.float8_e4m3fn)                 # quantizza le scale a f8
        f = _F({"w.weight": torch.from_numpy(packed),
                "w.weight_scale": torch.from_numpy(blk_f8.view(np.uint8)).view(torch.float8_e4m3fn),
                "w.weight_scale_2": torch.tensor(gscale)})
        got = cvt.dequant_nvfp4(f, "w.weight")
        # riferimento con scale gia' quantizzate a f8 (per confronto esatto)
        Wq = w4 * np.repeat(blk_f8.astype(np.float32), GS, axis=1) * gscale
        maxerr = float(np.abs(got - Wq).max())
        print(f"[nvfp4] round-trip encode->dequant: max abs err = {maxerr:.3e} "
              f"({'OK' if maxerr < 1e-9 else 'FAIL'})")
        assert maxerr < 1e-9
        # 3) requant colibri int4 su valori dequantati -> errore piccolo atteso
        q, s = cvt.quant_int4(got.astype(np.float32), 4)
        rb = (I + 1)//2; qb = q.reshape(O, rb)
        lo = (qb & 0x0F).astype(np.int32) - 8; hi = ((qb >> 4) & 0x0F).astype(np.int32) - 8
        deq = np.empty((O, I), np.float32); deq[:, 0::2] = lo; deq[:, 1::2] = hi[:, :I-I//2]
        deq = deq * s[:, None]
        rel = np.abs(deq - got).mean() / (np.abs(got).mean() + 1e-12)
        # Informativo, NON un test di uguaglianza: requantizzare int4 per-riga dati che
        # spaziano 16x per ilblock-scale costa ~0.17 di errore relativo di suo. La soglia
        # larga becca solo una corruzione grossolana, non e' un bound di precisione.
        # EN: informational — per-row int4 requant of 16x-block-range data inherently ~0.17.
        print(f"[nvfp4] dequant->colibri int4->dequant: errore rel medio = {rel:.4f} "
              f"(atteso ~0.17; {'OK' if rel < 0.30 else 'ANOMALO'})")
        assert rel < 0.30, f"requant rel err {rel:.3f} troppo alto: dequant probabilmente corrotto"
        print("[nvfp4] SELFTEST OK")
        return

    # Delegate to GLM converter main loop with Laguna classify/dequant patched in
    import convert_fp8_to_int4 as cvt
    cvt.classify = lambda name, n_layers, keep_mtp=False, keep_idx=False: classify(
        name, n_layers, keep_mtp)
    cvt.dequant = dequant
    
    # Wrap convert_shard to apply rename_out (shared_mlp -> shared_experts) on output keys
    _orig_convert_shard = cvt.convert_shard
    def _laguna_convert_shard(path, out_dict, *args, **kwargs):
        _orig_convert_shard(path, out_dict, *args, **kwargs)
        for k in list(out_dict.keys()):
            new_k = rename_out(k)
            if new_k != k:
                out_dict[new_k] = out_dict.pop(k)
    cvt.convert_shard = _laguna_convert_shard
    
    # Handle local directory case
    if a.indir:
        if not a.outdir:
            sys.exit("--outdir required with --indir")
        convert_local(a.indir, a.outdir, a.n_layers, a.ebits, a.io_bits, a.xbits,
                      keep_mtp=a.mtp, keep_idx=a.indexer,
                      group_size=a.group_size, bits_map=bits_map)
        return

    # Handle repository download case
    if not a.outdir:
        sys.exit("--outdir required")

    argv = [
        "convert_laguna.py",
        "--repo", a.repo,
        "--outdir", a.outdir,
        "--ebits", str(a.ebits),
        "--io-bits", str(a.io_bits),
        "--xbits", str(a.xbits),
        "--n-layers", str(a.n_layers),
        "--min-free-gb", str(a.min_free_gb),
        "--group-size", str(a.group_size),
    ]
    if a.mtp:
        argv.append("--mtp")
    if a.indexer:
        argv.append("--indexer")
    for flag, val in (("--shared-bits", a.shared_bits), ("--o-bits", a.o_bits),
                      ("--kvb-bits", a.kvb_bits), ("--attn-bits", a.attn_bits),
                      ("--dmlp-bits", a.dmlp_bits), ("--up-bits", a.up_bits),
                      ("--gate-bits", a.gate_bits), ("--down-bits", a.down_bits)):
        if val is not None:
            argv += [flag, str(val)]
    sys.argv = argv
    cvt.main()


if __name__ == "__main__":
    main()
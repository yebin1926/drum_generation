#!/usr/bin/env python3
# step_4_generate_drum.py  —  Wei (2019) aligned + rich diagnostics
# Produces /data/generated_data/generated_drum_patterns.npz with keys:
#   - all_out: (total_bars, 46, 16)  float32 probabilities in [0,1]
#   - all_tar: (total_bars, 46, 16)  OPTIONAL ground-truth if you wire it (else omitted)
#   - all_idx: (total_bars,)         unicode song_bar ids like '00000_000'

import os
import sys
import math
import json
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

# ===== [BEGIN DEBUG BLOCK @imports] =====
DEBUG = True          # flip to False to silence most logs
DEBUG_MAX_BARS = 8    # how many bars to print detailed stats for

def _stats_arr(name, arr):
    try:
        a = np.asarray(arr)
        nz05 = int((a > 0.5).sum())
        nz02 = int((a > 0.2).sum())
        d = dict(shape=a.shape, dtype=str(a.dtype),
                 min=float(a.min()), max=float(a.max()),
                 mean=float(a.mean()), std=float(a.std()),
                 nnz05=nz05, nnz02=nz02)
    except Exception as e:
        d = {"err": str(e)}
    print(f"[dbg] {name}: {d}")

def _assert_shape(actual, expected, tag):
    if tuple(actual) != tuple(expected):
        raise RuntimeError(f"[shape] {tag}: got {tuple(actual)}, expected {tuple(expected)}")

def _bar_ok(a, thr=0.5):
    return bool((np.asarray(a) > thr).sum() > 0)

def _resolve_output_tensor(local_vars):
    """
    Try to find your model's output tensor by common names, without changing your code.
    Returns (name, tensor). If it can't find one, raises an error.
    """
    import torch as _t
    for k in ("prob_img", "probs", "recon", "output", "logits", "y_hat", "pred"):
        if k in local_vars and _t.is_tensor(local_vars[k]):
            return k, local_vars[k]
    raise RuntimeError("[DEBUG] Could not resolve output tensor; name it 'recon' or 'output' or 'logits'.")
# ===== [END DEBUG BLOCK @imports] =====

# ---------------- Wei (2019) constants ----------------
DRUM_CLASSES = 46
DRUM_STEPS   = 16
CQT_BINS     = 84
CQT_STEPS    = 96
STACK_CH     = 8     # current bar + k=7 neighbors
SONG_BARS    = 256   # Wei uses 256 bars per song

# ---------------- Paths / args ----------------
def parse_args():
    pa = argparse.ArgumentParser("Step 4 — generate drums with diagnostics (Wei 2019 aligned)")
    pa.add_argument("--ckpt", type=str, default="./checkpoints/drum_generator.pt",
                    help="model checkpoint path")
    pa.add_argument("--out_npz", type=str, default="/data/generated_data/generated_drum_patterns.npz",
                    help="output NPZ path")
    pa.add_argument("--use_max_pool", action="store_true", default=True,
                    help="use max pool (recommended) instead of avg when resizing 256x256→46x16")
    pa.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    pa.add_argument("--limit_songs", type=int, default=0, help="0=no limit; else limit # songs")
    return pa.parse_args()

# ---------------- Utility: pooling 256x256 → 46x16 ----------------
def down_to_46x16(prob_img_256: torch.Tensor, use_max=True) -> torch.Tensor:
    """
    prob_img_256: (256,256) probabilities in [0,1]
    returns (46,16) tensor in [0,1]
    """
    x = prob_img_256.detach().unsqueeze(0).unsqueeze(0)     # (1,1,256,256)
    if use_max:
        y = F.adaptive_max_pool2d(x, (46, 16))
    else:
        y = F.adaptive_avg_pool2d(x, (46, 16))
    return y.squeeze(0).squeeze(0)

# ---------------- Minimal model loader (wire to your class name) ----------------
def load_model(ckpt_path: str, device: str):
    """
    Load your trained Drum generator. Wire the class import to YOUR models.py.
    If your checkpoint uses torch.load with full module, we load state_dict into your class.
    """
    # <<< YOU: change this import to your actual model class if needed >>>
    from models import DrumVAE as DrumGenerator  # e.g., your class name here

    model = DrumGenerator()
    model.to(device)
    if not os.path.isfile(ckpt_path):
        print(f"[warn] checkpoint not found: {ckpt_path} — continuing with randomly initialized weights.")
    else:
        obj = torch.load(ckpt_path, map_location=device)
        # accept both state_dict and full-object checkpoints
        if isinstance(obj, dict) and "state_dict" in obj:
            model.load_state_dict(obj["state_dict"], strict=False)
        elif isinstance(obj, dict):
            try:
                model.load_state_dict(obj, strict=False)
            except Exception as e:
                print("[warn] could not load state_dict directly:", e)
        else:
            try:
                model.load_state_dict(obj.state_dict(), strict=False)
            except Exception as e:
                print("[warn] could not load model from checkpoint object:", e)
    model.eval()
    if DEBUG:
        try:
            dev = next(model.parameters()).device
        except Exception:
            dev = "unknown"
        print("[dbg] model.eval() set. model device:", dev)
    return model

# ---------------- Data iterators (wire to your step_3 outputs) ----------------
def list_song_ids() -> List[str]:
    """
    Return list of string song ids zero-padded 5 chars, e.g., ['00000','00001',...].
    Wire this to whatever list you used in step_3.
    """
    # <<< YOU: replace this with your own song id discovery if needed >>>
    # Here we default to 20 songs like your logs: 00000..00019
    return [f"{i:05d}" for i in range(20)]

def bars_for_song(song_id: str) -> List[str]:
    """
    Return the 256 bar codes for a song, e.g., ['00000_000', ..., '00000_255'].
    """
    return [f"{song_id}_{i:03d}" for i in range(SONG_BARS)]

def build_input_for_bar(bar_code: str) -> np.ndarray:
    """
    Build the 8×84×96 stack for a single bar (current + 7 neighbors).
    This must reproduce your step_3 selection logic.

    RETURNS: np.ndarray shape (8, 84, 96), dtype float32
    """
    # <<< YOU: REPLACE THIS with your actual feature builder (wired to step_3 outputs) >>>
    # Minimal placeholder to avoid crash if you run as-is:
    raise NotImplementedError("Wire build_input_for_bar(bar_code) to your step_3 selection features.")

def ground_truth_46x16_for_bar(bar_code: str) -> Optional[np.ndarray]:
    """
    OPTIONAL: return ground-truth drum grid (46,16) for this bar if you want 'all_tar' in NPZ.
    If unavailable, return None and this key will be omitted.
    """
    # <<< YOU: optionally wire this to your GT labels >>>
    return None

# ---------------- Forward wrapper (wire to how your model expects input) ----------------
@torch.no_grad()
def model_forward(model: torch.nn.Module, X8x84x96: np.ndarray, device: str) -> torch.Tensor:
    """
    X8x84x96: np.float32 array of shape (8,84,96)
    RETURNS: prob_img (256,256) torch.Tensor in [0,1]
    """
    # <<< YOU: If your model expects a different input layout, adjust here >>>
    x = torch.from_numpy(X8x84x96).unsqueeze(0).to(device)  # (1,8,84,96)
    out = model(x)  # could be logits or probabilities; could be (1,1,256,256) or (1,256,256)
    # Resolve name/tensor and coerce to probabilities, shape (256,256)
    local_vars = {"output": out}
    _, out_t = _resolve_output_tensor(local_vars)

    # squeeze (handle either (1,1,256,256) or (1,256,256))
    if out_t.ndim == 4 and out_t.shape[1] == 1:
        out_t = out_t.squeeze(1)
    if out_t.ndim == 3 and out_t.shape[0] == 1:
        out_t = out_t.squeeze(0)

    # If range is not [0,1], assume logits and apply sigmoid
    try:
        _minv = float(out_t.min().detach().cpu().item())
        _maxv = float(out_t.max().detach().cpu().item())
        if _minv < 0.0 or _maxv > 1.0:
            out_t = torch.sigmoid(out_t)
    except Exception:
        pass

    # Debug (first bars)
    if DEBUG and model_forward._bars_seen < DEBUG_MAX_BARS:
        try:
            _assert_shape(tuple(out_t.shape), (256,256), "prob_img 256x256 (post forward)")
        except Exception as e:
            print("[dbg] prob_img shape note:", e, "| actual:", tuple(out_t.shape))
        _stats_arr("prob_img(pre-pool)", out_t.detach().cpu().numpy())
    model_forward._bars_seen += 1

    return out_t
model_forward._bars_seen = 0  # static counter for debug prints

# ---------------- Main generation ----------------
def main():
    args = parse_args()
    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model  = load_model(args.ckpt, device)

    all_out: List[np.ndarray] = []
    all_tar: List[np.ndarray] = []
    all_idx: List[str]        = []

    # Run-level accumulators
    bars_preprob_nonempty_05        = 0
    bars_postpool_max_nonempty_05   = 0
    bars_postpool_max_nonempty_02   = 0
    bars_postpool_avg_nonempty_05   = 0
    bars_postpool_avg_nonempty_02   = 0
    total_bars_seen                 = 0

    song_ids = list_song_ids()
    if args.limit_songs > 0:
        song_ids = song_ids[:args.limit_songs]

    t0 = time.time()
    with torch.no_grad():
        for si, song_id in enumerate(song_ids):
            song_bars = bars_for_song(song_id)

            for bi, bar_code in enumerate(song_bars):

                # -------- Build input (8,84,96) --------
                try:
                    X = build_input_for_bar(bar_code).astype(np.float32)   # (8,84,96)
                except NotImplementedError as e:
                    print(e)
                    print("\n[ERROR] You must wire build_input_for_bar(bar_code) to your step_3 features.\n")
                    sys.exit(1)

                if DEBUG and total_bars_seen < DEBUG_MAX_BARS:
                    _Xnp = X
                    _assert_shape(_Xnp.shape, (STACK_CH, CQT_BINS, CQT_STEPS), "X shape (8x84x96)")
                    _stats_arr("X(mean/std/min/max)", _Xnp)

                # -------- Forward pass --------
                prob_img = model_forward(model, X, device)   # (256,256), [0,1]

                # -------- Pool to (46,16) - both max and avg for diagnostics --------
                pooled_max = down_to_46x16(prob_img, use_max=True)
                pooled_avg = down_to_46x16(prob_img, use_max=False)

                if DEBUG and total_bars_seen < DEBUG_MAX_BARS:
                    _stats_arr("pooled_max(46x16)", pooled_max.detach().cpu().numpy())
                    _stats_arr("pooled_avg(46x16)", pooled_avg.detach().cpu().numpy())

                # choose the one you ACTUALLY save (Wei’s grid): recommend MAX
                grid_46x16 = pooled_max if args.use_max_pool else pooled_avg  # (46,16)

                # -------- Update counters --------
                _preprob = prob_img.detach().cpu().numpy()
                _post_max = pooled_max.detach().cpu().numpy()
                _post_avg = pooled_avg.detach().cpu().numpy()

                bars_preprob_nonempty_05      += int(_bar_ok(_preprob, thr=0.5))
                bars_postpool_max_nonempty_05 += int(_bar_ok(_post_max, thr=0.5))
                bars_postpool_max_nonempty_02 += int(_bar_ok(_post_max, thr=0.2))
                bars_postpool_avg_nonempty_05 += int(_bar_ok(_post_avg, thr=0.5))
                bars_postpool_avg_nonempty_02 += int(_bar_ok(_post_avg, thr=0.2))
                total_bars_seen               += 1

                # -------- Collect outputs --------
                all_out.append(grid_46x16.detach().cpu().numpy().astype("float32"))
                gt = ground_truth_46x16_for_bar(bar_code)
                if gt is not None:
                    if gt.shape != (DRUM_CLASSES, DRUM_STEPS):
                        raise RuntimeError(f"GT shape {gt.shape} != (46,16) for {bar_code}")
                    all_tar.append(gt.astype("float32"))
                all_idx.append(bar_code)

            # (Optional) per-song snapshot — comment out if too chatty
            if DEBUG:
                print(f"[song dbg] song={song_id} bars_seen={total_bars_seen} "
                      f"pre>0.5={bars_preprob_nonempty_05} | "
                      f"max>0.5={bars_postpool_max_nonempty_05} max>0.2={bars_postpool_max_nonempty_02} | "
                      f"avg>0.5={bars_postpool_avg_nonempty_05} avg>0.2={bars_postpool_avg_nonempty_02}")

    # -------- End-of-run summary --------
    if DEBUG:
        print(f"[run dbg] total_bars={total_bars_seen} "
              f"pre>0.5={bars_preprob_nonempty_05} | "
              f"max>0.5={bars_postpool_max_nonempty_05} max>0.2={bars_postpool_max_nonempty_02} | "
              f"avg>0.5={bars_postpool_avg_nonempty_05} avg>0.2={bars_postpool_avg_nonempty_02}")

    # -------- Save NPZ --------
    all_out = np.stack(all_out, axis=0).astype("float32")      # (B,46,16)
    out_dict = {"all_out": all_out, "all_idx": np.array(all_idx, dtype="<U9")}
    if len(all_tar) == len(all_out):
        out_dict["all_tar"] = np.stack(all_tar, axis=0).astype("float32")

    np.savez(out_path, **out_dict)
    print(f"[info] wrote: {out_path}  | bars={all_out.shape[0]}  shape={all_out.shape[1:]}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[info] interrupted.")
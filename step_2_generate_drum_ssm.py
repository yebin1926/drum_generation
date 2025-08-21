#!/usr/bin/env python3
"""
step_2_generate_drum_ssm_from_melodic_ssm_pytorch.py

- Uses YOUR models.py: SSMEncoder + SSMDecoder wrapped in SSMVAE (generator).
- Loads YOUR checkpoint from train_ssm_generator.py (best.pt / last.pt).
- Runs inference on melodic SSM pickles and saves:
    ./pre_processed_data/model_out_drum_ssm_pkg.pkl
  with [mel_ssm_batch, drum_ssm_gt_batch, drum_ssm_pred_batch]
"""

import os
import glob
import copy
import pickle
import argparse
import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

# Optional helpers for parity with the notebook (not required to run inference)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from models import SSMEncoder, SSMDecoder, SSMVAE  # your generator components


# ========= notebook helper parity (used by some downstream viz) =========
def get_extracted_full_ssm(ges_ssm_input: np.ndarray, ges_song_len_input: int) -> np.ndarray:
    out = ges_ssm_input[0:ges_song_len_input, 0:ges_song_len_input]
    return copy.deepcopy(out)

def get_extracted_triangle_ssm(ges_ssm_input: np.ndarray, ges_song_len_input: int) -> np.ndarray:
    out = ges_ssm_input[0:ges_song_len_input, 0:ges_song_len_input]
    triu_1 = np.triu(out, k=0)
    triu_2 = np.rot90(np.flipud(np.triu(out, k=1)), k=-1)
    return copy.deepcopy(triu_1 + triu_2)

def render_ssm_as_inverted_gray(mat: np.ndarray) -> np.ndarray:
    tmp = "./saving_tmp_file.png"
    fig = plt.figure(figsize=[8,8]); ax = fig.add_subplot(111)
    ax.imshow(mat, origin='lower', cmap='hot')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    plt.savefig(tmp, dpi=50, bbox_inches='tight', pad_inches=0); plt.close()
    img = cv2.imread(tmp); 
    if os.path.exists(tmp): os.remove(tmp)
    img = np.mean(img, axis=-1)
    return -img
# ========================================================================


def list_pairs(mel_dir: Path, drum_dir: Path) -> List[Tuple[Path, Path]]:
    """Match mel/drum SSM pkl pairs by stem."""
    mel_files = sorted(glob.glob(str(mel_dir / "song_barlv_ssm_*.pkl")))
    pairs = []
    for mp in mel_files:
        base = Path(mp).stem.replace("song_barlv_ssm_", "")
        dp = drum_dir / f"song_barlv_drum_ssm_{base}.pkl"
        if dp.exists():
            pairs.append((Path(mp), dp))
    return pairs


def load_arrays(pairs: List[Tuple[Path, Path]]):
    """Load all mel/drum SSM pickles -> np arrays with shapes:
       mel_all:  (N, 256, 256)
       drum_all: (N, 256, 256)
    """
    mel_list, drum_list = [], []
    for mp, dp in pairs:
        with open(mp, "rb") as f: mel = pickle.load(f)
        with open(dp, "rb") as f: drm = pickle.load(f)
        mel_list.append(np.asarray(mel, dtype=np.float32))
        drum_list.append(np.asarray(drm, dtype=np.float32))
    mel_all  = np.stack(mel_list,  axis=0) if mel_list else np.zeros((0,256,256), np.float32)
    drum_all = np.stack(drum_list, axis=0) if drum_list else np.zeros((0,256,256), np.float32)
    return mel_all, drum_all


def batched_infer(vae: nn.Module, mel: np.ndarray, device: torch.device, batch_size: int = 32) -> np.ndarray:
    """mel: (N,256,256) float32 in [0,1] -> returns pred drum SSM (N,256,256)"""
    vae.eval()
    preds = []
    with torch.no_grad():
        N = mel.shape[0]
        for i in range(0, N, batch_size):
            chunk = mel[i:i+batch_size]                   # (B,256,256)
            x = torch.from_numpy(chunk[:, None, ...]).to(device)  # (B,1,256,256)
            recon, _, _ = vae(x)                          # (B,1,256,256)
            preds.append(recon.squeeze(1).cpu().numpy())  # (B,256,256)
    return np.concatenate(preds, axis=0) if preds else np.zeros_like(mel)


def main():
    ap = argparse.ArgumentParser(description="Run step_2 inference using YOUR PyTorch generator.")
    ap.add_argument("--mel_dir",  type=str, default="./pre_processed_data/bar_level_cqt_ssm",
                    help="Directory with melodic SSM pickles (song_barlv_ssm_*.pkl).")
    ap.add_argument("--drum_dir", type=str, default="./pre_processed_data/bar_level_drum_ssm",
                    help="Directory with drum SSM pickles (song_barlv_drum_ssm_*.pkl).")
    ap.add_argument("--ckpt",     type=str, default="./checkpoints/ssm_generator/best.pt",
                    help="Path to your generator checkpoint from train_ssm_generator.py.")
    ap.add_argument("--out",      type=str, default="./pre_processed_data/model_out_drum_ssm_pkg.pkl",
                    help="Output PKL path to save [mel, drum_gt, drum_pred].")
    ap.add_argument("--device",   type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="'cuda' or 'cpu'")
    ap.add_argument("--batch_size", type=int, default=32, help="Inference batch size.")
    args = ap.parse_args()

    device = torch.device(args.device)
    print("[info] step_2 (PyTorch) starting…")
    print(f"[info] device: {device}")

    mel_dir  = Path(args.mel_dir)
    drum_dir = Path(args.drum_dir)
    out_path = Path(args.out)

    # 1) Build YOUR generator only (SSMVAE)
    enc = SSMEncoder(in_channels=1, base_channels=64, latent_dim=32)
    dec = SSMDecoder(out_channels=1, base_channels=64, latent_dim=32)
    vae = SSMVAE(enc, dec).to(device)

    # 2) Load YOUR checkpoint (state has key "vae" per your trainer)
    if os.path.exists(args.ckpt):
        state = torch.load(args.ckpt, map_location=device)
        sd = state.get("vae", state)  # tolerate raw state_dict or wrapped dict
        vae.load_state_dict(sd, strict=False)
        print(f"[info] loaded generator weights from: {args.ckpt}")
    else:
        # Fallback to last.pt in same folder if best.pt missing
        alt = str(Path(args.ckpt).with_name("last.pt"))
        if os.path.exists(alt):
            state = torch.load(alt, map_location=device)
            sd = state.get("vae", state)
            vae.load_state_dict(sd, strict=False)
            print(f"[info] loaded generator weights from: {alt}")
        else:
            print(f"[warn] checkpoint not found at {args.ckpt} (or {alt}); running with random weights.")

    # 3) Collect evaluation pairs (mel, drum_gt)
    pairs = list_pairs(mel_dir, drum_dir)
    print(f"[info] found pairs: {len(pairs)}")
    if len(pairs) == 0:
        raise FileNotFoundError("No matched mel/drum SSM pairs found. "
                                "Check your pre_processed_data directories.")

    # 4) Load arrays
    mel_all, drum_gt_all = load_arrays(pairs)  # shapes (N,256,256)
    print("[info] arrays loaded.",
          f"mel {mel_all.shape} drum_gt {drum_gt_all.shape}")

    # 5) Inference
    print("[info] Start testing…")
    print(datetime.datetime.now().strftime("[info] %Y-%m-%d %H:%M:%S") + "\n")
    drum_pred_all = batched_infer(vae, mel_all, device=device, batch_size=args.batch_size)
    print("[info] Test loop: [ 1 / 1 ]\n")
    print("[info] Drum SSM arrange test is finished.")
    print(f"[info] Songs tested: {mel_all.shape[0]}")
    print(datetime.datetime.now().strftime("[info] %Y-%m-%d %H:%M:%S") + "\n")

    # 6) Save merged output (same bundle layout as your step_2)
    merged = [mel_all, drum_gt_all, drum_pred_all]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(merged, f)
    print(f"[info] Saved: {out_path}")


if __name__ == "__main__":
    main()

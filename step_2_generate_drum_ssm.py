#!/usr/bin/env python3
"""
step_2_generate_drum_ssm2.py

PyTorch reimplementation of the inference stage from
`step_2_generate_drum_ssm_from_melodic_ssm.ipynb`.

Given melodic SSMs (256x256), predict drum SSMs using the trained
SSM VAE generator from models.py and the checkpoint produced by
train_ssm_generator.py.

Outputs:
- ./pre_processed_data/model_out_drum_ssm_pkg.pkl
  -> [n_bars_list_for_save, cqt_ssm_list_for_save, drum_ssm_list_for_save, drum_model_ssm_list_for_save]
- ./pre_processed_data/model_out_bar_level_drum_ssm/model_out_song_barlv_drum_ssm_<stem>.pkl
"""

import argparse
import sys
import glob
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Make sure we can import models.py
sys.path.append(str(Path(__file__).parent))
sys.path.append("/mnt/data")
try:
    from models import SSMEncoder, SSMDecoder, SSMVAE  # expected names
except Exception as e:
    print("[fatal] Could not import models.py (expecting SSMEncoder, SSMDecoder, SSMVAE).")
    raise

# -----------------------------
# Helpers
# -----------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_pickle_matrix(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        mat = pickle.load(f)
    return np.asarray(mat)

def save_pickle_matrix(path: Path, mat: np.ndarray):
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        pickle.dump(mat, f, protocol=pickle.HIGHEST_PROTOCOL)

def estimate_bars_from_ssm(ssm: np.ndarray) -> int:
    """
    Heuristic to recover original bar count from a 256x256 SSM that may be zero-padded.
    """
    if ssm.ndim != 2 or ssm.shape[0] != ssm.shape[1]:
        return 0
    row_energy = (np.abs(ssm).sum(axis=1) > 0).astype(np.uint8)
    col_energy = (np.abs(ssm).sum(axis=0) > 0).astype(np.uint8)
    rows = np.where(row_energy > 0)[0]
    cols = np.where(col_energy > 0)[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0
    return int(min(rows[-1] + 1, cols[-1] + 1))

def make_dummy_melodic_ssms(mel_dir: Path, n: int):
    """
    Create N synthetic 256x256 melodic SSMs with obvious block/diagonal structure,
    so bar-count estimation and saving logic can be exercised.
    """
    ensure_dir(mel_dir)
    rng = np.random.default_rng(7)
    for i in range(n):
        ssm = np.zeros((256, 256), dtype=np.float32)
        # Create 3 sections with higher intra-section similarity
        cuts = [0, 64, 128, 180]  # last section shorter to test padding
        for a, b in zip(cuts[:-1], cuts[1:]):
            block = rng.random((b-a, b-a), dtype=np.float32) * 0.2
            block = (block + block.T) * 0.5  # symmetric
            # brighten diagonal (higher similarity)
            block += np.eye(b-a, dtype=np.float32) * 0.8
            ssm[a:b, a:b] = np.clip(block, 0, 1)
        # Add a faint global diagonal so it's not wholly zero out of blocks
        ssm += np.eye(256, dtype=np.float32) * 0.1
        ssm = np.clip(ssm, 0, 1)
        out = mel_dir / f"song_barlv_ssm_dummy{i:04d}.pkl"
        with open(out, "wb") as f:
            pickle.dump(ssm, f, protocol=pickle.HIGHEST_PROTOCOL)

# -----------------------------
# Dataset for inference
# -----------------------------

class MelodicSSMInferenceDataset(Dataset):
    """
    Yields (mel_ssm_tensor, mel_path_str, drum_gt_tensor_or_None)
      - mel_ssm_tensor: (1,256,256) float32 in [0,1]
      - drum_gt_tensor: (1,256,256) float32 in [0,1] if available, else None
    """
    def __init__(self, mel_dir: Path, drum_dir: Optional[Path] = None, limit: int = 0):
        self.items: List[Tuple[Path, Optional[Path]]] = []
        mel_files = sorted(glob.glob(str(mel_dir / "song_barlv_ssm_*.pkl")))
        for p in mel_files:
            base = Path(p).stem.replace("song_barlv_ssm_", "")
            dpath = None
            if drum_dir is not None:
                cand = drum_dir / f"song_barlv_drum_ssm_{base}.pkl"
                if cand.exists():
                    dpath = cand
            self.items.append((Path(p), dpath))
        if limit and limit > 0:
            self.items = self.items[:limit]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        mpath, dpath = self.items[idx]
        mel = load_pickle_matrix(mpath)
        mel = np.nan_to_num(mel, nan=0.0, posinf=1.0, neginf=0.0)
        mel = np.clip(mel, 0.0, 1.0).astype(np.float32)
        mel_t = torch.from_numpy(mel[None, ...])  # (1,256,256)

        drum_t = None
        if dpath is not None:
            drm = load_pickle_matrix(dpath)
            drm = np.nan_to_num(drm, nan=0.0, posinf=1.0, neginf=0.0)
            drm = np.clip(drm, 0.0, 1.0).astype(np.float32)
            drum_t = torch.from_numpy(drm[None, ...])  # (1,256,256)
        return mel_t, mpath.as_posix(), drum_t

# ---- custom collate: lets drum GT be a list with None entries ----
def collate_infer(batch):
    # batch is a list of (mel_t, mpath_str, drum_t_or_none)
    mel_ts   = [b[0] for b in batch]                 # tensors (1,256,256)
    paths    = [b[1] for b in batch]                 # strings
    drum_gts = [b[2] for b in batch]                 # list of Tensor or None
    mel_batch = torch.stack(mel_ts, dim=0)           # (B,1,256,256)
    return mel_batch, paths, drum_gts

# -----------------------------
# Inference core
# -----------------------------

@torch.no_grad()
def predict_drum_ssm(vae: "SSMVAE", batch_mel: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Inputs:
      batch_mel: (B,1,256,256)
    Returns:
      pred: (B,1,256,256) clamped to [0,1]
    """
    vae.eval()
    batch_mel = batch_mel.to(device)
    recon, mu, logvar = vae(batch_mel)
    pred = torch.clamp(recon, 0.0, 1.0)
    return pred.cpu()

def locate_default_ckpt(pre_root: Path) -> Optional[Path]:
    """
    Try common checkpoint locations when --ckpt is not provided.
    """
    candidates = [
        pre_root.parent / "checkpoints" / "ssm_generator" / "best.pt",
        pre_root.parent / "checkpoints" / "ssm_generator" / "last.pt",
        Path("./checkpoints/ssm_generator/best.pt"),
        Path("./checkpoints/ssm_generator/last.pt"),
        Path("/data/checkpoints/ssm_generator/best.pt"),
        Path("/data/checkpoints/ssm_generator/last.pt"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

def main():
    ap = argparse.ArgumentParser(description="Predict drum SSMs from melodic SSMs (PyTorch)")
    ap.add_argument("--pre-root", type=str, default="./pre_processed_data",
                    help="Root of preprocessed data (contains bar_level_cqt_ssm & bar_level_drum_ssm)")
    ap.add_argument("--ckpt", type=str, default="",
                    help="Path to SSM generator checkpoint (best.pt). If omitted, we try common locations.")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="Limit number of songs (0 = all)")
    ap.add_argument("--save-merged-pkl", action="store_true",
                    help="Also save merged model_out_drum_ssm_pkg.pkl (default True)")
    # --- TEST MODES (no training needed) ---
    ap.add_argument("--allow-random", action="store_true",
                    help="If no checkpoint is found, proceed with RANDOMLY INITIALIZED weights (smoke test).")
    ap.add_argument("--make-dummy", type=int, default=0,
                    help="Create N dummy melodic SSM PKLs in bar_level_cqt_ssm for a full dry run.")
    ap.set_defaults(save_merged_pkl=True)
    args = ap.parse_args()

    PRE_ROOT = Path(args.pre_root).resolve()
    MEL_DIR  = PRE_ROOT / "bar_level_cqt_ssm"
    DRM_DIR  = PRE_ROOT / "bar_level_drum_ssm"  # optional
    OUT_DIR  = PRE_ROOT / "model_out_bar_level_drum_ssm"
    MERGED_PKL = PRE_ROOT / "model_out_drum_ssm_pkg.pkl"

    print(f"[paths] PRE_ROOT={PRE_ROOT}")
    print(f"[paths] MEL_DIR={MEL_DIR}")
    print(f"[paths] DRM_DIR={DRM_DIR} (ground-truth optional)")
    print(f"[paths] OUT_DIR={OUT_DIR}")

    if not MEL_DIR.exists():
        print("[fatal] Could not find melodic SSM directory:", MEL_DIR)
        sys.exit(1)

    # Optionally create dummy inputs for a full dry run
    if args.make_dummy and args.make_dummy > 0:
        print(f"[test] Creating {args.make_dummy} dummy melodic SSM(s) in {MEL_DIR} ...")
        make_dummy_melodic_ssms(MEL_DIR, args.make_dummy)

    # Device
    use_cuda = (args.device == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[cfg] device={device} (cuda_available={torch.cuda.is_available()})")

    # Build VAE as in training
    enc = SSMEncoder(in_channels=1, base_channels=64, latent_dim=32)
    dec = SSMDecoder(out_channels=1, base_channels=64, latent_dim=32)
    vae = SSMVAE(enc, dec).to(device)

    # Load checkpoint
    ckpt_path = Path(args.ckpt) if args.ckpt else locate_default_ckpt(PRE_ROOT)
    if ckpt_path is None or not ckpt_path.exists():
        if args.allow_random:
            print("[warn] No checkpoint found; proceeding with RANDOMLY INITIALIZED weights (smoke test only).")
            # leave `vae` as freshly initialized; do nothing
        else:
            print("[fatal] Could not find checkpoint. Provide --ckpt or add --allow-random for a smoke test.")
            sys.exit(1)
    else:
        print(f"[ckpt] Loading {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("vae", ckpt)  # allow both formats
        vae.load_state_dict(state, strict=False)
        print("[ckpt] Loaded VAE weights.")

    # Dataset & loader
    drum_dir_opt = DRM_DIR if DRM_DIR.exists() else None
    ds = MelodicSSMInferenceDataset(MEL_DIR, drum_dir_opt, limit=args.limit)
    if len(ds) == 0:
        print("[fatal] No melodic SSM PKLs found in", MEL_DIR)
        sys.exit(1)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_infer,   # <-- important
    )

    # Accumulators matching the notebook
    n_bars_list_for_save: List[int]   = []
    cqt_ssm_list_for_save: List[np.ndarray]  = []
    drum_ssm_list_for_save: List[Optional[np.ndarray]] = []
    drum_model_ssm_list_for_save: List[np.ndarray] = []

    # Optional quick sanity metric where GT exists
    sims: List[float] = []

    print(f"[run] Starting inference on {len(ds)} songs...")
    for batch in tqdm(dl, ncols=100):
        mel_t, mel_paths, drum_gt_list = batch   # drum_gt_list is a Python list (may contain None)
        preds = predict_drum_ssm(vae, mel_t, device)  # (B,1,256,256)

        B = preds.shape[0]
        for i in range(B):
            mel_path = Path(mel_paths[i])
            stem = mel_path.stem.replace("song_barlv_ssm_", "")
            pred_img = preds[i, 0].numpy()  # (256,256)

            # Save per-song predicted drum SSM
            out_path = OUT_DIR / f"model_out_song_barlv_drum_ssm_{stem}.pkl"
            save_pickle_matrix(out_path, pred_img)

            # Bars estimate from melodic SSM
            mel_img = mel_t[i, 0].numpy()
            est_bars = estimate_bars_from_ssm(mel_img)

            # Append for merged save
            n_bars_list_for_save.append(est_bars)
            cqt_ssm_list_for_save.append(mel_img)
            drum_model_ssm_list_for_save.append(pred_img)

            # Handle GT (may be Tensor or list-of-Optional[Tensors])
            gt_img = drum_gt_list[i]
            if isinstance(gt_img, torch.Tensor):
                gt_np = gt_img[0].numpy()                 # (256,256)
                drum_ssm_list_for_save.append(gt_np)
                # optional cosine similarity sanity check:
                gt_flat = gt_np.reshape(-1)
                pr_flat = pred_img.reshape(-1)
                den = (np.linalg.norm(gt_flat) * np.linalg.norm(pr_flat) + 1e-8)
                sims.append(float((gt_flat * pr_flat).sum() / den))
            else:
                drum_ssm_list_for_save.append(None)

            if gt_img is not None:
                drum_ssm_list_for_save.append(gt_img)
                # cosine similarity for sanity check
                gt_flat = gt_img.reshape(-1)
                pr_flat = pred_img.reshape(-1)
                den = (np.linalg.norm(gt_flat) * np.linalg.norm(pr_flat) + 1e-8)
                sims.append(float((gt_flat * pr_flat).sum() / den))
            else:
                drum_ssm_list_for_save.append(None)

    # Save the merged package
    merged = [
        n_bars_list_for_save,
        cqt_ssm_list_for_save,
        drum_ssm_list_for_save,
        drum_model_ssm_list_for_save,
    ]
    save_pickle_matrix(MERGED_PKL, merged)
    print(f"\n[info] Saved merged results to {MERGED_PKL}")
    print(f"[info] Also saved per-song predictions under {OUT_DIR}")

    if len(sims) > 0:
        print(f"[eval] Cosine similarity vs. oracle (mean over GT-present songs): {sum(sims)/len(sims):.4f}")

    print("[done] Drum SSM prediction completed.")

if __name__ == "__main__":
    main()

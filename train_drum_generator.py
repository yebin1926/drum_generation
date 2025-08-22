#!/usr/bin/env python3
# train_drum_generator.py
# Drum pattern generator training (Wei et al., 2019 §3.3–§3.4, §4.1–§4.4)
# - Input  (X): 8-channel bar-selected CQT spectrogram (8, 84, 96)
# - Target (Y): drum bar (46, 16) upsampled to (256, 256) to match DrumDecoder output
# - Loss   : BCE(recon, target_img) + KL + L1(note_density)
# References: Wei et al. (2019) ISMIR paper  (§3.1 data, §3.3 bar selection, §3.4 VAE) :contentReference[oaicite:0]{index=0}
# Models: DrumEncoder/DrumDecoder/DrumVAE from your models.py  :contentReference[oaicite:1]{index=1}

import os
import csv
import math
import time
import glob
import pickle
import random
import warnings
from pathlib import Path
from functools import lru_cache

import numpy as np
import librosa
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

import pypianoroll as ppr

# -------------------------
# Local models
# -------------------------
from models import DrumEncoder, DrumDecoder, DrumVAE  # uses your existing file

# -------------------------
# Config (you can override via CLI flags)
# -------------------------

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Paths will be (re)configured below; these are defaults if not overridden
PROJECT_DIR   = Path.cwd()
DATASET_ROOT  = Path("/data")    # directory that contains LPD-5 *.npz
OUT_PRE       = None             # will be set by configure_paths()
OUT_WAV_ND    = None
OUT_DRUM_SSM  = None
OUT_MEL_SSM   = None
OUT_CQT_POOL  = None
CKPT_DIR      = PROJECT_DIR / "checkpoints" / "drum_generator"
LOG_DIR       = PROJECT_DIR / "logs"

# Audio / CQT parameters (per Wei §3.1)
SR = 44100
HOP = 256
N_BINS = 84          # 84 CQT bins
BAR_STEPS = 96       # frames per bar in spectrogram
TEMPO_QPM = 120.0    # normalized tempo

# Drum representation (per Wei §4.1, §4.2)
DRUM_PITCH_MIN = 35  # GM percussion range 35..81
DRUM_PITCH_MAX = 81
# We drop 58 (Vibra Slap) to get 46 instruments total (47 - 1) as in paper
DRUM_KEEP_PITCHES = [p for p in range(DRUM_PITCH_MIN, DRUM_PITCH_MAX + 1) if p != 58]
N_INSTR = len(DRUM_KEEP_PITCHES)  # 46

# Model / training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_EPOCHS = 60
LR = 1e-4
BETA1, BETA2 = 0.5, 0.999
LAMBDA_REC = 1.0
LAMBDA_KL  = 1.0
LAMBDA_C   = 1.0  # weight for note density loss

EARLY_STOP_PATIENCE = 10   # optional early stopping
MIN_EPOCHS = 5

# -------------------------
# Small utilities
# -------------------------

def ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def configure_paths(dataset_root: Path = None, out_root: Path = None):
    """Make preprocessed paths consistent with your SSM trainer."""
    global DATASET_ROOT, OUT_PRE, OUT_WAV_ND, OUT_DRUM_SSM, OUT_MEL_SSM, OUT_CQT_POOL
    if dataset_root is not None:
        DATASET_ROOT = Path(dataset_root)
    # Prefer placing preprocessed data under dataset_root to save workspace
    out_root = Path(out_root) if out_root else DATASET_ROOT
    OUT_PRE      = out_root / "pre_processed_data"
    OUT_WAV_ND   = OUT_PRE / "proc_no_drum_wav"
    OUT_MEL_SSM  = OUT_PRE / "bar_level_cqt_ssm"
    OUT_DRUM_SSM = OUT_PRE / "bar_level_drum_ssm"
    OUT_CQT_POOL = OUT_PRE / "cqt_pooled_data"  # optional on-disk cache of (B,84,96)

    for p in [OUT_WAV_ND, OUT_MEL_SSM, OUT_DRUM_SSM, OUT_CQT_POOL, CKPT_DIR, LOG_DIR]:
        p.mkdir(parents=True, exist_ok=True)

    print(f"[paths] DATASET_ROOT={DATASET_ROOT}")
    print(f"[paths] OUT_PRE={OUT_PRE}")

def list_npz_files(root: Path):
    return sorted([Path(p) for p in glob.glob(str(root / "**" / "*.npz"), recursive=True)])

def multitrack_max_length_steps(mt: ppr.Multitrack) -> int:
    L = 0
    for tr in mt.tracks:
        if tr.pianoroll is not None:
            L = max(L, tr.pianoroll.shape[0])
    return L

def find_drum_track_index(mt: ppr.Multitrack):
    for i, tr in enumerate(mt.tracks):
        if getattr(tr, "is_drum", False):
            return i
    return None

def synth_downbeat_indices(mt: ppr.Multitrack):
    """Synthesize downbeats assuming 4/4 & uniform beat_resolution (like we used for SSMs)."""
    # Use attribute if present & valid
    db = getattr(mt, "downbeat", None)
    if db is not None:
        db = np.asarray(db).squeeze()
        if db.dtype != bool:
            db = db.astype(bool)
        if db.ndim == 1 and db.shape[0] == multitrack_max_length_steps(mt):
            idx = np.where(db)[0]
            if len(idx) > 1:
                return idx.tolist()
    # Fallback: assume 4*resolution steps per bar
    res = int(getattr(mt, "beat_resolution", 24))
    steps_per_bar = 4 * res
    T = multitrack_max_length_steps(mt)
    return list(range(0, T, steps_per_bar))

def steps_to_seconds(indices, resolution: int, tempo_qpm=TEMPO_QPM):
    sec_per_quarter = 60.0 / tempo_qpm
    sec_per_step = sec_per_quarter / resolution
    return [i * sec_per_step for i in indices]

def compute_bar_grid_seconds(mt: ppr.Multitrack):
    """Bar grid for pooling: list of arrays (96,2) per bar with (start,end) seconds per 1/96th slot."""
    res = int(getattr(mt, "beat_resolution", 24))
    T = multitrack_max_length_steps(mt)
    # Normalize tempo in Multitrack (in-place) to fixed 120 QPM for consistent timing
    mt.tempo = np.full((T, 1), TEMPO_QPM, dtype=float)
    db_steps = synth_downbeat_indices(mt)
    if len(db_steps) < 2:
        return []
    bar_edges_sec = steps_to_seconds(db_steps, res, TEMPO_QPM)

    grids = []
    for b in range(len(bar_edges_sec) - 1):
        t0, t1 = bar_edges_sec[b], bar_edges_sec[b+1]
        time_grid = np.linspace(t0, t1, BAR_STEPS + 1)
        bar_grid = np.stack([time_grid[:-1], time_grid[1:]], axis=1)
        # center by half-bin, non-negative
        half = (t1 - t0) / BAR_STEPS * 0.5
        bar_grid = bar_grid - half
        bar_grid[bar_grid < 0] = 0.0
        grids.append(bar_grid.astype(np.float32))
    return grids

def load_no_drum_wav_path(stem: str) -> Path:
    return OUT_WAV_ND / f"{stem}_no_drum.wav"

def load_drum_ssm_path(stem: str) -> Path:
    return OUT_DRUM_SSM / f"song_barlv_drum_ssm_{stem}.pkl"

def minmax01(x: np.ndarray):
    x = np.asarray(x, dtype=np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx <= mn + 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def compute_cqt_pooled(y: np.ndarray, sr: int, bar_grids: list):
    """Return per-bar CQT mean-pooled spectrograms: array (B,84,96)."""
    if len(bar_grids) == 0:
        return np.zeros((0, N_BINS, BAR_STEPS), dtype=np.float32)
    cqt = librosa.cqt(y, sr=sr, hop_length=HOP)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)   # (84, frames)
    dur = len(y) / sr
    fps = cqt_db.shape[1] / max(dur, 1e-7)

    bars_cqt = []
    for bar_grid in bar_grids:
        bins = []
        for n in range(BAR_STEPS):
            t0, t1 = bar_grid[n, 0], bar_grid[n, 1]
            f0 = int(np.round(t0 * fps))
            f1 = int(np.round(t1 * fps))
            if f1 <= f0:
                feat = bins[-1] if bins else np.zeros((N_BINS,), dtype=np.float32)
            else:
                sl = cqt_db[:, f0:f1]
                feat = sl.mean(axis=1).astype(np.float32)
            bins.append(feat)
        img = np.stack(bins, axis=1)          # (84,96)
        img = minmax01(img)                   # scale per-bar to [0,1]
        bars_cqt.append(img)
    return np.asarray(bars_cqt, dtype=np.float32)  # (B,84,96)

def build_pitch_index():
    """Map MIDI pitches to [0..45] for our 46 instruments."""
    mp = {p: i for i, p in enumerate(DRUM_KEEP_PITCHES)}
    return mp

PITCH_TO_IDX = build_pitch_index()

def slice_drum_bars_128x96(mt: ppr.Multitrack, drum_idx: int, bar_edges_steps: list):
    """Symbolic drum bars (128,96) binarized velocities."""
    drum_roll = mt.tracks[drum_idx].pianoroll  # (T,128)
    if drum_roll is None or drum_roll.size == 0:
        return np.zeros((0,128,96), dtype=np.float32)
    X = (drum_roll > 0).astype(np.float32)  # binarize
    bars = []
    for b in range(len(bar_edges_steps) - 1):
        s, e = bar_edges_steps[b], bar_edges_steps[b+1]
        bar = X[s:e, :]                   # (steps,128)
        # pad/trim to 96 steps
        if bar.shape[0] > BAR_STEPS:
            bar = bar[:BAR_STEPS, :]
        elif bar.shape[0] < BAR_STEPS:
            pad = np.zeros((BAR_STEPS - bar.shape[0], bar.shape[1]), dtype=bar.dtype)
            bar = np.concatenate([bar, pad], axis=0)
        bars.append(bar.T)                 # (128,96)
    return np.asarray(bars, dtype=np.float32)   # (B,128,96)

def quantize_96_to_16(bar_128x96: np.ndarray):
    """Downsample time from 96 to 16 steps by OR-pooling each 6-step window."""
    B, P, T = bar_128x96.shape
    assert T == 96, "expects (B,*,96)"
    out = np.zeros((B, P, 16), dtype=np.float32)
    for i in range(16):
        sl = bar_128x96[:, :, i*6:(i+1)*6]
        out[:, :, i] = (sl.sum(axis=2) > 0).astype(np.float32)
    return out  # (B,P,16)

def map_pitches_to_46(bar_PxT: np.ndarray):
    """Map 128 MIDI rows -> 46 instrument rows."""
    B, P, T = bar_PxT.shape
    out = np.zeros((B, N_INSTR, T), dtype=np.float32)
    for midi_pitch in range(128):
        if midi_pitch in PITCH_TO_IDX:
            out[:, PITCH_TO_IDX[midi_pitch], :] += bar_PxT[:, midi_pitch, :]
    out = (out > 0).astype(np.float32)
    return out  # (B,46,16)

def compute_note_density(bar_46x16: np.ndarray):
    """Simple proxy: proportion of active cells per bar (0..1)."""
    # shape (46,16)
    return float(bar_46x16.mean())

def knn_bar_indices_from_ssm(ssm: np.ndarray, k: int, bar_idx: int):
    """Given BxB drum SSM (Euclidean distances), return [self+7 nearest] indices for this bar."""
    B = ssm.shape[0]
    col = ssm[:, bar_idx]
    # nearest (smallest distances). Ensure self included at position 0.
    order = np.argsort(col)
    # put self first, then next k nearest excluding self
    near = [bar_idx] + [i for i in order if i != bar_idx][:k]
    return near[:(k+1)]

def channel_weights_from_ssm(ssm: np.ndarray, indices: list, col_j: int):
    """Weights: 1 - normalized distance within this column (per Wei §3.3)."""
    col = ssm[:, col_j]
    dmin, dmax = float(col.min()), float(col.max())
    denom = max(dmax - dmin, 1e-8)
    w = []
    for i in indices:
        d = float(col[i])
        w.append(1.0 - (d - dmin) / denom)
    return np.asarray(w, dtype=np.float32)  # len = len(indices)

def upsample_46x16_to_256x256(bar_46x16: torch.Tensor):
    """Nearest-neighbor upsample to (1,256,256) to match DrumDecoder output."""
    # bar_46x16: (1,46,16)
    x = bar_46x16.unsqueeze(0)  # (1,1,46,16)
    x = F.interpolate(x, size=(256, 256), mode="nearest")
    return x.squeeze(0)  # (1,256,256)

# -------------------------
# Dataset
# -------------------------

class DrumGenDataset(Dataset):
    """
    each sample = (X, Y, c)
      X: (8,84,96) float32  (bar-selected CQT channels)
      Y: (1,256,256) float32 target image (upsampled from (46,16))
      c: (1,) float32 note density (0..1)
    """
    def __init__(self, dataset_root: Path, limit_songs=None, k_neighbors=7, cache_cqt_to_disk=True):
        super().__init__()
        self.root = Path(dataset_root)
        self.k = k_neighbors
        self.cache_cqt_to_disk = cache_cqt_to_disk

        # Collect candidate songs: require wav_no_drum + drum SSM pickle + npz file
        npzs = list_npz_files(self.root)
        stems = {}
        for p in npzs:
            stems[p.stem] = p
        samples = []
        for stem, npz in stems.items():
            wav = load_no_drum_wav_path(stem)
            ssm_p = load_drum_ssm_path(stem)
            if wav.exists() and ssm_p.exists():
                samples.append((stem, npz, wav, ssm_p))
        if limit_songs is not None:
            samples = samples[:limit_songs]
        self.songs = samples  # list of tuples

        # Build global index over bars
        self.index = []  # (song_idx, bar_idx)
        self.song_meta = []  # per-song cache: (B, ...) computed lazily
        for si, (stem, npz, wav, ssm_p) in enumerate(self.songs):
            # Just find B (bars) now via SSM size
            with open(ssm_p, "rb") as f:
                ssm = pickle.load(f)  # (256,256) padded 0..1
            # Recover actual bar count by scanning padded diagonal until last <1 pad boundary
            # We instead estimate from unpadded region: find rows/cols that are not all equal to border
            # Simpler: derive B from npz downbeats
            mt = ppr.load(str(npz))
            res = int(getattr(mt, "beat_resolution", 24))
            db_steps = synth_downbeat_indices(mt)
            B = max(0, len(db_steps) - 1)
            self.index += [(si, b) for b in range(B)]
            self.song_meta.append({"stem": stem, "res": res, "B": B})
        print(f"[dataset] songs usable: {len(self.songs)}  total bars: {len(self.index)}")

        # In-memory cache per song
        self.cache = {}  # stem -> dict with bars_cqt, ssm_BxB, drum_46x16, density

    def __len__(self):
        return len(self.index)

    def _load_song_cache(self, si):
        """Compute & cache all per-song tensors at first access."""
        stem, npz, wav, ssm_p = self.songs[si]
        if stem in self.cache:
            return self.cache[stem]

        # Load multitrack for timing + drums
        mt = ppr.load(str(npz))
        res = int(getattr(mt, "beat_resolution", 24))
        T  = multitrack_max_length_steps(mt)
        db_steps = synth_downbeat_indices(mt)
        if len(db_steps) < 2:
            # no bars
            self.cache[stem] = {"B": 0}
            return self.cache[stem]
        # bar edges (seconds)
        bar_grids = compute_bar_grid_seconds(mt)

        # Load / compute CQT pooled bars
        if self.cache_cqt_to_disk:
            npy_path = OUT_CQT_POOL / f"{stem}_bars_cqt.npy"
            if npy_path.exists():
                bars_cqt = np.load(npy_path)  # (B,84,96)
            else:
                y, sr = librosa.load(str(wav), sr=SR, mono=True)
                bars_cqt = compute_cqt_pooled(y, sr, bar_grids)  # (B,84,96)
                try:
                    np.save(npy_path, bars_cqt)
                except Exception:
                    pass
        else:
            y, sr = librosa.load(str(wav), sr=SR, mono=True)
            bars_cqt = compute_cqt_pooled(y, sr, bar_grids)

        # Load drum SSM and unpad to BxB
        with open(ssm_p, "rb") as f:
            ssm256 = pickle.load(f).astype(np.float32)  # (256,256) in [0,1]
        B = bars_cqt.shape[0]
        ssm = ssm256[:B, :B]
        # Convert similarity-like [0,1] to distances (small=near). If already distances, minmax ok.
        # We invert because our stored SSM came from minmax01 of Euclidean distances.
        # So treat ssm already as distance in [0,1].
        ssm_BxB = ssm

        # Prepare ground-truth drum bars (46,16)
        d_idx = find_drum_track_index(mt)
        if d_idx is None:
            drum_46x16 = np.zeros((B, N_INSTR, 16), dtype=np.float32)
        else:
            bars_128x96 = slice_drum_bars_128x96(mt, d_idx, db_steps)  # (B,128,96)
            bars_128x16 = quantize_96_to_16(bars_128x96)               # (B,128,16)
            drum_46x16  = map_pitches_to_46(bars_128x16)               # (B,46,16)

        # Note density per bar
        density = np.array([compute_note_density(drum_46x16[b]) for b in range(B)], dtype=np.float32)  # (B,)

        self.cache[stem] = {
            "B": B,
            "bars_cqt": bars_cqt,          # (B,84,96)
            "ssm": ssm_BxB,                # (B,B)
            "drum_46x16": drum_46x16,      # (B,46,16)
            "density": density             # (B,)
        }
        return self.cache[stem]

    def __getitem__(self, idx):
        si, b = self.index[idx]
        stem, npz, wav, ssm_p = self.songs[si]
        cache = self._load_song_cache(si)
        B = cache["B"]
        if B == 0 or b >= B:
            # should not happen; return dummy
            X = np.zeros((8, N_BINS, BAR_STEPS), dtype=np.float32)
            Y = np.zeros((1, 256, 256), dtype=np.float32)
            c = np.zeros((1,), dtype=np.float32)
            return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(c)

        # --- Build 8-channel input via k-NN bar selection (Wei §3.3) ---
        ssm = cache["ssm"]  # (B,B) distances
        inds = knn_bar_indices_from_ssm(ssm, self.k, b)      # [self + 7 nearest]
        w    = channel_weights_from_ssm(ssm, inds, b)        # (8,)
        bars = cache["bars_cqt"][inds]                        # (8,84,96)
        X = (bars * w[:, None, None]).astype(np.float32)      # weight each channel
        # chans-first (8,84,96) already fine for Conv2d(in_ch=8)

        # --- Ground truth drum target for this bar (46,16) -> upsample to (1,256,256) ---
        bar_46x16 = cache["drum_46x16"][b]                   # (46,16)
        c = np.array([compute_note_density(bar_46x16)], dtype=np.float32)  # (1,)
        # torch upsample
        bar_t = torch.from_numpy(bar_46x16[None, ...])        # (1,46,16)
        Yimg = upsample_46x16_to_256x256(bar_t).numpy()       # (1,256,256)

        return torch.from_numpy(X), torch.from_numpy(Yimg), torch.from_numpy(c)

# -------------------------
# Training helpers
# -------------------------

def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def count_params(m: nn.Module):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def get_lr(opt):
    for g in opt.param_groups:
        return g["lr"]

def save_checkpoint(path: Path, epoch, vae, opt, val_loss):
    ensure_dir(path)
    torch.save({
        "epoch": epoch,
        "vae": vae.state_dict(),
        "opt": opt.state_dict(),
        "val": val_loss
    }, path)

# -------------------------
# Main training
# -------------------------

def train_drum(limit=None, epochs=None, batch_size=None, device=None, out_root=None,
               early_stop_patience=EARLY_STOP_PATIENCE):
    global NUM_EPOCHS, BATCH_SIZE, DEVICE
    if device: DEVICE = torch.device(device)
    if epochs is not None: NUM_EPOCHS = epochs
    if batch_size is not None: BATCH_SIZE = batch_size

    configure_paths(DATASET_ROOT, out_root)

    # Dataset (training uses ground-truth drum SSM for bar selection; testing would use predicted SSM) :contentReference[oaicite:2]{index=2}
    ds = DrumGenDataset(DATASET_ROOT, limit_songs=limit, k_neighbors=7, cache_cqt_to_disk=True)
    assert len(ds) > 0, "Dataset is empty — ensure WAV_no_drum and drum SSM pickles exist."

    # Train/val split by bars
    n_total = len(ds)
    n_train = int(0.9 * n_total)
    n_val   = n_total - n_train
    train_set, val_set = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))

    # Batches
    train_bs = max(1, min(BATCH_SIZE, len(train_set)))
    val_bs   = max(1, min(BATCH_SIZE, len(val_set)))
    train_loader = DataLoader(train_set, batch_size=train_bs, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=val_bs,   shuffle=False, num_workers=2, pin_memory=True)

    # Model
    enc = DrumEncoder(in_channels=8, base_channels=64, latent_dim=32)
    dec = DrumDecoder(out_channels=1, base_channels=64, latent_dim=32)
    vae = DrumVAE(enc, dec).to(DEVICE)
    print(f"[model] enc params: {count_params(enc):,}  dec params: {count_params(dec):,}")

    # Opt & losses
    opt = torch.optim.Adam(vae.parameters(), lr=LR, betas=(BETA1, BETA2))
    bce = nn.BCELoss()  # decoder ends with Sigmoid in your models.py

    # CSV log
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = LOG_DIR / "drum_metrics.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch","split","loss","rec","kl","c_l1","lr","secs"])

    # Quick forward sanity
    X0, Y0, c0 = next(iter(train_loader))
    X0 = X0.to(DEVICE, non_blocking=True)
    vae.eval()
    with torch.no_grad():
        recon0, mu0, logvar0, c_hat0 = vae(X0[:1])
    print(f"[sanity] X {tuple(X0.shape)} -> recon {tuple(recon0.shape)}  mu {tuple(mu0.shape)}  ĉ {tuple(c_hat0.shape)}")

    # Train
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    no_improve = 0
    epochs_range = range(1, NUM_EPOCHS+1)

    pbar_epochs = tqdm(epochs_range, desc="[epochs]", unit="ep")
    for epoch in pbar_epochs:
        t0 = time.time()
        vae.train()
        sums = {"loss":0.0, "rec":0.0, "kl":0.0, "c":0.0}
        nb = 0

        pbar = tqdm(train_loader, desc=f"[train ep{epoch:03d}]", leave=False)
        for X, Yimg, c in pbar:
            nb += 1
            X = X.to(DEVICE, non_blocking=True)              # (B,8,84,96)
            Yimg = Yimg.to(DEVICE, non_blocking=True)        # (B,1,256,256)
            c = c.to(DEVICE, non_blocking=True)              # (B,1)

            opt.zero_grad(set_to_none=True)
            recon, mu, logvar, c_hat = vae(X)                # recon (B,1,256,256)
            loss_rec = bce(recon, Yimg)
            loss_kl  = kl_divergence(mu, logvar)
            loss_c   = F.l1_loss(c_hat.view_as(c), c)
            loss = LAMBDA_REC*loss_rec + LAMBDA_KL*loss_kl + LAMBDA_C*loss_c
            if not torch.isfinite(loss):   # safety
                continue
            loss.backward()
            opt.step()

            sums["loss"] += loss.item()
            sums["rec"]  += loss_rec.item()
            sums["kl"]   += loss_kl.item()
            sums["c"]    += loss_c.item()

            pbar.set_postfix(loss=f"{sums['loss']/nb:.3f}",
                             rec=f"{sums['rec']/nb:.3f}",
                             kl=f"{sums['kl']/nb:.3f}",
                             c=f"{sums['c']/nb:.3f}")

        # Validation
        vae.eval()
        val_sums = {"loss":0.0, "rec":0.0, "kl":0.0, "c":0.0}
        vnb = 0
        with torch.no_grad():
            for X, Yimg, c in val_loader:
                vnb += 1
                X = X.to(DEVICE, non_blocking=True)
                Yimg = Yimg.to(DEVICE, non_blocking=True)
                c = c.to(DEVICE, non_blocking=True)
                recon, mu, logvar, c_hat = vae(X)
                loss_rec = bce(recon, Yimg)
                loss_kl  = kl_divergence(mu, logvar)
                loss_c   = F.l1_loss(c_hat.view_as(c), c)
                loss = LAMBDA_REC*loss_rec + LAMBDA_KL*loss_kl + LAMBDA_C*loss_c
                val_sums["loss"] += loss.item()
                val_sums["rec"]  += loss_rec.item()
                val_sums["kl"]   += loss_kl.item()
                val_sums["c"]    += loss_c.item()

        secs = time.time() - t0
        tr_loss = sums["loss"]/max(1,nb)
        va_loss = val_sums["loss"]/max(1,vnb)
        tr_rec = sums["rec"]/max(1,nb); tr_kl = sums["kl"]/max(1,nb); tr_c = sums["c"]/max(1,nb)
        va_rec = val_sums["rec"]/max(1,vnb); va_kl = val_sums["kl"]/max(1,vnb); va_c = val_sums["c"]/max(1,vnb)
        lr = get_lr(opt)

        pbar_epochs.set_postfix(tr=f"{tr_loss:.3f}", val=f"{va_loss:.3f}", lr=f"{lr:.1e}")

        # CSV
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch,"train",tr_loss,tr_rec,tr_kl,tr_c,lr,secs])
            w.writerow([epoch,"val",  va_loss,va_rec,va_kl,va_c,lr,secs])

        # Checkpointing + early stopping
        if va_loss < best_val:
            best_val = va_loss
            no_improve = 0
            save_checkpoint(CKPT_DIR / "best.pt", epoch, vae, opt, best_val)
            print("  ✓ Saved best checkpoint.")
        else:
            no_improve += 1

        # Always save last
        save_checkpoint(CKPT_DIR / "last.pt", epoch, vae, opt, va_loss)

        if epoch >= MIN_EPOCHS and early_stop_patience and no_improve >= early_stop_patience:
            print(f"[early-stop] no improvement for {no_improve} epochs. Best Val={best_val:.4f}")
            break

    print("Training complete.")

# -------------------------
# CLI
# -------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, default="/data", help="Root dir containing **/*.npz (LPD-5)")
    ap.add_argument("--out_root", type=str, default=None, help="Where to write pre_processed_data (default=dataset_root)")
    ap.add_argument("--limit", type=int, default=None, help="limit #songs for training set build (useful for tests)")
    ap.add_argument("--epochs", type=int, default=None, help="override NUM_EPOCHS")
    ap.add_argument("--batch_size", type=int, default=None, help="override BATCH_SIZE")
    ap.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu'")
    ap.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE, help="early-stop patience (epochs)")
    args = ap.parse_args()

    # seed
    torch.backends.cudnn.benchmark = True

    # set paths
    DATASET_ROOT = Path(args.dataset_root)

    train_drum(
        limit=args.limit,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        out_root=args.out_root,
        early_stop_patience=args.patience
    )

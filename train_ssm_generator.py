# train_ssm_generator.py
# End-to-end: prepare data from LPD-5 (via Pypianoroll) and train the SSM VAE-GAN (Wei et al., 2019)
# Paper refs: §3.1 (preprocessing, 96 steps/bar, zero-pad 256 bars), §3.2 (SSM VAE-GAN), §4.3 (train split),
# §4.4 (8 conv + 3 FC + skip, 32-dim latent)

import os
import glob
import random
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import pypianoroll as ppr   # LPD NPZ <-> Multitrack I/O (recommended)  # see docs
from models import SSMEncoder, SSMDecoder, SSMVAE, SSMDiscriminator

import time, csv, math, logging
from tqdm.auto import tqdm
from collections import defaultdict
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False
from typing import Optional
import traceback
import copy

# -------------------------
# Config (paths, seeds, hyperparams)
# -------------------------

# PROJECT_DIR   = Path("/home/marg_intern/marg_intern_2025/yebinpyun/drum_generation") #change to this later
PROJECT_DIR   = Path("/workspace")
DATASET_ROOT  = Path("/mnt/ssd2/marg_intern_2025_summer/yebinpyun")   # will scan **/*.npz
SOUND_FONT    = Path("/workspace/sound_front_lib.sf2")      # change later to home ?

# OUT_PRE       = PROJECT_DIR / "pre_processed_data"
# OUT_MIDI_ALL  = OUT_PRE / "proc_all_tracks_mid"
# OUT_MIDI_ND   = OUT_PRE / "proc_no_drum_mid"
# OUT_MIDI_DO   = OUT_PRE / "proc_drum_only_mid"
# OUT_WAV_ND    = OUT_PRE / "proc_no_drum_wav"
# OUT_OBJ_PKL   = OUT_PRE / "proc_midi_object.pkl"          # like step_1's object list (lightweight)
# OUT_CQT_POOL  = OUT_PRE / "cqt_pooled_data"               # per-bar pooled CQT (84 x 96 per bar)
# OUT_MEL_SSM   = OUT_PRE / "bar_level_cqt_ssm"             # melodic bar-level SSM (NxN)
# OUT_DRUM_SSM  = OUT_PRE / "bar_level_drum_ssm"            # drum bar-level SSM (NxN)
OUT_PRE = None
OUT_MIDI_ALL = None
OUT_MIDI_ND  = None
OUT_MIDI_DO  = None
OUT_WAV_ND   = None
OUT_OBJ_PKL  = None
OUT_CQT_POOL = None
OUT_MEL_SSM  = None
OUT_DRUM_SSM = None
CKPT_DIR      = PROJECT_DIR / "checkpoints" / "ssm_generator"

def configure_paths():
    """Configure all preprocessing output paths under DATASET_ROOT/pre_processed_data."""
    global OUT_PRE, OUT_MIDI_ALL, OUT_MIDI_ND, OUT_MIDI_DO, OUT_WAV_ND
    global OUT_OBJ_PKL, OUT_CQT_POOL, OUT_MEL_SSM, OUT_DRUM_SSM

    OUT_PRE       = (DATASET_ROOT / "pre_processed_data").resolve()
    OUT_MIDI_ALL  = OUT_PRE / "proc_all_tracks_mid"
    OUT_MIDI_ND   = OUT_PRE / "proc_no_drum_mid"
    OUT_MIDI_DO   = OUT_PRE / "proc_drum_only_mid"
    OUT_WAV_ND    = OUT_PRE / "proc_no_drum_wav"
    OUT_OBJ_PKL   = OUT_PRE / "proc_midi_object.pkl"
    OUT_CQT_POOL  = OUT_PRE / "cqt_pooled_data"
    OUT_MEL_SSM   = OUT_PRE / "bar_level_cqt_ssm"
    OUT_DRUM_SSM  = OUT_PRE / "bar_level_drum_ssm"

# training
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_EPOCHS = 60
# LR_GEN = 2e-4
# LR_DIS = 2e-4
LR_GEN = 1e-4
LR_DIS = 1e-4
BETA1, BETA2 = 0.5, 0.999
LAMBDA_REC = 1.0
LAMBDA_KL  = 1.0
LAMBDA_GAN = 1.0

# constants per paper
BAR_STEPS = 96            # 96 time steps per bar (§3.1)
TARGET_BARS = 256         # zero-pad songs to 256 bars (§3.1)
TEMPO_QPM = 120.0         # normalize tempo (§3.1)
SR = 44100
# HOP = 256                 # CQT hop (matches your step_1)
HOP = 512
N_BINS = 84               # pooled CQT freq bins (as in step_1)

# --- housekeeping: control disk usage, delete if unnecessary ---
KEEP_WAV  = False   # delete rendered WAVs right after we finish computing SSMs
KEEP_MIDI = False   # delete MIDI variants right after SSMs are saved
WRITE_META = False  # skip writing proc_midi_object.pkl (not used by training)

# schedules
KL_WARMUP_EPOCHS = 20     # linearly ramp β from 0 -> 1 over 20 epochs
GAN_START_EPOCH   = 5     # train VAE-only for first 5 epochs, then add GAN

# --- data cleaning (Wei et al. style) ---
DRUM_OUTLIER_SIGMA = 2.0   # keep songs whose drum-onset count lies within μ ± 2σ
MIN_DRUM_ONSETS    = 8     # also require at least this many onsets (guards near-silence)



# -------------------------
# Utilities
# -------------------------

def ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def list_npz_files(root: Path):
    # recurse and collect .npz (LPD files)
    return sorted([Path(p) for p in glob.glob(str(root / "**" / "*.npz"), recursive=True)])

def syn_midi_to_wav(midi_path: Path, wav_path: Path, sr=SR):
    """Render MIDI to WAV with Fluidsynth (edit to your synth if needed)."""
    ensure_dir(wav_path)
    if not SOUND_FONT.exists():
        raise FileNotFoundError(f"SoundFont not found: {SOUND_FONT}")
    cmd = [
        "fluidsynth", "-ni", str(SOUND_FONT), str(midi_path),
        "-F", str(wav_path), "-r", str(sr)
    ]
    subprocess.run(cmd, check=True)

def minmax01(x: np.ndarray):
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmax <= xmin + 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - xmin) / (xmax - xmin)).astype(np.float32)

def pad_to_256(ssm: np.ndarray, fill_value=None):
    """Return a 256x256 SSM. If input is larger, crop; if smaller, pad."""
    ssm = np.asarray(ssm)
    assert ssm.ndim == 2 and ssm.shape[0] == ssm.shape[1], f"SSM must be square, got {ssm.shape}"
    n = ssm.shape[0]

    # If too long, crop to first 256 bars (paper: zero-pad to 256; for >256, truncation is standard)
    if n >= TARGET_BARS:
        return ssm[:TARGET_BARS, :TARGET_BARS].astype(np.float32)

    # Else pad up to 256
    out = np.zeros((TARGET_BARS, TARGET_BARS), dtype=ssm.dtype)
    if fill_value is None:
        fill_value = float(np.max(ssm)) if n > 0 else 0.0
    out[:] = float(fill_value)
    out[:n, :n] = ssm
    return out.astype(np.float32)

def pairwise_euclidean_bar_ssm(bar_mats: np.ndarray):
    """Euclidean bar-level SSM from bar-wise matrices (flattened)."""
    if bar_mats.ndim == 3:
        N, F, T = bar_mats.shape
        feat = bar_mats.reshape(N, F*T)
    else:
        feat = bar_mats
        N = feat.shape[0]
    # compute (X - Y)^2
    X2 = (feat**2).sum(axis=1, keepdims=True)
    dist2 = X2 + X2.T - 2.0 * (feat @ feat.T)
    np.maximum(dist2, 0.0, out=dist2)
    return np.sqrt(dist2).astype(np.float32)

def safe_unlink(path: Path):
    """Delete a file if it exists, ignoring 'file not found'."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[warn] could not delete {path}: {e}")

def prune_empty_dirs(*dirs: Path):
    """Remove empty directories (useful after file cleanup)."""
    for d in dirs:
        try:
            if d.exists() and d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        except Exception as e:
            print(f"[warn] could not remove empty dir {d}: {e}")

def is_valid_ssm_pickle(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            arr = pickle.load(f)
        arr = np.asarray(arr)
        return arr.shape == (TARGET_BARS, TARGET_BARS) and np.isfinite(arr).all()
    except Exception:
        return False

def atomic_pickle_dump(obj, dst: Path):
    """Write pickle to a temp file then atomically rename -> no truncated files."""
    ensure_dir(dst)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)


def drum_onset_count_from_npz(npz_path: Path) -> Optional[int]:
    """
    Fast, symbolic drum 'note onset' count:
      - find the drum track (is_drum=True)
      - binarize velocities
      - count rising edges along time for ALL pitches (sums multiple hits per step)
    Returns None if no drum track is present.
    """
    try:
        mt = ppr.load(str(npz_path))
        drum = next((tr for tr in mt.tracks if getattr(tr, "is_drum", False)), None)
        if drum is None or drum.pianoroll is None or drum.pianoroll.size == 0:
            return None
        pr = (drum.pianoroll > 0).astype(np.uint8)   # (T,128)
        first = pr[0, :].sum()
        # rising edges over time for each pitch; sum counts per-pitch onsets
        onsets = ((pr[1:, :] > 0) & (pr[:-1, :] == 0)).sum()
        return int(first + onsets)
    except Exception:
        return None

def regrid_bar_to_96(x_steps, start_idx, end_idx):
    """
    x_steps: np.ndarray with time-first axis [T, ...] (pianoroll frames, CQT frames, etc.)
    start_idx, end_idx: integers in [0, x_steps.shape[0]] delimiting the bar in 'step' units
    Returns: [96, ...] array resampled by averaging frames into 96 equal bins.
    """
    T = x_steps.shape[0]
    start = int(max(0, min(T, start_idx)))
    end   = int(max(0, min(T, end_idx)))
    if end <= start:
        # empty bar → return zeros like one frame
        return np.zeros((96,) + x_steps.shape[1:], dtype=np.float32)

    edges = np.linspace(start, end, 97)              # 96 bins ⇒ 97 edges
    edges = np.clip(edges, 0, T).astype(int)

    bins = []
    for i in range(96):
        a, b = edges[i], edges[i+1]
        if b <= a:
            # duplicate previous or zeros to avoid holes
            bins.append(bins[-1].copy() if bins else np.zeros_like(x_steps[0]))
        else:
            bins.append(x_steps[a:b].mean(axis=0))
    return np.stack(bins, axis=0).astype(np.float32)

def cleanup_intermediates_for_stems(stems):
    """Delete leftover MID/WAV for given stems, respecting KEEP_*."""
    for s in stems:
        if not KEEP_WAV:
            safe_unlink(OUT_WAV_ND   / f"{s}_no_drum.wav")
        if not KEEP_MIDI:
            safe_unlink(OUT_MIDI_ALL / f"{s}_all_tracks.mid")
            safe_unlink(OUT_MIDI_ND  / f"{s}_no_drum.mid")
            safe_unlink(OUT_MIDI_DO  / f"{s}_drum_only.mid")
    prune_empty_dirs(OUT_WAV_ND, OUT_MIDI_ALL, OUT_MIDI_ND, OUT_MIDI_DO)

def _as_colvec(x, T, dtype=None):
    x = np.asarray(x)
    if x.ndim == 0:
        x = np.full((T,), x, dtype=dtype if dtype is not None else np.float32)
    if x.ndim == 1:
        x = x.reshape(T, 1)
    if dtype is not None:
        x = x.astype(dtype, copy=False)
    return x

def fix_multitrack_for_write(mt: ppr.Multitrack,
                             bar_edges_steps: list,
                             tempo_qpm: float = TEMPO_QPM) -> ppr.Multitrack:
    """Force valid shapes for write(): same T for all tracks, 1D tempo & downbeat of length T."""
    # 1) unify track lengths
    T = multitrack_max_length_steps(mt)
    for tr in mt.tracks:
        pr = tr.pianoroll
        if pr is None:
            tr.pianoroll = np.zeros((T, 128), dtype=np.uint8)
            continue
        if pr.shape[0] < T:
            pad = np.zeros((T - pr.shape[0], pr.shape[1]), dtype=pr.dtype)
            tr.pianoroll = np.vstack([pr, pad])
        elif pr.shape[0] > T:
            tr.pianoroll = pr[:T]

    # 2) tempo: 1D float array (T,)
    if not isinstance(getattr(mt, "tempo", None), np.ndarray) or mt.tempo.ndim != 1 or mt.tempo.shape[0] != T:
        mt.tempo = np.full((T,), float(tempo_qpm), dtype=np.float32)

    # 3) downbeat: 1D bool array (T,)
    db = getattr(mt, "downbeat", None)
    ok = isinstance(db, np.ndarray) and db.ndim == 1 and db.shape[0] == T and db.dtype == np.bool_
    if not ok:
        db = np.zeros((T,), dtype=np.bool_)
        idx = np.asarray(bar_edges_steps, dtype=int)
        idx = idx[(idx >= 0) & (idx < T)]
        db[idx] = True
        mt.downbeat = db

    return mt

def log_mt(tag, mt):
    def s(x):
        try: return f"shape={x.shape}, ndim={x.ndim}, dtype={x.dtype}"
        except: return "None" if x is None else str(type(x))
    T = multitrack_max_length_steps(mt)
    print(f"[{tag}] T={T} tempo:{s(getattr(mt,'tempo', None))} "
          f"downbeat:{s(getattr(mt,'downbeat', None))}")

def assert_mt_ok(mt, tag):
    T = multitrack_max_length_steps(mt)
    t = getattr(mt, "tempo", None)
    db = getattr(mt, "downbeat", None)
    def shp(x): return None if x is None else getattr(x, "shape", None)
    print(f"[{tag}] T={T} tempo_shape={shp(t)} downbeat_shape={shp(db)}")
    assert isinstance(t, np.ndarray) and t.ndim == 1 and t.shape[0] == T, \
        f"tempo must be 1D len T, got {None if t is None else t.shape}"
    assert isinstance(db, np.ndarray) and db.ndim == 1 and db.shape[0] == T, \
        f"downbeat must be 1D len T, got {None if db is None else db.shape}"

# -------------------------
# LPD loading & bar slicing (Pypianoroll)
# -------------------------
# LPD NPZ -> pypianoroll.Multitrack (recommended I/O format)  :contentReference[oaicite:3]{index=3}

def load_multitrack(npz_path: Path) -> ppr.Multitrack:
    # return ppr.load(npz_path)  # supports LPD NPZ format
    return ppr.load(str(npz_path)) #changed

def normalize_tempo_to_120(multitrack: ppr.Multitrack):
    """Set tempo array to 120 QPM uniformly (§3.1)."""
    T = multitrack_max_length_steps(multitrack)
    # multitrack.tempo = np.full((T, 1), TEMPO_QPM, dtype=float)
    multitrack.tempo = np.full(T, TEMPO_QPM, dtype=float) #change to (T,) not (T, 1) so its 1D
    return multitrack

def multitrack_max_length_steps(mt):
    L = 0
    for tr in mt.tracks:
        if tr.pianoroll is not None:
            L = max(L, tr.pianoroll.shape[0])
    return L

def get_downbeat_indices(multitrack: ppr.Multitrack):
    """Indices where downbeat is True/1 (start of each bar in LPD)."""
    # db = multitrack.downbeat.squeeze() #flag per timestep - (1 at the start of every bar, 0 elsewhere). squeexe it to 1D
    # return np.where(db > 0)[0].tolist()
    db = multitrack.downbeat
    if db.ndim == 2:        # (T,1) -> (T,)
        db = db.squeeze(1)
    db = db.astype(bool)
    return np.where(db)[0].tolist()

def steps_to_seconds(indices, resolution: int, tempo_qpm=TEMPO_QPM):
    """Convert time-step indices to seconds given resolution and tempo (qpm)."""
    sec_per_quarter = 60.0 / tempo_qpm            # at 120 QPM => 0.5 s per quarter
    sec_per_step = sec_per_quarter / resolution   # e.g., if res=24 => ~0.020833 s/step
    return [i * sec_per_step for i in indices]

def slice_bars(track_roll: np.ndarray, bar_edges_steps: list, steps_per_bar=BAR_STEPS):
    """Split a (T, 128) pianoroll into list of bar matrices shaped (128, steps_per_bar)."""
    bars = []
    for b in range(len(bar_edges_steps) - 1):
        start, end = bar_edges_steps[b], bar_edges_steps[b+1]
        # as LPD uses symbolic time, each bar should be 4*resolution steps; res * 4 == 96 default
        bar = track_roll[start:end, :]  # (steps, 128) -> transpose later to (128, steps)
    #     # Ensure exact width (trim/pad)
    #     if bar.shape[0] != (BAR_STEPS):
    #         # If not exact (rare), resample by simple pad/trim to BAR_STEPS
    #         if bar.shape[0] > steps_per_bar:
    #             bar = bar[:BAR_STEPS, :]
    #         else:
    #             pad = np.zeros((BAR_STEPS - bar.shape[0], bar.shape[1]), dtype=bar.dtype)
    #             bar = np.concatenate([bar, pad], axis=0)
    #     bars.append(bar.T)  # (128, BAR_STEPS)
    # return bars
        if bar.shape[0] > steps_per_bar:
            bar = bar[:steps_per_bar, :]
        elif bar.shape[0] < steps_per_bar:
            pad = np.zeros((steps_per_bar - bar.shape[0], bar.shape[1]), dtype=bar.dtype)
            bar = np.concatenate([bar, pad], axis=0)

        bars.append(bar.T)                           # (128, steps_per_bar)
    return bars

# -------------------------
# Computing Downbeats
# -------------------------

# --- deps ---
import copy
from typing import List, Tuple, Dict, Optional
import pretty_midi as pm


# ============== I/O helpers ==============
def load_pretty_midi_from_npz(npz_path) -> pm.PrettyMIDI:
    """LPD NPZ -> pypianoroll.Multitrack -> PrettyMIDI."""
    mt = ppr.load(str(npz_path))                 # LPD NPZ
    return mt.to_pretty_midi()                   # identical to what the paper did


def load_pretty_midi_from_midi(midi_path) -> pm.PrettyMIDI:
    return pm.PrettyMIDI(str(midi_path))


# ============== bar grid (raw) ==============
def get_beats(pmidi: pm.PrettyMIDI) -> np.ndarray:
    """Return beat times (seconds) as a 1-D float array."""
    beats = pmidi.get_beats()        # works even if time-signatures are missing
    return np.asarray(beats, dtype=float)


def get_downbeats_raw(pmidi: pm.PrettyMIDI) -> List[float]:
    """
    Try pretty_midi.get_downbeats(); if empty/degenerate, synthesize
    downbeats from beats and a time-signature guess (default 4/4).
    """
    db = pmidi.get_downbeats()
    if len(db) >= 2:
        return db.tolist()

    # Fallback: derive from beats every 'numerator' beats (default 4)
    # If you have real TS, you can parse pmidi.time_signature_changes; most LPD files lack it.
    beats = get_beats(pmidi)
    if beats.size < 2:
        return []   # hopeless case; the file is broken

    numerator = _guess_numerator(pmidi)          # default 4
    idx = np.arange(0, beats.size, numerator, dtype=int)
    return beats[idx].tolist()


def _guess_numerator(pmidi: pm.PrettyMIDI, default_num: int = 4) -> int:
    """Return likely TS numerator; use first/most-common if present; else default 4."""
    ts = pmidi.time_signature_changes
    if len(ts) > 0:
        nums = [t.numerator for t in ts]
        vals, counts = np.unique(nums, return_counts=True)
        return int(vals[np.argmax(counts)])
    return default_num


# ============== downbeat correction (paper’s note-density shift) ==============
def fix_downbeats_by_note_density(pmidi: pm.PrettyMIDI,
                                  downbeats_list: List[float],
                                  bar_portion: int = 96) -> Dict[str, np.ndarray]:
    """
    Implements the logic you posted from step_1:
    - make a +- (half basic unit) window around each downbeat
    - count note onsets falling in [standard, +1 unit, -1 unit] windows
    - shift the downbeat by +1/-1 unit if more notes fall there than in standard
    Returns dict with keys:
      - downbeats_fixed (np.ndarray)
      - basic_time_unit (float)
      - counts (dict of arrays for debugging)
    """
    if len(downbeats_list) < 2:
        return {
            "downbeats_fixed": np.asarray(downbeats_list, dtype=float),
            "basic_time_unit": 0.0,
            "counts": {}
        }

    # Basic unit = average bar period / 96
    db = np.asarray(downbeats_list, dtype=float)
    bar_periods = db[1:] - db[:-1]
    bar_period_avg = float(np.mean(bar_periods))
    basic_time_unit = bar_period_avg / float(bar_portion)

    # Construct per-bar windows centered at each downbeat ± 0.5 unit
    downbeats_range = np.vstack([db - 0.5*basic_time_unit,
                                 db + 0.5*basic_time_unit])
    downbeats_range[downbeats_range < 0.0] = 0.0
    n_bars = db.shape[0]

    # Three counters per bar: [standard], [shift+1], [shift-1]
    cnt_std = np.zeros(n_bars, dtype=int)
    cnt_p1  = np.zeros(n_bars, dtype=int)
    cnt_n1  = np.zeros(n_bars, dtype=int)

    # Iterate all note onsets in the piece
    # (paper used all tracks; you can restrict to non-drum if desired)
    for inst in pmidi.instruments:
        for note in inst.notes:
            onset = float(note.start)

            # standard window
            hit = _accum_if_in_ranges(onset, downbeats_range, cnt_std)
            if hit:
                continue

            # +1 unit
            rng_p1 = downbeats_range + basic_time_unit
            hit = _accum_if_in_ranges(onset, rng_p1, cnt_p1)
            if hit:
                continue

            # -1 unit
            rng_n1 = downbeats_range - basic_time_unit
            _accum_if_in_ranges(onset, rng_n1, cnt_n1)

    # Decide shifts: +1 if cnt_p1 > cnt_std; -1 if cnt_n1 > cnt_std
    shift = np.zeros(n_bars, dtype=int)
    for i in range(n_bars):
        if max(cnt_p1[i], cnt_n1[i]) > cnt_std[i]:
            shift[i] = 1 if cnt_p1[i] > cnt_n1[i] else -1

    downbeats_fixed = db + shift.astype(float) * basic_time_unit
    downbeats_fixed[downbeats_fixed < 0.0] = 0.0

    return {
        "downbeats_fixed": downbeats_fixed,
        "basic_time_unit": basic_time_unit,
        "counts": {"std": cnt_std, "+1": cnt_p1, "-1": cnt_n1, "shift": shift}
    }


def _accum_if_in_ranges(onset: float, ranges: np.ndarray, counter: np.ndarray) -> bool:
    """
    ranges: shape (2, n_bars) [[start_i], [end_i]]
    counter: shape (n_bars,)
    If onset in any [start_i, end_i), increment that bar's counter and return True.
    """
    starts = ranges[0]; ends = ranges[1]
    # vectorized test; most efficient: look for first match
    mask = (onset >= starts) & (onset < ends)
    if mask.any():
        i = int(np.argmax(mask))   # first True
        counter[i] += 1
        return True
    return False


# ============== bar windows (centered; paper’s half-unit shift) ==============
def build_bar_ranges(downbeats_fixed: np.ndarray, bar_portion: int = 96) -> List[Tuple[float, float]]:
    """
    Given fixed downbeats (seconds), build [start,end] for each bar:
      [db_i - half_unit, db_{i+1} - half_unit], clipped at 0
    """
    db = np.asarray(downbeats_fixed, dtype=float)
    out = []
    for i in range(len(db) - 1):
        bar_len = db[i+1] - db[i]
        half = (bar_len / float(bar_portion)) * 0.5
        start = max(db[i]   - half, 0.0)
        end   = max(db[i+1] - half, 0.0)
        out.append((start, end))
    return out


# ============== full wrapper ==============
def compute_downbeats_for_song(
    source_path,
    kind: str = "npz",      # "npz" (LPD) or "midi"
    bar_portion: int = 96
) -> Dict[str, object]:
    """
    End-to-end:
      - load PrettyMIDI
      - get raw downbeats (or synthesize from beats)
      - apply note-density correction (± one 96th)
      - produce centered bar ranges
    Returns:
      {
        "downbeats_raw": list[float],
        "downbeats_fixed": np.ndarray,
        "basic_time_unit": float,
        "bar_ranges": list[(start,end)],
        "tempo_bpm": float
      }
    """
    pmidi = (load_pretty_midi_from_npz(source_path)
             if kind == "npz" else
             load_pretty_midi_from_midi(source_path))

    # raw (or synthesized) downbeats
    db_raw = get_downbeats_raw(pmidi)

    # safety: if still <2, abort early
    if len(db_raw) < 2:
        return {
            "downbeats_raw": db_raw,
            "downbeats_fixed": np.asarray(db_raw, dtype=float),
            "basic_time_unit": 0.0,
            "bar_ranges": [],
            "tempo_bpm": estimate_global_tempo(pmidi)
        }

    # note-density based shift
    fix = fix_downbeats_by_note_density(pmidi, db_raw, bar_portion=bar_portion)
    db_fixed = fix["downbeats_fixed"]

    # centered windows
    bars = build_bar_ranges(db_fixed, bar_portion=bar_portion)

    return {
        "downbeats_raw": db_raw,
        "downbeats_fixed": db_fixed,
        "basic_time_unit": float(fix["basic_time_unit"]),
        "bar_ranges": bars,
        "tempo_bpm": estimate_global_tempo(pmidi)
    }


def estimate_global_tempo(pmidi: pm.PrettyMIDI) -> float:
    """Simple: mean beat period → BPM."""
    beats = get_beats(pmidi)
    if beats.size < 2:
        return 120.0
    periods = np.diff(beats)
    return float(np.round(60.0 / np.mean(periods), 2))


# -------------------------
# Build melodic and drum SSMs for each NPZ
# -------------------------

# def prepare_one_song(npz_path: Path): #change later to this
def prepare_one_song(npz_path: Path, only_cache_cqt: bool = False):
    """
    - Load LPD NPZ with Pypianoroll
    - Force tempo to 120 QPM
    - Write 3 MIDI variants: all-tracks / no-drum / drum-only
    - Render no-drum WAV
    - Build bar grid in seconds from downbeats
    - Compute pooled CQT per bar (84 x 96)
    - Compute melodic SSM from CQT bars (Euclidean)
    - Compute drum SSM from symbolic drum bars (Euclidean)
    - Save both padded to 256 x 256
    """
    # ---- skip if both pickles already exist ----
    stem = npz_path.stem
    fname_mel = OUT_MEL_SSM / f"song_barlv_ssm_{stem}.pkl"
    fname_drm = OUT_DRUM_SSM / f"song_barlv_drum_ssm_{stem}.pkl"
    if fname_mel.exists() and fname_drm.exists() and not only_cache_cqt:
        # still clean up intermediates to avoid clutter
        if not KEEP_WAV:
            safe_unlink(OUT_WAV_ND   / f"{stem}_no_drum.wav")
        if not KEEP_MIDI:
            safe_unlink(OUT_MIDI_ALL / f"{stem}_all_tracks.mid")
            safe_unlink(OUT_MIDI_ND  / f"{stem}_no_drum.mid")
            safe_unlink(OUT_MIDI_DO  / f"{stem}_drum_only.mid")
        prune_empty_dirs(OUT_WAV_ND, OUT_MIDI_ALL, OUT_MIDI_ND, OUT_MIDI_DO)

        print(f"[prep] (skip) {npz_path.name} -> SSMs exist; cleaned leftover WAV/MID")
        return {
            "stem": stem,
            "midi_all": "",
            "midi_nd": "",
            "midi_do": "",
            "wav_nd": "",
            "bars": None
        }
    # -------------------------------------------------

    mt = load_multitrack(npz_path) #lturn NPZ into Multitrack

    # --- checking start ---
    print(f"\n[dbg] === {npz_path.name} ===")
    # 1) Which resolution attribute exists?
    res_attr = "beat_resolution" if hasattr(mt, "beat_resolution") else (
            "resolution"       if hasattr(mt, "resolution")       else None)

    if res_attr is None:
        # Bail out early if the file is malformed and has no resolution info
        raise AttributeError("Multitrack missing both 'beat_resolution' and 'resolution'")

    resolution = int(getattr(mt, res_attr))
    steps_per_bar = resolution * 4
    print(f"[dbg] steps_per_quarter={resolution}  steps_per_bar={steps_per_bar}")

    # assert res_attr is not None, "Multitrack has neither beat_resolution nor resolution."

    resolution = int(getattr(mt, res_attr))
    steps_per_bar = resolution * 4
    print(f"[dbg] steps_per_quarter={resolution}  steps_per_bar={steps_per_bar}")
    if steps_per_bar != 96:
        print(f"[warn] steps_per_bar={steps_per_bar} != 96 (paper uses 96). "
            f"Your slice_bars() will pad/trim per bar.")

    # 2) Basic timeline length
    T = max((tr.pianoroll.shape[0] for tr in mt.tracks if tr.pianoroll is not None), default=0)
    # print(f"[dbg] T (max time steps across tracks) = {T}")
    # end: end of check


    # ## Fixed downbeat calculation -- is it really ok to delete this...?
    # pmidi = mt.to_pretty_midi()

    # # 3a) get raw downbeats (or synthesize from beats if missing), then fix by note density
    # downbeats_raw_sec = get_downbeats_raw(pmidi)
    # if len(downbeats_raw_sec) < 2:
    #     print("[warn] not enough downbeats; skipping song.")
    #     return None

    # fix = fix_downbeats_by_note_density(pmidi, downbeats_raw_sec, bar_portion=BAR_STEPS)
    # downbeats_sec = fix["downbeats_fixed"]                  # seconds
    # if downbeats_sec.size < 2:
    #     print("[warn] fixed downbeats too few; skipping song.")
    #     return None

    # # 3b) convert those seconds to step indices (because your drum pianoroll is in steps)
    # sec_per_quarter = 60.0 / TEMPO_QPM                      # normalized tempo
    # sec_per_step    = sec_per_quarter / resolution          # e.g., 0.5/24 ≈ 0.020833 s
    # bar_edges_steps = np.clip(
    #     np.round(downbeats_sec / sec_per_step).astype(int),
    #     0, max(0, T-1)
    # ).tolist()

    # resolution = int(mt.beat_resolution)      # steps per quarter (LPD default 24 fits 96 per bar)  :contentReference[oaicite:4]{index=4}

    # LPD-5 has 5 merged tracks: Drums, Piano, Guitar, Bass, Strings  :contentReference[oaicite:5]{index=5}
    # Find drum track index via is_drum
    drum_idx = None
    for i, tr in enumerate(mt.tracks): #get is_drum index
        if getattr(tr, "is_drum", False):
            drum_idx = i; break
    if drum_idx is None:
        # No drum track? skip
        return None

    # Normalize tempo to 120 QPM so bar timing is consistent in audio (§3.1)
    mt = normalize_tempo_to_120(mt)
    log_mt("post-normalize", mt)

    ## --- Compute downbeats from PrettyMIDI (robust), then correct by note density ---
    # Important: do this AFTER normalize_tempo_to_120 so pmidi aligns with 120 QPM
    pmidi = mt.to_pretty_midi()

    downbeats_raw_sec = get_downbeats_raw(pmidi)                # either true downbeats or synthesized from beats
    if len(downbeats_raw_sec) < 2:
        return None  # still unusable

    fix = fix_downbeats_by_note_density(pmidi, downbeats_raw_sec, bar_portion=BAR_STEPS)
    downbeats_sec = fix["downbeats_fixed"].tolist()
    if len(downbeats_sec) < 2:
        return None

    # Keep seconds-based edges for audio CQT pooling
    bar_edges_sec = downbeats_sec

    # Also convert these seconds to step indices for slicing the symbolic drum pianoroll
    # Because we normalized tempo to constant 120 QPM, step duration is fixed:
    sec_per_quarter = 60.0 / TEMPO_QPM                 # 0.5 s at 120 QPM
    sec_per_step    = sec_per_quarter / resolution     # e.g., 0.5 / 24 ≈ 0.020833 s/step
    bar_edges_steps = [int(round(t / sec_per_step)) for t in downbeats_sec]
    ## end: end of downbeat calculation

    # make 1-D boolean downbeat of length T (after normalization)
    T = multitrack_max_length_steps(mt)
    db = np.zeros(T, dtype=np.bool_)
    idx = np.asarray(bar_edges_steps, dtype=int)
    idx = idx[(idx >= 0) & (idx < T)]
    db[idx] = True
    mt.downbeat = db

    # handy alias used later
    bar_edges_sec = downbeats_sec


    # Normalize shapes so pypianoroll.write() won't complain
    mt = fix_multitrack_for_write(mt, bar_edges_steps, TEMPO_QPM)

    # --- Write three MIDI variants (single, safe block) ---
    # --- No-drum audio, rendered in-memory (no temp MIDI/WAV) ---
    stem = npz_path.stem

    # cache check
    cqt_npy = OUT_CQT_POOL / f"{stem}_bars_cqt.npy"
    bars_cqt = None
    if cqt_npy.exists():
        try:
            bars_cqt = np.load(cqt_npy)  # shape (B, 84, 96)
            B = int(bars_cqt.shape[0])
            if only_cache_cqt:
                # nothing to do; cache already there
                return {"stem": stem, "bars": B, "cached_only": True, "cache_hit": True}
        except Exception:
            bars_cqt = None  # fall back to recompute if file is corrupt

    # Build a no-drum copy in memory
    mt_nd = mt.copy()
    mt_nd.tracks[drum_idx].pianoroll[:] = 0
    # keep tempo/downbeat as 1-D arrays on the copy
    mt_nd.tempo    = mt.tempo.copy()
    mt_nd.downbeat = mt.downbeat.copy()
    assert_mt_ok(mt_nd, "pre-write:no-drum")

    pm_nd = mt_nd.to_pretty_midi()
    midi_nd = None
    wav_nd  = None

    if bars_cqt is None:
        # ------- Build bar grid (needed to pool CQT per bar) -------
        song_bar_grid_range_list = []
        for b in range(len(bar_edges_sec) - 1):
            t0, t1 = bar_edges_sec[b], bar_edges_sec[b+1]
            time_grid = np.linspace(t0, t1, BAR_STEPS + 1)
            bar_grid = np.stack([time_grid[:-1], time_grid[1:]], axis=1)
            half = (t1 - t0) / BAR_STEPS * 0.5
            bar_grid = np.clip(bar_grid - half, 0.0, None)
            song_bar_grid_range_list.append(bar_grid)

        # ------- Render no-drum audio (in-RAM if possible) -------
        try:
            y = pm_nd.fluidsynth(fs=SR, sf2_path=str(SOUND_FONT)).astype(np.float32)
        except Exception as e:
            print(f"[warn] fluidsynth in-memory failed: {e} — falling back to disk render.")
            midi_nd = OUT_MIDI_ND / f"{stem}_no_drum.mid"
            ensure_dir(midi_nd)
            ppr.write(mt_nd, str(midi_nd))
            wav_nd = OUT_WAV_ND / f"{stem}_no_drum.wav"
            syn_midi_to_wav(midi_nd, wav_nd, sr=SR)
            y, _ = librosa.load(str(wav_nd), sr=SR, mono=True)

        if y.ndim == 2:
            y = y.mean(axis=1)
        y = y.astype(np.float32, copy=False)
         # ---- guard: empty or silent after removing drums → skip ----
        if y.size == 0 or float(np.max(np.abs(y))) == 0.0:
            msg = f"[skip] {stem}: no-drum audio empty/silent; skipping CQT."
            print(msg)
            if only_cache_cqt:
                # cache-only pass → just report and bail
                prune_empty_dirs(OUT_WAV_ND, OUT_MIDI_ALL, OUT_MIDI_ND, OUT_MIDI_DO)
                return {"stem": stem, "bars": 0, "cached_only": True, "skipped": "silent_nodrum"}
            else:
                # full pipeline → we can’t build a meaningful Mel-SSM; skip this song
                return None

        # ------- CQT -------
        dur = len(y) / SR
        try:
            cqt = librosa.cqt(y, sr=SR, hop_length=HOP, n_bins=N_BINS, bins_per_octave=12)
        except librosa.util.exceptions.ParameterError as e:
            print(f"[skip] {stem}: CQT failed ({e}); skipping.")
            if only_cache_cqt:
                prune_empty_dirs(OUT_WAV_ND, OUT_MIDI_ALL, OUT_MIDI_ND, OUT_MIDI_DO)
                return {"stem": stem, "bars": 0, "cached_only": True, "skipped": "cqt_too_short"}
            else:
                return None
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        cqt_db = np.clip(np.nan_to_num(cqt_db, neginf=-120.0, posinf=0.0), -120.0, 0.0).astype(np.float32)

        # ------- Pool per bar (→ bars_cqt: (B, 84, 96)) -------
        n_frames = cqt_db.shape[1]
        fps = n_frames / max(1e-9, dur)
        bar_grids = np.stack(song_bar_grid_range_list, axis=0)  # (B, 96, 2)
        f0 = np.floor(bar_grids[..., 0] * fps).astype(np.int64)
        f1 = np.ceil( bar_grids[..., 1] * fps).astype(np.int64)
        f0 = np.clip(f0, 0, n_frames - 1)
        f1 = np.clip(f1, 1, n_frames)
        mask = f1 <= f0
        f1[mask] = np.minimum(f0[mask] + 1, n_frames)

        # cum = np.concatenate([np.zeros((N_BINS, 1), dtype=np.float32),
        #                     np.cumsum(cqt_db, axis=1)], axis=1)
        # sums = np.take(cum, f1 + 1, axis=1) - np.take(cum, f0 + 1, axis=1)  # (84, B, 96)
        # NEW (correct):
        cum = np.concatenate([np.zeros((N_BINS, 1), dtype=np.float32),
                            np.cumsum(cqt_db, axis=1)], axis=1)  # shape (84, n_frames+1)
        # Use [a,b) sum = cum[:, b] - cum[:, a]; a ∈ [0,n_frames-1], b ∈ [1,n_frames]
        sums = np.take(cum, f1, axis=1) - np.take(cum, f0, axis=1)
        means = sums / (f1 - f0)[None, :, :].astype(np.float32)
        bars_cqt = np.transpose(means, (1, 0, 2)).astype(np.float32)
        bars_cqt = np.nan_to_num(bars_cqt, nan=0.0, posinf=0.0, neginf=0.0)
        B = int(bars_cqt.shape[0])

        # ------- Save cache atomically -------
        OUT_CQT_POOL.mkdir(parents=True, exist_ok=True)
        tmp = cqt_npy.with_suffix(".tmp.npy")
        np.save(tmp, bars_cqt)
        os.replace(tmp, cqt_npy)

        # Optional cleanup if you created temp files
        if not KEEP_MIDI and midi_nd is not None: safe_unlink(midi_nd)
        if not KEEP_WAV  and wav_nd  is not None: safe_unlink(wav_nd)

    # If you ran with --only_cache_cqt, exit now (cache was created above)
    if only_cache_cqt:
        prune_empty_dirs(OUT_WAV_ND, OUT_MIDI_ALL, OUT_MIDI_ND, OUT_MIDI_DO)
        return {"stem": stem, "bars": int(bars_cqt.shape[0]), "cached_only": True}
    
    # If we're called just to backfill CQT cache, stop here
    # if only_cache_cqt:
    #     # clean up even on cache-only pass
    #     if not KEEP_WAV:
    #         safe_unlink(wav_nd)
    #     if not KEEP_MIDI:
    #         #safe_unlink(midi_all)
    #         safe_unlink(midi_nd)
    #         #safe_unlink(midi_do)
    #     prune_empty_dirs(OUT_WAV_ND, OUT_MIDI_ALL, OUT_MIDI_ND, OUT_MIDI_DO)
    #     return {"stem": stem, "bars": int(B), "cached_only": True}

    # if only_cache_cqt:
    #     if not KEEP_WAV and wav_nd is not None:
    #         safe_unlink(wav_nd)
    #     if not KEEP_MIDI and midi_nd is not None:
    #         safe_unlink(midi_nd)
    #     prune_empty_dirs(OUT_WAV_ND, OUT_MIDI_ALL, OUT_MIDI_ND, OUT_MIDI_DO)
    #     return {"stem": stem, "bars": int(B), "cached_only": True}

    # ------- Melodic SSM from CQT bars (Euclidean) -------
    # mel_ssm = pairwise_euclidean_bar_ssm(bars_cqt)   # melodic SSM (B x B) sec 3.1
    # mel_ssm = minmax01(pad_to_256(mel_ssm)) # return max number, convert it to 256x256
    # -- changed for wei-style nan and inf handling
    mel_ssm = pairwise_euclidean_bar_ssm(bars_cqt)
    mel_ssm = np.nan_to_num(mel_ssm, nan=0.0, posinf=0.0, neginf=0.0)
    mel_ssm = minmax01(pad_to_256(mel_ssm))

    # ------- Drum SSM from symbolic drum bars (Euclidean) -------
    drum_roll_bin = (mt.tracks[drum_idx].pianoroll > 0).astype(np.float32)

    drum_bars_list = slice_bars(drum_roll_bin, bar_edges_steps, steps_per_bar=BAR_STEPS)  # list of (128,96)

    # Defensive: ensure all bars are exactly (128,96) before stacking
    for i, b in enumerate(drum_bars_list):
        if b.shape != (128, BAR_STEPS):
            raise ValueError(f"drum bar[{i}] shape {b.shape} != (128,{BAR_STEPS})")

    drum_bars = np.stack(drum_bars_list, axis=0).astype(np.float32)  # (B,128,96)
    np.nan_to_num(drum_bars, copy=False)  # one-time sanitize

    drum_ssm = pairwise_euclidean_bar_ssm(drum_bars)  # (B,B)
    # If you've never seen NaNs here, you can drop the next line:
    # np.nan_to_num(drum_ssm, copy=False)

    drum_ssm = minmax01(pad_to_256(drum_ssm))

    # ------- Save pickles to the same names our trainer expects -------
    # base = stem  # we’ll keep the raw stem
    # fname_mel = OUT_MEL_SSM / f"song_barlv_ssm_{base}.pkl"
    # fname_drm = OUT_DRUM_SSM / f"song_barlv_drum_ssm_{base}.pkl"

    # before:
    # ensure_dir(fname_mel); ensure_dir(fname_drm)
    # with open(fname_mel, "wb") as f: pickle.dump(mel_ssm, f)
    # with open(fname_drm, "wb") as f: pickle.dump(drum_ssm, f)
    atomic_pickle_dump(mel_ssm, fname_mel)
    atomic_pickle_dump(drum_ssm, fname_drm)

    # # --- CLEAN UP intermediates ASAP to save disk ---
    # if not KEEP_WAV:
    #     safe_unlink(wav_nd)  # delete rendered WAV once we've extracted features

    # if not KEEP_MIDI:
    #     #safe_unlink(midi_all)
    #     safe_unlink(midi_nd)
    #     #safe_unlink(midi_do)

    # --- CLEAN UP intermediates ---
    if not KEEP_WAV and wav_nd is not None:
        safe_unlink(wav_nd)
    if not KEEP_MIDI and midi_nd is not None:
        safe_unlink(midi_nd)
    prune_empty_dirs(OUT_WAV_ND, OUT_MIDI_ALL, OUT_MIDI_ND, OUT_MIDI_DO)

    # Build return info without referencing undefined paths
    ret = {"stem": stem, "bars": len(drum_bars)}
    if midi_nd is not None:
        ret["midi_nd"] = str(midi_nd)
    if wav_nd is not None:
        ret["wav_nd"] = str(wav_nd)
    return ret

def prepare_dataset(limit=None, cache_limit = None):

    if OUT_PRE is None or OUT_MEL_SSM is None or OUT_DRUM_SSM is None \
       or OUT_MIDI_ALL is None or OUT_MIDI_ND is None or OUT_MIDI_DO is None \
       or OUT_WAV_ND is None:
        configure_paths()
    # ... existing code ...

    #create output folders
    OUT_MIDI_ALL.mkdir(parents=True, exist_ok=True)
    OUT_MIDI_ND.mkdir(parents=True, exist_ok=True)
    OUT_MIDI_DO.mkdir(parents=True, exist_ok=True)
    OUT_WAV_ND.mkdir(parents=True, exist_ok=True)
    OUT_CQT_POOL.mkdir(parents=True, exist_ok=True)
    OUT_MEL_SSM.mkdir(parents=True, exist_ok=True)
    OUT_DRUM_SSM.mkdir(parents=True, exist_ok=True)

    # print("[dbg] first_10_npz =", list_npz_files(DATASET_ROOT)[:10]) #checking: list firrst few npz files found under DATASET_ROOT & print total count
    #finds all npz files
    npz_files = list_npz_files(DATASET_ROOT)
    
    # ---- skip stems that already have both pickles ----
    have_mel   = {p.stem.replace("song_barlv_ssm_", "") for p in OUT_MEL_SSM.glob("*.pkl")}
    have_drm   = {p.stem.replace("song_barlv_drum_ssm_", "") for p in OUT_DRUM_SSM.glob("*.pkl")}
    already    = have_mel & have_drm

    def has_cqt(stem: str) -> bool:
        return (OUT_CQT_POOL / f"{stem}_bars_cqt.npy").exists()
    
    # NEW: clean intermediates even for stems we won't process this run
    already_with_cache = {s for s in already if has_cqt(s)}
    cleanup_intermediates_for_stems(already_with_cache)

    # What needs doing?
    need_full  = [p for p in npz_files if p.stem not in already]                    # no pickles yet
    need_cache = [p for p in npz_files if (p.stem in already) and (not has_cqt(p.stem))]  # pickles exist, cache missing

    # ---- pass 1: scan drum-onset counts across BOTH sets; compute μ/σ; keep μ ± kσ ----
    pool_for_stats = need_full + need_cache
    counts_map, counts = {}, []
    for p in pool_for_stats:
        c = drum_onset_count_from_npz(p)
        if c is not None:
            counts_map[p] = c
            counts.append(c)

    if len(counts) == 0:
        print("[clean] No songs with detectable drum track in this subset; nothing to prepare.")
        need_full, need_cache = [], []
    else:
        mu  = float(np.mean(counts))
        sig = float(np.std(counts))
        lo  = max(MIN_DRUM_ONSETS, int(np.floor(mu - DRUM_OUTLIER_SIGMA * sig)))
        hi  = int(np.ceil(mu + DRUM_OUTLIER_SIGMA * sig))
        before_full, before_cache = len(need_full), len(need_cache)
        need_full  = [p for p in need_full  if (p in counts_map and lo <= counts_map[p] <= hi)]
        need_cache = [p for p in need_cache if (p in counts_map and lo <= counts_map[p] <= hi)]
        print(f"[clean] μ={mu:.1f} σ={sig:.1f} keep[{lo},{hi}] → "
              f"full {len(need_full)}/{before_full}, cache {len(need_cache)}/{before_cache}")

    # ---- honor --limit: fill FULL first, then CACHE with leftover ----
    if limit is not None:
        limit_full = min(limit, len(need_full))
        need_full  = need_full[:limit_full]
        left = max(0, limit - limit_full)
        need_cache = need_cache[:left]

    # ---- hard cap PASS B regardless of --limit ----
    if cache_limit is not None:
        need_cache = need_cache[:cache_limit]

    print(f"[prep] plan → PASS A (full): {len(need_full)} | PASS B (cache): {len(need_cache)}")

    meta = []
    n_ok = n_skip = n_err = 0
    # ---------- PASS A: full preparation ----------/
    if need_full:
        pbar = tqdm(need_full, desc="[prep] songs (full)", unit="song", dynamic_ncols=True)
        for p in pbar:
            try:
                info = prepare_one_song(p)  # full pipeline
                if info is not None:
                    meta.append(info); n_ok += 1
                    pbar.set_postfix_str(f"ok={n_ok} last={p.stem[-8:]}")
                else:
                    n_skip += 1
                    pbar.set_postfix_str(f"skipped={n_skip} last={p.stem[-8:]}")
            except Exception as e:
                n_err += 1
                print(f"\n[ERROR] stem={p.stem} during prepare_one_song")
                traceback.print_exc()
                pbar.set_postfix_str(f"ERROR({n_err})={str(e)[:40]}")
        pbar.close()

    # ---------- PASS B: cache-only backfill ----------
    MAX_CACHE = 20
    need_cache = need_cache[:MAX_CACHE]
    if need_cache:
        pbar = tqdm(need_cache, desc="[cache] backfill CQT", unit="song", dynamic_ncols=True)
        for p in pbar:
            try:
                # cache-only: create bars_cqt.npy without touching pickles
                info = prepare_one_song(p, only_cache_cqt=True)
                # meta entry optional; we can still log it
                meta.append(info or {"stem": p.stem, "cached_only": True})
                n_ok += 1
                pbar.set_postfix_str(f"ok={n_ok} last={p.stem[-8:]}")
            except Exception as e:
                n_err += 1
                print(f"\n[ERROR] stem={p.stem} during prepare_one_song")
                traceback.print_exc()
                pbar.set_postfix_str(f"ERROR({n_err})={str(e)[:40]}")
        pbar.close()
    
    # NEW: final sweep in case anything slipped through
    if not KEEP_WAV:
        for p in OUT_WAV_ND.glob("*.wav"):
            safe_unlink(p)
    if not KEEP_MIDI:
        for d in (OUT_MIDI_ALL, OUT_MIDI_ND, OUT_MIDI_DO):
            for p in d.glob("*.mid"):
                safe_unlink(p)
    prune_empty_dirs(OUT_WAV_ND, OUT_MIDI_ALL, OUT_MIDI_ND, OUT_MIDI_DO)

    if WRITE_META:
        ensure_dir(OUT_OBJ_PKL)
        with open(OUT_OBJ_PKL, "wb") as f:
            pickle.dump(meta, f)
        print(f"[prep] done. saved meta with {len(meta)} items.")
    else:
        print(f"[prep] done. processed {len(meta)} items (meta file skipped).")


# -------------------------
# Dataset & training (same as before, but paths fixed)
# -------------------------

class SSMTrainDataset(Dataset):
    """Pairs (melodic_SSM, drum_SSM), both (1,256,256) float32 in [0,1]."""
    def __init__(self, mel_dir: Path, drum_dir: Path):
        self.index = []
        mel_files = sorted(glob.glob(str(mel_dir / "song_barlv_ssm_*.pkl")))
        for p in mel_files:
            base = Path(p).stem.replace("song_barlv_ssm_", "")
            dpath = drum_dir / f"song_barlv_drum_ssm_{base}.pkl"
            if dpath.exists():
                self.index.append((p, str(dpath)))

    def __len__(self): return len(self.index)

    def __getitem__(self, i): #loads each mel_ssm, drum_SSM pickle -> torch shaped (1, 256, 256)
        mpath, dpath = self.index[i]
        with open(mpath, "rb") as f: mel = pickle.load(f)  # (256,256)
        with open(dpath, "rb") as f: drm = pickle.load(f)  # (256,256)
        mel = torch.from_numpy(mel[None, ...]).float()
        drm = torch.from_numpy(drm[None, ...]).float()
        # sanitize: replace NaN/Inf and clamp to [0,1]
        mel = torch.nan_to_num(mel, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        drm = torch.nan_to_num(drm, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        return mel, drm


def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

#bce = nn.BCELoss()
bce = nn.BCEWithLogitsLoss()

# -------------------------
# Helper Functions for Logging
# -------------------------

LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def setup_logger(name="train"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt); logger.addHandler(ch)
    fh = logging.FileHandler(LOG_DIR / "ssm_train.log")
    fh.setFormatter(fmt); logger.addHandler(fh)
    return logger

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.data
            total += float(g.norm(2).item()**2)
    return math.sqrt(total) if total > 0 else 0.0

def get_lr(optim):
    return optim.param_groups[0]["lr"]

def save_sample_png(epoch, stem, mel, drm, recon):
    """mel/drm/recon: torch tensors in shape (1,1,256,256) or numpy (256,256)"""
    if not _HAS_MPL: 
        return
    SAVE_DIR = PROJECT_DIR / "sample_pngs"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    def _to_np(x):
        if hasattr(x, "detach"):
            x = x.detach().cpu().squeeze().numpy()
        return x
    mel_np = _to_np(mel); drm_np = _to_np(drm); rec_np = _to_np(recon)
    diff = (rec_np - drm_np)
    vmax = 1.0; vmin = 0.0
    fig, axs = plt.subplots(1, 4, figsize=(14, 3))
    axs[0].imshow(mel_np, cmap="magma", vmin=vmin, vmax=vmax); axs[0].set_title("Mel-SSM")
    axs[1].imshow(drm_np, cmap="magma", vmin=vmin, vmax=vmax); axs[1].set_title("Drum-SSM (target)")
    axs[2].imshow(rec_np, cmap="magma", vmin=vmin, vmax=vmax); axs[2].set_title("Recon")
    axs[3].imshow(diff, cmap="bwr"); axs[3].set_title("Recon-Target")
    for ax in axs: ax.axis("off")
    fig.tight_layout()
    out = SAVE_DIR / f"e{epoch:03d}_{stem}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)


def train_ssm(limit=None, cache_limit=None):
    if OUT_PRE is None or OUT_MEL_SSM is None or OUT_DRUM_SSM is None \
       or OUT_MIDI_ALL is None or OUT_MIDI_ND is None or OUT_MIDI_DO is None \
       or OUT_WAV_ND is None:
        configure_paths()
    # print("[dbg] mel_pkls =", sum(1 for _ in OUT_MEL_SSM.glob("*.pkl")))    #checking: how many precomputed pickles (melodic SSMs) you alr have
    # print("[dbg] drm_pkls =", sum(1 for _ in OUT_DRUM_SSM.glob("*.pkl")))   #checking: how many precomputed pickles (drum SSMs) you alr have
    # If no SSMs yet, run preparation first

    prepare_dataset(limit=limit)
    
    # checking begin
    # --- preflight: what’s on disk? ---
    # print("[dbg] MEL dir:", OUT_MEL_SSM.resolve(), "exists:", OUT_MEL_SSM.exists())
    # print("[dbg] DRM dir:", OUT_DRUM_SSM.resolve(), "exists:", OUT_DRUM_SSM.exists())

    # mel_files  = sorted(OUT_MEL_SSM.glob("song_barlv_ssm_*.pkl"))
    # drum_files = sorted(OUT_DRUM_SSM.glob("song_barlv_drum_ssm_*.pkl"))
    # print("[dbg] #mel_pkls:", len(mel_files), "  #drum_pkls:", len(drum_files))
    # print("[dbg] mel examples:", [p.name for p in mel_files[:5]])
    # print("[dbg] drm examples:", [p.name for p in drum_files[:5]])

    # stem matching
    # mel_stems  = {p.stem.replace("song_barlv_ssm_", "") for p in mel_files}
    # drum_stems = {p.stem.replace("song_barlv_drum_ssm_", "") for p in drum_files}
    # common     = sorted(mel_stems & drum_stems)
    # only_mel   = sorted(mel_stems - drum_stems)
    # only_drum  = sorted(drum_stems - mel_stems)
    # print("[dbg] #matched stems:", len(common))
    # print("[dbg] first matched stem:", (common[0] if common else None))
    # print("[dbg] mel-without-drum (up to 5):", only_mel[:5])
    # print("[dbg] drum-without-mel (up to 5):", only_drum[:5])

    # # try loading one matched pair
    # if common:
    #     stem = common[0]
    #     import pickle, numpy as np, os
    #     mpath = OUT_MEL_SSM  / f"song_barlv_ssm_{stem}.pkl"
    #     dpath = OUT_DRUM_SSM / f"song_barlv_drum_ssm_{stem }.pkl"
    #     with open(mpath, "rb") as f: mel = pickle.load(f)
    #     with open(dpath, "rb") as f: drm = pickle.load(f)
    #     mel = np.asarray(mel); drm = np.asarray(drm)
    #     print("[dbg] mel shape:", mel.shape, "range:", (mel.min(), mel.max()))
    #     print("[dbg] drm shape:", drm.shape, "range:", (drm.min(), drm.max()))
    # checking end

    dataset = SSMTrainDataset(OUT_MEL_SSM, OUT_DRUM_SSM) #instantiate SSM Train Datatset
    assert len(dataset) > 0, "No paired SSM samples found."

    #train test split
    n_total = len(dataset)
    n_train = int(0.9 * n_total)
    n_val   = n_total - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))

    # --- after you compute train_set, val_set ---
    train_bs = max(2, min(BATCH_SIZE, len(train_set)))
    val_bs   = max(1, min(BATCH_SIZE, len(val_set)))  # avoid 0

    train_loader = DataLoader(train_set, batch_size=train_bs, shuffle=True, num_workers=2, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=val_bs, shuffle=False, num_workers=2, drop_last=False)

    # === DEBUG: dataset/batch sizes & one-batch probe ===
    total_pairs = len(train_set) + len(val_set)
    print(f"[dbg] after cleaning/pairing: total={total_pairs}  → train={len(train_set)}  val={len(val_set)}")
    print(f"[dbg] batch sizes: train_bs={train_bs}  val_bs={val_bs}  drop_last(val)=False")

    # quick probe: does val yield a batch at all?
    try:
        _mel, _drm = next(iter(val_loader))
        print(f"[dbg] val probe batch shapes: mel{tuple(_mel.shape)}  drm{tuple(_drm.shape)}")
        print(f"[dbg] val probe finite? mel={torch.isfinite(_mel).all().item()} drm={torch.isfinite(_drm).all().item()}")
    except StopIteration:
        print("[dbg] val probe: NO BATCHES")
    # --- end: end of debug

    # --- debug: dataset sizes & batch sizes ---
    total_pairs = len(train_set) + len(val_set)
    print(f"[dbg] after cleaning/pairing: total={total_pairs}  → train={len(train_set)}  val={len(val_set)}")
    print(f"[dbg] batch sizes: train_bs={train_bs}  val_bs={val_bs}  drop_last(val)=False")

    # probe whether validation can yield at least one batch
    try:
        _mel, _drm = next(iter(val_loader))
        print(f"[dbg] val probe batch: mel{tuple(_mel.shape)}  drm{tuple(_drm.shape)}")
    except StopIteration:
        print("[dbg] val probe: NO BATCHES")
    # end : end of debugging val nan error

    # Models (as in §4.4: 8 conv + 3 FC (+skip), 32-d latent)
    enc = SSMEncoder(in_channels=1, base_channels=64, latent_dim=32)
    dec = SSMDecoder(out_channels=1, base_channels=64, latent_dim=32)
    vae = SSMVAE(enc, dec).to(DEVICE)
    dis = SSMDiscriminator(in_channels=1, base_channels=64).to(DEVICE)

    #create adam optimizers for VAE and 
    optG = torch.optim.Adam(vae.parameters(), lr=LR_GEN, betas=(BETA1, BETA2))
    optD = torch.optim.Adam(dis.parameters(), lr=LR_DIS, betas=(BETA1, BETA2))

    # --- checking ---
    logger = setup_logger("train")
    csv_path = LOG_DIR / "ssm_metrics.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch","split","G","D","rec","kl","gan","D_real","D_fake","lrG","lrD","gradG","gradD","secs"])

    logger.info(f"dataset size: {len(dataset)}  (train: {len(train_set)}, val: {len(val_set)})")
    logger.info(f"enc params: {count_params(enc):,}  dec params: {count_params(dec):,}  dis params: {count_params(dis):,}")
    # quick forward shape sanity
    mel0, drm0 = next(iter(train_loader))
    mel0 = mel0.to(DEVICE)

    vae.eval()
    with torch.no_grad():
        recon0, mu0, logvar0 = vae(mel0[:1])  # can keep 1 sample now
    logger.info(
        f"sanity forward: mel {tuple(mel0.shape)} -> recon {tuple(recon0.shape)}  "
        f"mu {tuple(mu0.shape)} logvar {tuple(logvar0.shape)}"
    )
    # end: end of check

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    # progress bar over epochs
    epbar = tqdm(range(1, NUM_EPOCHS + 1), desc="[epochs]", unit="ep", dynamic_ncols=True)
    for epoch in epbar:  # repeat NUM_EPOCHS times
        t0 = time.time()
        vae.train(); dis.train()  # train vae model and discriminator
        # Linear β warmup for KL (0 -> 1 over KL_WARMUP_EPOCHS)
        current_beta = min(1.0, float(epoch) / float(KL_WARMUP_EPOCHS))

        # Two-stage: no GAN for early epochs, then enable
        current_lambda_gan = (LAMBDA_GAN if epoch >= GAN_START_EPOCH else 0.0)
        # total_G, total_D = 0.0, 0.0 
        from collections import defaultdict
        total = defaultdict(float)
        num_batches = 0

        # progress bar over training batches
        pbar = tqdm(
            train_loader,
            desc=f"[train] epoch {epoch}/{NUM_EPOCHS}",
            unit="batch", leave=False, dynamic_ncols=True
        )

        for mel, drm in pbar:  # for every (mel ssm, drum ssm) pair,
            if mel.size(0) < 2:
                # Avoid BatchNorm crash on tiny batch
                continue
            num_batches += 1
            mel, drm = mel.to(DEVICE), drm.to(DEVICE)  # move to GPU/CPU as needed

            # --- Train D (Training discriminator) --- Teaching judge to tell real vs fake
            if current_lambda_gan > 0.0:
                optD.zero_grad(set_to_none=True)  # zero the gradient buffers
                with torch.no_grad():  # don't want to update generator while making fakes - no tracking for backprop needed
                    recon, _, _ = vae(mel)  # fake drum SSM made by us - reconstruction loss
                pred_real = dis(drm)         # get D's score for the actual drum (should be 1)
                pred_fake = dis(recon.detach())  # get D's score for the fake drum made by vae generator(should be 0)
                d_loss = 0.5 * (bce(pred_real, torch.ones_like(pred_real)) +
                                bce(pred_fake, torch.zeros_like(pred_fake)))    # BCE Loss on real & fake -> ipldd
                if torch.isfinite(d_loss):
                    d_loss.backward()  # backprop according to this loss^
                    gnD = grad_norm(dis)
                    optD.step()  # next step?
            else:
                # skip D entirely during VAE pretrain
                d_loss = torch.tensor(0.0, device=DEVICE)
                pred_real = torch.zeros((drm.size(0),1), device=DEVICE)  # dummies for logging
                pred_fake = torch.zeros((drm.size(0),1), device=DEVICE)
                gnD = 0.0

            # --- Train G (Training VAE Generator) ---
            optG.zero_grad(set_to_none=True)        # zero the gradient buffers
            recon, mu, logvar = vae(mel)            # get vae's output for melody input - make fake drum ssm, this time with grads!
            loss_rec = F.mse_loss(recon, drm)       # recon loss (mse): helps predicted drum SSM be close to real drum ssm,
            loss_kl  = kl_divergence(mu, logvar)    # VAE regularizer - keeps latent space nice & gaussian
            pred_fake_for_G = dis(recon)            # discriminator's prediction for the fake output
            loss_gan = bce(pred_fake_for_G, torch.ones_like(pred_fake_for_G))
            g_loss = LAMBDA_REC*loss_rec + current_beta*loss_kl + current_lambda_gan*loss_gan  # combine losses to get total loss for fooling discriminator
            if not torch.isfinite(g_loss):
                # logger.info("NaN/Inf in g_loss — skipping batch")
                continue
            g_loss.backward()                       # run backprop
            gnG = grad_norm(vae)
            optG.step()

            # keep running totals so u can print avg generator/discriminator losses later
            total["G"]     += g_loss.item()
            total["D"]     += d_loss.item()
            total["rec"]   += loss_rec.item()
            total["kl"]    += loss_kl.item()
            total["gan"]   += loss_gan.item()
            total["Dreal"] += torch.sigmoid(pred_real).mean().item()
            total["Dfake"] += torch.sigmoid(pred_fake).mean().item()
            total["gnG"]   += gnG
            total["gnD"]   += gnD

            if num_batches % 100 == 0:
                logger.info(f"[ep{epoch:03d} it{num_batches:05d}] "
                            f"G={total['G']/num_batches:.4f} D={total['D']/num_batches:.4f} "
                            f"rec={total['rec']/num_batches:.4f} kl={total['kl']/num_batches:.4f} gan={total['gan']/num_batches:.4f} "
                            f"D(real)={total['Dreal']/num_batches:.3f} D(fake)={total['Dfake']/num_batches:.3f} "
                            f"||∇G||={total['gnG']/num_batches:.2f} ||∇D||={total['gnD']/num_batches:.2f}")

            # live postfix on the tqdm bar
            pbar.set_postfix({
                "G":     f"{total['G']/num_batches:.3f}",
                "D":     f"{total['D']/num_batches:.3f}",
                "rec":   f"{total['rec']/num_batches:.3f}",
                "kl":    f"{total['kl']/num_batches:.3f}",
                "gan":   f"{total['gan']/num_batches:.3f}",
                "Dr":    f"{total['Dreal']/num_batches:.2f}",
                "Df":    f"{total['Dfake']/num_batches:.2f}",
                "||∇G||": f"{total['gnG']/num_batches:.2f}",
                "||∇D||": f"{total['gnD']/num_batches:.2f}",
            })

        pbar.close()

        # validation
        vae.eval(); dis.eval()
        from collections import defaultdict
        val = defaultdict(float)
        # with torch.no_grad():  # don't track gradients
        #     nvb = 0
        #     if len(val_loader) > 0:
        #         vbar = tqdm(
        #             val_loader,
        #             desc=f"[val]   epoch {epoch}/{NUM_EPOCHS}",
        #             unit="batch", leave=False, dynamic_ncols=True
        #         )
        #         for mel, drm in vbar:  # for each validation batch
        #             nvb += 1
        #             mel, drm = mel.to(DEVICE), drm.to(DEVICE) 
        #             recon, mu, logvar = vae(mel)
        #             loss_rec = F.mse_loss(recon, drm)
        #             loss_kl  = kl_divergence(mu, logvar)
        #             pred_fake = dis(recon)          # logits
        #             loss_gan = bce(pred_fake, torch.ones_like(pred_fake))
        #             g_loss = LAMBDA_REC*loss_rec + LAMBDA_KL*loss_kl + LAMBDA_GAN*loss_gan
        #             val["G"]     += g_loss.item()
        #             val["rec"]   += loss_rec.item()
        #             val["kl"]    += loss_kl.item()
        #             val["gan"]   += loss_gan.item()
        #             val["Dfake"] += torch.sigmoid(pred_fake).mean().item()
        #             vbar.set_postfix({"ValG": f"{val['G']/nvb:.3f}"})
        #         vbar.close()

        ## rec + B*kl only, GAN omitted
        # if you use a KL warmup in training, reuse the current beta; otherwise fall back to LAMBDA_KL
        beta_val = current_beta

        with torch.no_grad():
            nvb = 0
            good = 0
            bad_inp = bad_fwd = bad_loss = 0
            for mel, drm in val_loader:
                nvb += 1
                if not torch.isfinite(mel).all() or not torch.isfinite(drm).all():
                    bad_inp += 1
                    continue
                mel, drm = mel.to(DEVICE), drm.to(DEVICE)

                recon, mu, logvar = vae(mel)
                if (not torch.isfinite(recon).all() or
                    not torch.isfinite(mu).all() or
                    not torch.isfinite(logvar).all()):
                    bad_fwd += 1
                    continue

                loss_rec = F.mse_loss(recon, drm, reduction='mean')
                loss_kl  = kl_divergence(mu, logvar)
                g_loss   = LAMBDA_REC * loss_rec + LAMBDA_KL * loss_kl

                if (not torch.isfinite(g_loss) or
                    not torch.isfinite(loss_rec) or
                    not torch.isfinite(loss_kl)):
                    bad_loss += 1
                    continue

                # accumulate only for good batches
                good += 1
                val["G"]   += g_loss.item()
                val["rec"] += loss_rec.item()
                val["kl"]  += loss_kl.item()
                val["gan"] += 0.0
                val["Dfake"] += 0.0

        print(f"[dbg] validation batches processed (nvb) = {nvb}")    
        for k in list(val.keys()):
            val[k] /= max(1, good)   # average over good batches
        print(f"[val dbg] nvb={nvb} bad_inp={bad_inp} bad_fwd={bad_fwd} bad_loss={bad_loss} good={good}")
        # end: validation end   

        print(f"[val dbg] nvb={nvb} bad_inp={bad_inp} bad_fwd={bad_fwd} bad_loss={bad_loss}") 
        
        secs  = time.time() - t0
        avgG  = total["G"]   / max(1, num_batches)
        avgD  = total["D"]   / max(1, num_batches)
        avgRec= total["rec"] / max(1, num_batches)
        avgKl = total["kl"]  / max(1, num_batches)
        avgGan= total["gan"] / max(1, num_batches)
        mDr   = total["Dreal"]/ max(1, num_batches)
        mDf   = total["Dfake"]/ max(1, num_batches)
        gnGm  = total["gnG"]  / max(1, num_batches)
        gnDm  = total["gnD"]  / max(1, num_batches)
        lrG, lrD = get_lr(optG), get_lr(optD)

        logger.info(f"[Epoch {epoch:03d}] "
                    f"G:{avgG:.4f} D:{avgD:.4f} (rec {avgRec:.4f} kl {avgKl:.4f} gan {avgGan:.4f}) "
                    f"D(real):{mDr:.3f} D(fake):{mDf:.3f} "
                    f"ValG:{val.get('G', float('nan')):.4f}  lrG:{lrG:.1e} lrD:{lrD:.1e}  "
                    f"||∇G||:{gnGm:.2f} ||∇D||:{gnDm:.2f}  {secs:.1f}s")

        # update epoch bar summary
        epbar.set_postfix({
            "G":    f"{avgG:.3f}",
            "ValG": f"{val.get('G', float('nan')):.3f}",
            "secs": f"{secs:.1f}",
        })

        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch,"train",avgG,avgD,avgRec,avgKl,avgGan,mDr,mDf,lrG,lrD,gnGm,gnDm,secs])
            w.writerow([epoch,"val",val.get("G", float("nan")),"",val.get("rec", float("nan")),
                        val.get("kl", float("nan")),val.get("gan", float("nan")),"",
                        val.get("Dfake", float("nan")),lrG,lrD,"","",secs])

        # save a visual sample each epoch (if matplotlib available)
        try:
            if len(val_loader) > 0:
                mel_s, drm_s = next(iter(val_loader))
                mel_s, drm_s = mel_s.to(DEVICE), drm_s.to(DEVICE)
                with torch.no_grad():
                    recon_s, _, _ = vae(mel_s[:1])
                save_sample_png(epoch, f"samp{epoch:03d}", mel_s[:1], drm_s[:1], recon_s[:1])
        except Exception:
            pass

        # if validation generator loss improved, save checkpoint
        current_valG = val.get("G", float("inf"))
        if current_valG < best_val:
            best_val = current_valG
            torch.save({
                "epoch": epoch,
                "vae": vae.state_dict(),
                "dis": dis.state_dict(),
                "optG": optG.state_dict(),
                "optD": optD.state_dict(),
                "val": best_val
            }, CKPT_DIR / "best.pt")
            print("  ✓ Saved best checkpoint.")
    
    
    #save final weights at end of training, regardless of whether last epoch was best or not
    torch.save({
        "epoch": NUM_EPOCHS,
        "vae": vae.state_dict(),
        "dis": dis.state_dict(),
        "optG": optG.state_dict(),
        "optD": optD.state_dict()
    }, CKPT_DIR / "last.pt")
    print("Training complete.")
    #best.pt -> best validation performance during training
    #last.pt -> very last epoch's weights
    #these two files are saved at: /drum_generation/checkpoints/ssm_generator/

#change back
# if __name__ == "__main__":
#     train_ssm()

# delete later
if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="limit #songs to prepare")
    ap.add_argument("--epochs", type=int, default=None, help="override NUM_EPOCHS")
    ap.add_argument("--batch_size", type=int, default=None, help="override BATCH_SIZE")
    ap.add_argument("--dataset_root", type=str, default=None, help="override DATASET_ROOT")
    ap.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu'")

    # NEW: hyperparams for optimizers
    ap.add_argument("--lrG", type=float, default=None, help="Adam LR for generator (VAE)")
    ap.add_argument("--lrD", type=float, default=None, help="Adam LR for discriminator")
    ap.add_argument("--beta1", type=float, default=None, help="Adam beta1")
    ap.add_argument("--beta2", type=float, default=None, help="Adam beta2")
    ap.add_argument("--cache_limit", type=int, default=None,
                help="cap #songs to backfill CQT cache in PASS B")
    args = ap.parse_args()

    # Override globals if flags provided
    if args.dataset_root:
        DATASET_ROOT = Path(args.dataset_root)
    if args.epochs is not None:
        NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size
    if args.device:
        DEVICE = torch.device(args.device)

    if args.lrG is not None:
        LR_GEN = args.lrG
    if args.lrD is not None:
        LR_DIS = args.lrD
    if args.beta1 is not None:
        BETA1 = args.beta1
    if args.beta2 is not None:
        BETA2 = args.beta2
    
    configure_paths()
    print(f"[paths] OUT_PRE={OUT_PRE.resolve()}")
    print(f"[cfg] KEEP_WAV={KEEP_WAV} KEEP_MIDI={KEEP_MIDI}")


    # pass limit into prepare if needed
    # quick hack: set an env that prepare_dataset() can read
    os.environ["PREP_LIMIT"] = str(args.limit) if args.limit is not None else ""

    # small patch: change prepare_dataset(limit=None) call to:
    #   limit_env = os.environ.get("PREP_LIMIT")
    #   limit = int(limit_env) if limit_env else None
    #   prepare_dataset(limit=limit)

    #    train_ssm() 

    train_ssm(limit=args.limit, cache_limit=args.cache_limit) # delete

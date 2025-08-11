# train_ssm_generator.py
# End-to-end: prepare data from LPD-5 (via Pypianoroll) and train the SSM VAE-GAN (Wei et al., 2019)
# Paper refs: §3.1 (preprocessing, 96 steps/bar, zero-pad 256 bars), §3.2 (SSM VAE-GAN), §4.3 (train split),
# §4.4 (8 conv + 3 FC + skip, 32-dim latent)

import os
import glob
import math
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

# -------------------------
# Config (paths, seeds, hyperparams)
# -------------------------

PROJECT_DIR   = Path("/home/marg_intern/marg_intern_2025/yebinpyun/drum_generation")
DATASET_ROOT  = Path("/mnt/ssd2/marg_intern_2025_summer/yebinpyun")   # will scan **/*.npz
SOUND_FONT    = Path("/usr/share/sounds/sf2/FluidR3_GM.sf2")          # change to your SF2

OUT_PRE       = PROJECT_DIR / "pre_processed_data"
OUT_MIDI_ALL  = OUT_PRE / "proc_all_tracks_mid"
OUT_MIDI_ND   = OUT_PRE / "proc_no_drum_mid"
OUT_MIDI_DO   = OUT_PRE / "proc_drum_only_mid"
OUT_WAV_ND    = OUT_PRE / "proc_no_drum_wav"
OUT_OBJ_PKL   = OUT_PRE / "proc_midi_object.pkl"          # like step_1's object list (lightweight)
OUT_CQT_POOL  = OUT_PRE / "cqt_pooled_data"               # per-bar pooled CQT (84 x 96 per bar)
OUT_MEL_SSM   = OUT_PRE / "bar_level_cqt_ssm"             # melodic bar-level SSM (NxN)
OUT_DRUM_SSM  = OUT_PRE / "bar_level_drum_ssm"            # drum bar-level SSM (NxN)
CKPT_DIR      = PROJECT_DIR / "checkpoints" / "ssm_generator"

# training
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_EPOCHS = 50
LR_GEN = 2e-4
LR_DIS = 2e-4
BETA1, BETA2 = 0.5, 0.999
LAMBDA_REC = 1.0
LAMBDA_KL  = 1.0
LAMBDA_GAN = 1.0

# constants per paper
BAR_STEPS = 96            # 96 time steps per bar (§3.1)
TARGET_BARS = 256         # zero-pad songs to 256 bars (§3.1)
TEMPO_QPM = 120.0         # normalize tempo (§3.1)
SR = 44100
HOP = 256                 # CQT hop (matches your step_1)
N_BINS = 84               # pooled CQT freq bins (as in step_1)


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
    """Pad square SSM (N x N) to 256 x 256 (§3.1)."""
    n = ssm.shape[0]
    assert ssm.shape[0] == ssm.shape[1]
    if n == TARGET_BARS:
        return ssm.astype(np.float32)
    out = np.zeros((TARGET_BARS, TARGET_BARS), dtype=ssm.dtype)
    if fill_value is None:
        fill_value = float(np.max(ssm)) if n > 0 else 0.0
    out[:] = fill_value
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


# -------------------------
# LPD loading & bar slicing (Pypianoroll)
# -------------------------
# LPD NPZ -> pypianoroll.Multitrack (recommended I/O format)  :contentReference[oaicite:3]{index=3}

def load_multitrack(npz_path: Path) -> ppr.Multitrack:
    return ppr.load(npz_path)  # supports LPD NPZ format

def normalize_tempo_to_120(multitrack: ppr.Multitrack):
    """Set tempo array to 120 QPM uniformly (§3.1)."""
    T = multitrack.get_max_length()
    multitrack.tempo = np.full((T, 1), TEMPO_QPM, dtype=float)
    return multitrack

def get_downbeat_indices(multitrack: ppr.Multitrack):
    """Indices where downbeat is True/1 (start of each bar in LPD)."""
    db = multitrack.downbeat.squeeze() #flag per timestep - (1 at the start of every bar, 0 elsewhere). squeexe it to 1D
    return np.where(db > 0)[0].tolist()

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
        bar = track_roll[start:end, :].T  # (steps, 128) -> transpose later to (128, steps)
        # Ensure exact width (trim/pad)
        if bar.shape[0] != (BAR_STEPS):
            # If not exact (rare), resample by simple pad/trim to BAR_STEPS
            if bar.shape[0] > BAR_STEPS:
                bar = bar[:BAR_STEPS, :]
            else:
                pad = np.zeros((BAR_STEPS - bar.shape[0], bar.shape[1]), dtype=bar.dtype)
                bar = np.concatenate([bar, pad], axis=0)
        bars.append(bar.T)  # (128, BAR_STEPS)
    return bars


# -------------------------
# Build melodic and drum SSMs for each NPZ
# -------------------------

def prepare_one_song(npz_path: Path):
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
    mt = load_multitrack(npz_path) #lturn NPZ into Multitrack
    resolution = int(mt.resolution)     # steps per quarter (LPD default 24 fits 96 per bar)  :contentReference[oaicite:4]{index=4}

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

    # Downbeat indices (bar edges) at symbolic steps, then seconds
    bar_edges_steps = get_downbeat_indices(mt) #find where each bar starts as list
    if len(bar_edges_steps) < 2:
        return None
    bar_edges_sec = steps_to_seconds(bar_edges_steps, resolution, TEMPO_QPM) #convert downbeat steps -> seconds

    # Write three MIDI variants
    stem = npz_path.stem
    midi_all = OUT_MIDI_ALL / f"{stem}_all_tracks.mid"
    midi_nd  = OUT_MIDI_ND  / f"{stem}_no_drum.mid"
    midi_do  = OUT_MIDI_DO  / f"{stem}_drum_only.mid"
    ensure_dir(midi_all); ensure_dir(midi_nd); ensure_dir(midi_do) 

    # all tracks
    ppr.write(str(midi_all), mt)   # LPD Pypianoroll write()  :contentReference[oaicite:6]{index=6}

    # no-drum (zero out drum track)
    mt_nd = mt.copy()
    mt_nd.tracks[drum_idx].pianoroll[:] = 0 #zeroing out drum tracl
    ppr.write(str(midi_nd), mt_nd)

    # drum-only (zero out other tracks)
    mt_do = mt.copy()
    for i, tr in enumerate(mt_do.tracks):
        if i != drum_idx:
            tr.pianoroll[:] = 0
    ppr.write(str(midi_do), mt_do)

    # Render no-drum to WAV (CQT input)
    wav_nd = OUT_WAV_ND / f"{stem}_no_drum.wav"
    syn_midi_to_wav(midi_nd, wav_nd, sr=SR)

    # ------- Build bar grid with 96 sub-intervals in *seconds* -------
    # Each bar: [t_b, t_{b+1}), subdivided into 96 equal bins (then shifted by half-bin)
    song_bar_grid_range_list = []
    for b in range(len(bar_edges_sec) - 1):
        t0, t1 = bar_edges_sec[b], bar_edges_sec[b+1]
        time_grid = np.linspace(t0, t1, BAR_STEPS + 1)
        bar_grid = np.stack([time_grid[:-1], time_grid[1:]], axis=1)
        # shift by half-bin (centered), clip at 0
        half = (t1 - t0) / BAR_STEPS * 0.5
        bar_grid = bar_grid - half
        bar_grid[bar_grid < 0] = 0.0
        song_bar_grid_range_list.append(bar_grid)

    # ------- Load audio & compute CQT -------
    y, sr = librosa.load(str(wav_nd), sr=SR, mono=True) #load WAV file using librosa
    dur = len(y) / sr 
    cqt = librosa.cqt(y, sr=sr, hop_length=HOP) #compute CQT for bar-to-bar conversion into freq bins
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    fps = cqt_db.shape[1] / dur  # frames per second

    # Mean Pooling CQT per (bar, 96 bins) -> (84 x 96) bar "image"
    # pool: swuashing bunch of CQT frames inside tiny frame window so that every bar ends up the same fixed size
    bars_cqt = []
    for bar_grid in song_bar_grid_range_list: # for each bar in the song,
        note_feats = []
        for n in range(BAR_STEPS): #from 0 till 96,
            t_start, t_end = bar_grid[n, 0], bar_grid[n, 1] 
            f0 = int(np.round(t_start * fps))
            f1 = int(np.round(t_end * fps))
            if f1 <= f0:
                # degenerate slice: reuse previous or zeros
                if len(note_feats) == 0:
                    feat = np.zeros((N_BINS,), dtype=np.float32)
                else:
                    feat = note_feats[-1]
            else:
                slice_ = cqt_db[:, f0:f1] #grab CQT frames that fall inside the bin
                feat = slice_.mean(axis=1).astype(np.float32)  # 84-dim. #average across time to make one 84-dim vector
            note_feats.append(feat)
        bar_img = np.stack(note_feats, axis=1)  # (84, 96) stack to make it a 84x96 bar image
        bars_cqt.append(bar_img)
    bars_cqt = np.array(bars_cqt, dtype=np.float32)  # (B, 84, 96)

    # ------- Melodic SSM from CQT bars (Euclidean) -------
    mel_ssm = pairwise_euclidean_bar_ssm(bars_cqt)   # melodic SSM (B x B) sec 3.1
    mel_ssm = minmax01(pad_to_256(mel_ssm)) # return max number, convert it to 256x256

    # ------- Drum SSM from symbolic drum bars (Euclidean) -------
    # Pull raw drum pianoroll (T, 128)
    drum_roll = mt.tracks[drum_idx].pianoroll  # shape (T, 128)
    # Binarize velocities for SSM (paper uses symbolic drum matrices)
    drum_roll_bin = (drum_roll > 0).astype(np.float32)
    # slice into per-bar matrices (128, 96)
    drum_bars = slice_bars(drum_roll_bin, bar_edges_steps, steps_per_bar=BAR_STEPS)  # list of (128,96)
    drum_bars = np.array(drum_bars, dtype=np.float32)  # (B,128,96)
    drum_ssm = pairwise_euclidean_bar_ssm(drum_bars) # drum SSM (BxB)
    drum_ssm = minmax01(pad_to_256(drum_ssm)) # return max number, convert it to 256x256

    # ------- Save pickles to the same names our trainer expects -------
    base = stem  # we’ll keep the raw stem
    fname_mel = OUT_MEL_SSM / f"song_barlv_ssm_{base}.pkl"
    fname_drm = OUT_DRUM_SSM / f"song_barlv_drum_ssm_{base}.pkl"
    ensure_dir(fname_mel); ensure_dir(fname_drm)
    with open(fname_mel, "wb") as f: pickle.dump(mel_ssm, f)
    with open(fname_drm, "wb") as f: pickle.dump(drum_ssm, f)

    # (Optional) save a lightweight “midi object list” entry (compat with your older pipeline)
    # Only saving what we need now
    return {
        "stem": stem,
        "midi_all": str(midi_all),
        "midi_nd": str(midi_nd),
        "midi_do": str(midi_do),
        "wav_nd": str(wav_nd),
        "bars": len(drum_bars)
    }


def prepare_dataset(limit=None):
    #create output folders
    OUT_MIDI_ALL.mkdir(parents=True, exist_ok=True)
    OUT_MIDI_ND.mkdir(parents=True, exist_ok=True)
    OUT_MIDI_DO.mkdir(parents=True, exist_ok=True)
    OUT_WAV_ND.mkdir(parents=True, exist_ok=True)
    OUT_CQT_POOL.mkdir(parents=True, exist_ok=True)
    OUT_MEL_SSM.mkdir(parents=True, exist_ok=True)
    OUT_DRUM_SSM.mkdir(parents=True, exist_ok=True)

    #finds all npz files
    npz_files = list_npz_files(DATASET_ROOT)
    if limit is not None: #if there is limit on range of npz files, apply it
        npz_files = npz_files[:limit]

    meta = []
    for i, p in enumerate(npz_files, 1): #for each npz file,
        try:
            info = prepare_one_song(p) #process the data inside, collect per-song metadata (paths, #bars)
            if info is not None:
                meta.append(info)
                print(f"[prep] {i}/{len(npz_files)} {p.name} -> bars: {info['bars']}")
            else:
                print(f"[prep] {i}/{len(npz_files)} {p.name} -> skipped (no drums or no bars)")
        except Exception as e:
            print(f"[prep] {i}/{len(npz_files)} {p.name} -> ERROR {e}")

    ensure_dir(OUT_OBJ_PKL)
    with open(OUT_OBJ_PKL, "wb") as f:
        pickle.dump(meta, f)
    print(f"[prep] done. saved meta with {len(meta)} items.")


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
        return mel, drm


def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

bce = nn.BCELoss()

def train_ssm():
    # If no SSMs yet, run preparation first
    #check if preprocessed melodic/drum SSM pickles exist. If not, call prepare_dataset()
    if len(list(OUT_MEL_SSM.glob("*.pkl"))) == 0 or len(list(OUT_DRUM_SSM.glob("*.pkl"))) == 0:
        print("[info] No SSMs detected — preparing dataset from LPD via Pypianoroll...")
        prepare_dataset(limit=None)

    dataset = SSMTrainDataset(OUT_MEL_SSM, OUT_DRUM_SSM) #instantiate SSM Train Datatset
    assert len(dataset) > 0, "No paired SSM samples found."

    #train test split
    n_total = len(dataset)
    n_train = int(0.9 * n_total)
    n_val   = n_total - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=False)

    # Models (as in §4.4: 8 conv + 3 FC (+skip), 32-d latent)
    enc = SSMEncoder(in_channels=1, base_channels=64, latent_dim=32)
    dec = SSMDecoder(out_channels=1, base_channels=64, latent_dim=32)
    vae = SSMVAE(enc, dec).to(DEVICE)
    dis = SSMDiscriminator(in_channels=1, base_channels=64).to(DEVICE)

    #create adam optimizers for VAE and 
    optG = torch.optim.Adam(vae.parameters(), lr=LR_GEN, betas=(BETA1, BETA2))
    optD = torch.optim.Adam(dis.parameters(), lr=LR_DIS, betas=(BETA1, BETA2))

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, NUM_EPOCHS+1): #repeat NUM_EPOCHS times
        vae.train(); dis.train() #train vae model and discriminator
        total_G, total_D = 0.0, 0.0 

        for mel, drm in train_loader: #for every (mel ssm, drum ssm) pair,
            mel, drm = mel.to(DEVICE), drm.to(DEVICE) #move to GPU/CPU as needed

            # --- Train D (Training discriminator) --- Teaching judge to tell real vs fake
            optD.zero_grad(set_to_none=True) #zero the gradient buffers
            with torch.no_grad(): #don't want to update generator while making fakes - no tracking for backprop needed
                recon, mu_g, logvar_g = vae(mel) #fake drum SSM made by us - reconstruction loss
            pred_real = dis(drm) #get D's score for the actual drum (should be 1)
            pred_fake = dis(recon.detach()) #get D's score for the fake drum made by vae generator(should be 0)
            d_loss = 0.5 * (bce(pred_real, torch.ones_like(pred_real)) +
                            bce(pred_fake, torch.zeros_like(pred_fake)))    #BCE Loss on real & fake -> ipldd
            d_loss.backward() #backprop according to this loss^
            optD.step() #next step?

            # --- Train G (Training VAE Generator) ---
            optG.zero_grad(set_to_none=True)        #zero the gradient buffers
            recon, mu, logvar = vae(mel)            #get vae's output for melody input - make fake drum ssm, this time with grads!
            loss_rec = F.mse_loss(recon, drm)       #recon loss (mse): helps predicted drum SSM be close to real drum ssm,
            loss_kl  = kl_divergence(mu, logvar)    # VAE regularizer - keeps latent space nice & gaussian
            pred_fake_for_G = dis(recon)            # discriminator's prediction for the fake output
            loss_gan = bce(pred_fake_for_G, torch.ones_like(pred_fake_for_G))
            g_loss = LAMBDA_REC*loss_rec + LAMBDA_KL*loss_kl + LAMBDA_GAN*loss_gan  #combine losses to get total loss for fooling discriminator
            g_loss.backward()                       # run backprop
            optG.step()

            total_G += g_loss.item()    #keep running totals so u can print avg generator/discriminator losses later
            total_D += d_loss.item()

        # validation
        vae.eval(); dis.eval()
        with torch.no_grad(): #don't track gradients
            val_G = 0.0
            for mel, drm in val_loader: #for each validation batch
                mel, drm = mel.to(DEVICE), drm.to(DEVICE) 
                recon, mu, logvar = vae(mel)
                loss_rec = F.mse_loss(recon, drm)
                loss_kl  = kl_divergence(mu, logvar)
                pred_fake = dis(recon)
                loss_gan = bce(pred_fake, torch.ones_like(pred_fake))
                val_G += (LAMBDA_REC*loss_rec + LAMBDA_KL*loss_kl + LAMBDA_GAN*loss_gan).item()
            val_G /= max(1, len(val_loader))

        avgG = total_G / max(1, len(train_loader)) #printing avg G and D loss
        avgD = total_D / max(1, len(train_loader))
        print(f"[Epoch {epoch:03d}]  G:{avgG:.4f}  D:{avgD:.4f}  ValG:{val_G:.4f}")

        #if validation generator loss improved, save checkpoint
        if val_G < best_val:
            best_val = val_G
            torch.save({
                "epoch": epoch,
                "vae": vae.state_dict(),
                "dis": dis.state_dict(),
                "optG": optG.state_dict(),
                "optD": optD.state_dict(),
                "val": val_G
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


if __name__ == "__main__":
    train_ssm()

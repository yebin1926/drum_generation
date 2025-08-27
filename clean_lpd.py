#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_lpd_wei.py

Cleans LPD songs following Wei et al. (2019):

  Step 1) Remove songs with inconsistent duration after synthesis.
          (Default uses a fast symbolic check; 'synth' is slower but exact.)
  Step 2) Remove songs with empty/noisy drum tracks:
          compute drum-note counts over *survivors of Step 1*,
          then drop songs outside mean ± 2 * std.
  Step 3) Apply 16th-beat quantization on drum tracks:
          snap notes to the nearest 16th and remove only those that would
          be shifted too far (outside a small tolerance window).

Implementation notes
- Robust loader handles multiple .npz layouts and returns a pypianoroll.Multitrack.
- Pass 1 runs in parallel: we check duration, count drums, bars, and off-grid%.
- Pass 2 computes mean±2σ on the kept candidates from Step 1 (paper order).
- Quantization *modifies notes*, not songs. (Optionally drop songs with high off-grid% if you want stricter filtering.)
- Resume-friendly manifests avoid reprocessing: lpd_manifest_kept.txt / lpd_manifest_trash.txt
- Optional writing of "<stem>.clean.npz" for quantized copies (off by default).

Typical usage (fast, recommended):
  python3 clean_lpd_wei.py --data_root /data --duration_check symbolic --workers 16

Slower, exact audio duration check (requires fluidsynth + SoundFont):
  python3 clean_lpd_wei.py --data_root /data --duration_check synth --workers 4 --write_clean 1
"""

import os
import sys
import math
import time
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List, Dict

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

# External deps
import pypianoroll as ppr   # LPD NPZ <-> Multitrack
import librosa              # audio duration fallback
import soundfile as sf      # audio duration (fast)


# -------------------------
# Config (paths & constants)
# -------------------------

DEFAULT_DATA_ROOT = "/data"
LPD_SUBDIR   = "lpd"
TRASH_SUBDIR = "data_trash"

DEFAULT_SF2 = "/workspace/sound_front_lib.sf2"
SOUND_FONT  = Path(os.environ.get("SOUNDFONT", DEFAULT_SF2))

SR = 44100
ALLOW_REL_ERR = 0.10  # ±10%

# Drum-track + structure filters
MIN_NOTES_HARD = 1        # drop if literally no drum notes
MIN_BARS = 8              # drop very short songs (paper mentions empty/short/noisy)
QUANT_TOL_STEPS = 1       # keep notes within ±1 step of a 16th center

# Optional: drop if too many off-grid notes BEFORE quantization.
# Paper does not hard-drop here; leave None to disable. Set e.g. 0.05 to be stricter.
DROP_IF_OFFGRID_PCT_GT_DEFAULT = None  # or 0.05 to get closer to ~9.9k

# Resume manifests
MANIFEST_KEPT  = "lpd_manifest_kept.txt"
MANIFEST_TRASH = "lpd_manifest_trash.txt"


# -------------------------
# Logging helpers
# -------------------------

def now_ts() -> str:
    return time.strftime("%H:%M:%S")

def log(level: str, msg: str, debug: bool = False):
    if level in ("INFO", "WARN", "ERROR"):
        print(f"{now_ts()} | {level:<5} | {msg}")
    elif debug:
        print(f"{now_ts()} | {level:<5} | {msg}")


# -------------------------
# I/O helpers
# -------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_npz_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.npz"))

def append_line(path: Path, line: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")

def move_to_trash(src: Path, trash_root: Path) -> Path:
    ensure_dir(trash_root)
    dest = trash_root / src.name
    if dest.exists():
        i = 1
        while True:
            cand = trash_root / f"{src.stem}.dup{i}{src.suffix}"
            if not cand.exists():
                dest = cand
                break
            i += 1
    shutil.move(str(src), str(dest))
    return dest


# -------------------------
# NPZ <-> Multitrack utils
# -------------------------

def load_multitrack(npz_path: Path, debug: bool = False) -> ppr.Multitrack:
    """Robust loader: try ppr.load, otherwise reconstruct from arrays."""
    try:
        mt = ppr.load(str(npz_path))
        if isinstance(mt, ppr.Multitrack):
            log("DEBUG", f"[load] {npz_path.name} via ppr.load()", debug)
            return mt
    except Exception as e:
        log("DEBUG", f"[load] ppr.load failed for {npz_path.name}: {e}", debug)

    try:
        data = np.load(str(npz_path), allow_pickle=True)
        keys = set(data.keys())

        if "tracks" in keys:
            tracks = []
            for t in data["tracks"]:
                pr = np.array(t["pianoroll"])
                prog = int(t.get("program", 0))
                is_d = bool(t.get("is_drum", False))
                name = str(t.get("name", ""))
                tracks.append(ppr.Track(pianoroll=pr, program=prog, is_drum=is_d, name=name))
            beat_res = int(data.get("beat_resolution", 24))
            tempo    = np.array(data.get("tempo")) if "tempo" in keys else None
            downbeat = np.array(data.get("downbeat")) if "downbeat" in keys else None
            return ppr.Multitrack(tracks=tracks, tempo=tempo, downbeat=downbeat, beat_resolution=beat_res)

        if {"pianoroll", "programs", "is_drum"} <= keys:
            pr = np.array(data["pianoroll"])
            if pr.ndim == 2:
                pr = pr[None, ...]
            programs = np.array(data["programs"]).tolist()
            is_drums = np.array(data["is_drum"]).tolist()
            names    = data["names"].tolist() if "names" in keys else [""] * len(programs)
            tracks = []
            for i in range(len(programs)):
                tracks.append(ppr.Track(pianoroll=pr[i], program=int(programs[i]),
                                        is_drum=bool(is_drums[i]), name=str(names[i])))
            beat_res = int(data.get("beat_resolution", 24))
            tempo    = np.array(data.get("tempo")) if "tempo" in keys else None
            downbeat = np.array(data.get("downbeat")) if "downbeat" in keys else None
            return ppr.Multitrack(tracks=tracks, tempo=tempo, downbeat=downbeat, beat_resolution=beat_res)

        if "multitrack" in keys:
            obj = data["multitrack"].item() if np.ndim(data["multitrack"]) else data["multitrack"]
            if isinstance(obj, ppr.Multitrack):
                return obj

    except Exception as e:
        log("DEBUG", f"[load] np.load fallback failed for {npz_path.name}: {e}", debug)

    raise TypeError("Unsupported NPZ layout for pypianoroll.Multitrack")


def multitrack_max_length_steps(mt: ppr.Multitrack) -> int:
    L = 0
    for tr in mt.tracks:
        pr = getattr(tr, "pianoroll", None)
        if pr is not None:
            L = max(L, pr.shape[0])
    return L

def steps_per_16th(beat_res: int) -> int:
    # beat_resolution is steps per quarter note. 16th = quarter/4.
    return max(1, int(round(beat_res / 4.0)))

def find_drum_track_idx(mt: ppr.Multitrack) -> Optional[int]:
    for i, tr in enumerate(mt.tracks):
        if getattr(tr, "is_drum", False):
            return i
    return None

def drum_note_count(mt: ppr.Multitrack, drum_idx: int) -> int:
    roll = mt.tracks[drum_idx].pianoroll
    if roll is None or roll.size == 0:
        return 0
    return int((roll > 0).sum())


# -------------------------
# Duration & quantization
# -------------------------

def write_temp_midi(mt: ppr.Multitrack, tmpdir: Path) -> Path:
    midi_path = tmpdir / "temp.mid"
    ppr.write(str(midi_path), mt)  # requires Multitrack object
    return midi_path

def synthesize_midi_to_wav(midi_path: Path, wav_path: Path, sr: int = SR):
    if not SOUND_FONT.exists():
        raise FileNotFoundError(f"SoundFont not found: {SOUND_FONT}")
    cmd = ["fluidsynth", "-ni", str(SOUND_FONT), str(midi_path), "-F", str(wav_path), "-r", str(sr)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def audio_duration_seconds(path: Path) -> float:
    try:
        f = sf.SoundFile(str(path))
        dur = len(f) / float(f.samplerate)
        f.close()
        return float(dur)
    except Exception:
        y, sr = librosa.load(str(path), sr=None, mono=True)
        return float(len(y) / sr)

def symbolic_duration_seconds(mt: ppr.Multitrack) -> float:
    T = multitrack_max_length_steps(mt)
    res = int(getattr(mt, "beat_resolution", 24))
    tempo = getattr(mt, "tempo", None)

    if tempo is None:
        sec_per_step = 60.0 / (120.0 * res)  # assume 120 QPM
        return T * sec_per_step

    tempo = np.asarray(tempo).squeeze()
    if tempo.ndim != 1:
        tempo = tempo.reshape(-1)
    if tempo.shape[0] < T:
        pad = np.full(T - tempo.shape[0], float(np.median(tempo)) if tempo.size else 120.0, dtype=float)
        tempo = np.concatenate([tempo, pad], axis=0)
    elif tempo.shape[0] > T:
        tempo = tempo[:T]

    sec_per_step = 60.0 / (tempo * float(res))
    sec_per_step = np.clip(sec_per_step, 1e-6, 10.0)
    return float(sec_per_step.sum())

def duration_consistency_ok(mt: ppr.Multitrack, mode: str = "symbolic", debug: bool = False) -> bool:
    if mode == "none":
        return True

    if mode == "symbolic":
        T = multitrack_max_length_steps(mt)
        tempo = getattr(mt, "tempo", None)
        if tempo is None:
            return True
        tempo = np.asarray(tempo).squeeze()
        if tempo.ndim != 1:
            tempo = tempo.reshape(-1)
        tol = int(max(1, ALLOW_REL_ERR * max(T, len(tempo))))
        ok = abs(len(tempo) - T) <= tol
        if debug and not ok:
            log("DEBUG", f"[dur/sym] T={T} tempo_len={len(tempo)} tol={tol}", True)
        return ok

    # 'synth' mode: render to WAV and compare seconds (slow)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        midi = write_temp_midi(mt, td)
        wav  = td / "temp.wav"
        try:
            synthesize_midi_to_wav(midi, wav, sr=SR)
            audio_sec = audio_duration_seconds(wav)
        except Exception as e:
            if debug:
                log("DEBUG", f"[synth] render error: {e}", True)
            return False

    sym_sec = symbolic_duration_seconds(mt)
    if sym_sec <= 0.1:
        return False
    ratio = audio_sec / sym_sec
    ok = (1.0 - ALLOW_REL_ERR) <= ratio <= (1.0 + ALLOW_REL_ERR)
    if debug and not ok:
        log("DEBUG", f"[dur] mismatch: audio={audio_sec:.2f}s symbolic={sym_sec:.2f}s ratio={ratio:.3f}", True)
    return ok

def offgrid_percentage(roll: np.ndarray, res: int, tol_steps: int) -> float:
    """Compute percentage of active frames that are not within ±tol of a 16th center."""
    T = roll.shape[0]
    if T == 0:
        return 0.0
    s16 = steps_per_16th(res)
    on = np.zeros(T, dtype=bool)
    # mark centers ± tol
    for c in range(0, T, s16):
        lo = max(0, c - tol_steps)
        hi = min(T, c + tol_steps + 1)
        on[lo:hi] = True
    active = (roll > 0)
    total = int(active.sum())
    if total == 0:
        return 0.0
    off = int(active[~on].sum())
    return off / total

def quantize_drum_track_inplace(mt: ppr.Multitrack, drum_idx: int, tol_steps: int = QUANT_TOL_STEPS) -> Tuple[int, int, float]:
    """
    Snap notes to nearest 16th center within ±tol; remove notes outside that window.
    Implementation: for each 16th center, take max over the local window, write it at the center,
    zero the window elsewhere. Returns (kept_notes, removed_notes, removed_pct).
    """
    res = int(getattr(mt, "beat_resolution", 24))
    s16 = steps_per_16th(res)
    roll = mt.tracks[drum_idx].pianoroll
    T = roll.shape[0]

    cur = roll > 0
    total_notes = int(cur.sum())
    if total_notes == 0:
        return 0, 0, 0.0

    removed_notes = 0
    # Work on a copy to avoid interfering while sweeping windows
    out = np.zeros_like(roll, dtype=roll.dtype)

    for c in range(0, T, s16):
        lo = max(0, c - tol_steps)
        hi = min(T, c + tol_steps + 1)
        window = roll[lo:hi]  # (win, 128)
        if window.size == 0:
            continue
        grid_val = (window > 0).max(axis=0).astype(roll.dtype)
        out[c, :] = np.maximum(out[c, :], grid_val)

    kept_notes = int((out > 0).sum())
    removed_notes = total_notes - kept_notes
    mt.tracks[drum_idx].pianoroll = out
    removed_pct = 0.0 if total_notes == 0 else removed_notes / total_notes
    return kept_notes, removed_notes, removed_pct


# -------------------------
# PASS 1 (parallel): duration, bars, drum presence, counts
# -------------------------

def pass1_one(args) -> Dict:
    p, duration_check, min_bars, tol_steps, debug = args
    try:
        mt = load_multitrack(p, debug=False)
        res = int(getattr(mt, "beat_resolution", 24))
        T = multitrack_max_length_steps(mt)
        bars = int(round(T / float(res * 4)))

        di = find_drum_track_idx(mt)
        if di is None:
            return {"path": p, "status": "no_drum"}

        roll = mt.tracks[di].pianoroll
        if roll is None or roll.size == 0 or (roll > 0).sum() < MIN_NOTES_HARD:
            return {"path": p, "status": "empty_drum"}

        if bars < min_bars:
            return {"path": p, "status": "too_short", "bars": bars}

        if not duration_consistency_ok(mt, mode=duration_check, debug=debug):
            return {"path": p, "status": "bad_duration"}

        # compute drum counts + off-grid%
        drum_cnt = drum_note_count(mt, di)
        og_pct = offgrid_percentage(roll, res, tol_steps)
        return {"path": p, "status": "ok", "drum_notes": drum_cnt, "offgrid_pct": og_pct, "bars": bars}

    except Exception as e:
        return {"path": p, "status": "error", "error": str(e)}

def pass1_collect(npz_files: List[Path], duration_check: str, min_bars: int,
                  tol_steps: int, workers: int, debug: bool) -> List[Dict]:
    args = [(p, duration_check, min_bars, tol_steps, debug) for p in npz_files]
    if workers <= 1:
        out = []
        for a in tqdm(args, desc="[pass1] scan", unit="file"):
            out.append(pass1_one(a))
        return out
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(tqdm(ex.map(pass1_one, args),
                         total=len(args), desc="[pass1] scan", unit="file"))


# -------------------------
# Stats & pipeline
# -------------------------

def compute_outlier_thresholds(counts: List[int]) -> Tuple[float, float, float, float]:
    if len(counts) == 0:
        return 0.0, 0.0, -math.inf, math.inf
    arr = np.asarray(counts, dtype=float)
    mu = float(arr.mean())
    sd = float(arr.std(ddof=0))
    lo = mu - 2.0 * sd
    hi = mu + 2.0 * sd
    return mu, sd, lo, hi


def clean_dataset(
    data_root: Path,
    lpd_dir: Path,
    trash_dir: Path,
    dry_run: bool = False,
    duration_check: str = "symbolic",
    workers: int = 8,
    write_clean: bool = False,
    resume: bool = True,
    debug: bool = False,
    min_bars: int = MIN_BARS,
    drop_if_offgrid_pct_gt: Optional[float] = DROP_IF_OFFGRID_PCT_GT_DEFAULT,
    quant_tol_steps: int = QUANT_TOL_STEPS,
):
    log("INFO",  f"[paths] DATA_ROOT={data_root}")
    log("INFO",  f"[paths] LPD_DIR={lpd_dir}")
    log("INFO",  f"[paths] TRASH_DIR={trash_dir}")
    log("INFO",  f"[conf ] duration_check={duration_check}  workers={workers}  write_clean={int(write_clean)}  resume={int(resume)}")
    log("INFO",  f"[conf ] min_bars={min_bars}  drop_if_offgrid_pct_gt={drop_if_offgrid_pct_gt}  quant_tol_steps={quant_tol_steps}")
    if duration_check == "synth" and not SOUND_FONT.exists():
        log("WARN", f"SoundFont not found at {SOUND_FONT}. 'synth' check will fail.")

    ensure_dir(trash_dir)

    kept_file  = data_root / MANIFEST_KEPT
    trash_file = data_root / MANIFEST_TRASH
    done_kept  = set(kept_file.read_text().splitlines())  if (resume and kept_file.exists())  else set()
    done_trash = set(trash_file.read_text().splitlines()) if (resume and trash_file.exists()) else set()
    done = done_kept | done_trash

    all_npz_all = list_npz_files(lpd_dir)
    all_npz = [p for p in all_npz_all if p.stem not in done]

    log("INFO", f"[info] discovered {len(all_npz_all)} .npz files under {lpd_dir}")
    if resume:
        log("INFO", f"[info] skipping {len(done)} already decided files (kept={len(done_kept)}, trash={len(done_trash)})")
        log("INFO", f"[info] scanning remaining {len(all_npz)} files")

    # ---- PASS 1: check duration/bars/drums; collect counts & off-grid ----
    p1 = pass1_collect(all_npz, duration_check, min_bars, quant_tol_steps, workers, debug)

    # Candidates = survivors of Step 1 (paper order)
    candidates = [m for m in p1 if m.get("status") == "ok"]
    # Compute stats for Step 2 only on candidates
    counts = [m["drum_notes"] for m in candidates if m["drum_notes"] >= MIN_NOTES_HARD]
    mu, sd, lo, hi = compute_outlier_thresholds(counts)
    log("INFO", f"[stats] drum-note (post-step1): mean={mu:.1f} std={sd:.1f} keep=[{max(lo,0):.1f},{hi:.1f}]")
    log("INFO", f"[filter] step1 candidates: {len(candidates)} / scanned {len(p1)}")

    kept_stems = 0
    trashed_stems = 0

    # Apply Step 2 (mean±2σ) and optional off-grid% drop
    survivors = []
    for m in candidates:
        p = m["path"]; stem = p.stem
        dn = m["drum_notes"]; og = m["offgrid_pct"]

        if not (lo <= dn <= hi):
            log("INFO", f"[drop] {p.name} (drum notes={dn} out of [{lo:.1f},{hi:.1f}])")
            if not dry_run:
                move_to_trash(p, trash_dir); append_line(trash_file, stem)
            trashed_stems += 1
            continue

        if drop_if_offgrid_pct_gt is not None and og > float(drop_if_offgrid_pct_gt):
            log("INFO", f"[drop] {p.name} (off-grid {og*100:.2f}% > {float(drop_if_offgrid_pct_gt)*100:.2f}%)")
            if not dry_run:
                move_to_trash(p, trash_dir); append_line(trash_file, stem)
            trashed_stems += 1
            continue

        survivors.append(m)

    log("INFO", f"[filter] step2 survivors: {len(survivors)}")

    # Step 3: quantize (modify notes), and optionally write clean copies
    for m in tqdm(survivors, desc="[quantize] drum tracks", unit="file"):
        p = m["path"]; stem = p.stem
        try:
            mt = load_multitrack(p, debug=debug)
            di = find_drum_track_idx(mt)
            if di is None:
                # should not happen (already filtered), but be safe
                log("INFO", f"[drop] {p.name} (lost drum track?)")
                if not dry_run:
                    move_to_trash(p, trash_dir); append_line(trash_file, stem)
                trashed_stems += 1
                continue

            kept, removed, removed_pct = quantize_drum_track_inplace(mt, di, tol_steps=quant_tol_steps)
            log("DEBUG", f"[quant] {p.name}: kept={kept} removed={removed} ({removed_pct*100:.2f}%)", debug)

            if not dry_run:
                # Mark kept
                append_line(kept_file, stem)
                # Optionally write a quantized copy
                if write_clean:
                    out_clean = p.with_suffix("").with_name(p.stem + ".clean.npz")
                    try:
                        ppr.save(str(out_clean), mt)
                    except Exception:
                        np.savez_compressed(
                            str(out_clean),
                            tracks=[tr.pianoroll for tr in mt.tracks],
                            tempo=mt.tempo,
                            beat_resolution=getattr(mt, "beat_resolution", 24),
                            downbeat=getattr(mt, "downbeat", None),
                            programs=[tr.program for tr in mt.tracks],
                            is_drum=[tr.is_drum for tr in mt.tracks],
                            names=[getattr(tr, "name", "") for tr in mt.tracks],
                        )
            kept_stems += 1

        except Exception as e:
            log("ERROR", f"[quant] {p.name}: {e}")
            if not dry_run:
                try:
                    move_to_trash(p, trash_dir); append_line(trash_file, stem)
                except Exception:
                    pass
            trashed_stems += 1

    # Totals incl. already-done stems
    total_all = len(list_npz_files(lpd_dir))
    kept_total  = kept_stems + len(done_kept)
    trash_total = trashed_stems + len(done_trash)

    log("INFO", "")
    log("INFO", f"[summary] kept (this run / total): {kept_stems} / {kept_total}")
    log("INFO", f"[summary] trashed (this run / total): {trashed_stems} / {trash_total}")
    log("INFO", f"[summary] total npz observed: {total_all}")


# -------------------------
# CLI
# -------------------------

def parse_bool(x: str) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Clean LPD dataset (Wei et al., 2019).")
    ap.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT,
                    help="Root that contains 'lpd/' and where manifests will be written.")
    ap.add_argument("--lpd_subdir", type=str, default=LPD_SUBDIR,
                    help="Relative subfolder under data_root containing .npz files.")
    ap.add_argument("--trash_subdir", type=str, default=TRASH_SUBDIR,
                    help="Relative subfolder under data_root for trashed files.")
    ap.add_argument("--dry_run", type=parse_bool, default=False,
                    help="If True, log decisions but do not move/write files.")
    ap.add_argument("--duration_check", choices=["synth", "symbolic", "none"], default="symbolic",
                    help="Duration consistency mode (synth is slow; symbolic is fast).")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel workers for Pass 1.")
    ap.add_argument("--write_clean", type=parse_bool, default=False,
                    help="If True, write '<stem>.clean.npz' quantized copies next to originals.")
    ap.add_argument("--resume", type=parse_bool, default=True,
                    help="If True, skip stems already listed in kept/trash manifests.")
    ap.add_argument("--debug", type=parse_bool, default=False,
                    help="Verbose debug logs.")
    ap.add_argument("--min_bars", type=int, default=MIN_BARS,
                    help="Minimum bars required to keep a song (before outlier step).")
    ap.add_argument("--drop_if_offgrid_pct_gt", type=float, default=DROP_IF_OFFGRID_PCT_GT_DEFAULT if DROP_IF_OFFGRID_PCT_GT_DEFAULT is not None else -1.0,
                    help="Optional: drop if off-grid%% > threshold (e.g., 0.05). Use negative to disable.")
    ap.add_argument("--quant_tol_steps", type=int, default=QUANT_TOL_STEPS,
                    help="±steps around 16th center kept during quantization.")

    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    lpd_dir   = data_root / args.lpd_subdir
    trash_dir = data_root / args.trash_subdir

    if not lpd_dir.exists():
        log("ERROR", f"LPD directory not found: {lpd_dir}")
        sys.exit(1)

    drop_thresh = None if args.drop_if_offgrid_pct_gt is None or args.drop_if_offgrid_pct_gt < 0 else float(args.drop_if_offgrid_pct_gt)

    clean_dataset(
        data_root=data_root,
        lpd_dir=lpd_dir,
        trash_dir=trash_dir,
        dry_run=bool(args.dry_run),
        duration_check=args.duration_check,
        workers=int(args.workers),
        write_clean=bool(args.write_clean),
        resume=bool(args.resume),
        debug=bool(args.debug),
        min_bars=int(args.min_bars),
        drop_if_offgrid_pct_gt=drop_thresh,
        quant_tol_steps=int(args.quant_tol_steps),
    )

if __name__ == "__main__":
    main()
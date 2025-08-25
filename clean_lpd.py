#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_lpd.py

Clean the LPD dataset following Wei et al. (2019):

 1) Remove songs with inconsistent duration after synthesis.
    (Fast defaults avoid real audio rendering; you can enable true synthesis.)
 2) Remove songs with empty/noisy drum tracks (drum-note counts outside mean ± 2*std).
 3) Apply 16th-beat quantization on the drum track (keep notes on or near 16th grid).

Implementation notes:
- Robust NPZ loader handles multiple on-disk layouts; falls back to reconstructing a
  pypianoroll.Multitrack from arrays when ppr.load() fails.
- Pass 0 (count drum notes) is parallelized with ProcessPoolExecutor.
- Resume-friendly: writes manifests (kept/trash) so you can re-run without redoing work.
- Optional writing of ".clean.npz" per kept song (off by default to reduce I/O).

Default paths assume:
  --data_root /data
  NPZ files under /data/lpd
  SoundFont at /workspace/sound_front_lib.sf2 (can override via $SOUNDFONT).

Example (fast, recommended):
  python3 clean_lpd.py --data_root /data --duration_check symbolic --workers 16 --write_clean 0

Exact Wei-style check (slow: does fluidsynth render):
  python3 clean_lpd.py --data_root /data --duration_check synth --workers 4 --write_clean 1
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

# DATA ROOT (contains lpd/ and data_trash/)
DEFAULT_DATA_ROOT = "/data"

# Where LPD npz live (under data_root)
LPD_SUBDIR   = "lpd"
TRASH_SUBDIR = "data_trash"

# SoundFont configuration
DEFAULT_SF2 = "/workspace/sound_front_lib.sf2"  # your fixed path
SOUND_FONT  = Path(os.environ.get("SOUNDFONT", DEFAULT_SF2))

# Synthesis & timing
SR = 44100
ALLOW_REL_ERR = 0.10  # ±10% tolerance for duration/length checks

# Drum-track outlier filter
MIN_NOTES_HARD = 1    # drop songs with literally zero drum notes

# Quantization: 16th-beat grid at LPD's 96 steps/bar -> every 6 steps
SIXTEENTH_STEPS = 6
TOL_STEPS = 1  # keep notes within ±1 step of 16th grid

# Resume manifest filenames (written under data_root)
MANIFEST_KEPT  = "lpd_manifest_kept.txt"
MANIFEST_TRASH = "lpd_manifest_trash.txt"

# -------------------------
# Small helpers & logging
# -------------------------

def now_ts() -> str:
    return time.strftime("%H:%M:%S")

def log(level: str, msg: str, debug: bool = False):
    if level in ("INFO", "WARN", "ERROR"):
        print(f"{now_ts()} | {level:<5} | {msg}")
    elif debug:
        print(f"{now_ts()} | {level:<5} | {msg}")

def list_npz_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.npz"))

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def append_line(path: Path, line: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")

def move_to_trash(src: Path, trash_root: Path) -> Path:
    ensure_dir(trash_root)
    dest = trash_root / src.name
    if dest.exists():
        stem, suf = src.stem, src.suffix
        i = 1
        while True:
            cand = trash_root / f"{stem}.dup{i}{suf}"
            if not cand.exists():
                dest = cand
                break
            i += 1
    shutil.move(str(src), str(dest))
    return dest

# -------------------------
# NPZ <-> Multitrack utils
# -------------------------

def multitrack_max_length_steps(mt: ppr.Multitrack) -> int:
    L = 0
    for tr in mt.tracks:
        pr = getattr(tr, "pianoroll", None)
        if pr is not None:
            L = max(L, pr.shape[0])
    return L

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

def load_multitrack(npz_path: Path, debug: bool = False) -> ppr.Multitrack:
    """
    Robust loader:
      1) try ppr.load(...)
      2) fallback: reconstruct from arrays in npz (common LPD variants)
    """
    # 1) canonical path
    try:
        mt = ppr.load(str(npz_path))
        if isinstance(mt, ppr.Multitrack):
            log("DEBUG", f"[load] {npz_path.name} via ppr.load()", debug)
            return mt
    except Exception as e:
        log("DEBUG", f"[load] ppr.load failed for {npz_path.name}: {e}", debug)

    # 2) fallback reconstructions
    try:
        data = np.load(str(npz_path), allow_pickle=True)
        keys = set(data.keys())

        # Case A: "tracks" object array of dict-like entries
        if "tracks" in keys:
            tracks = []
            for t in data["tracks"]:
                # handle np.void, dict, etc.
                pianoroll = np.array(t["pianoroll"])
                program   = int(t.get("program", 0))
                is_drum   = bool(t.get("is_drum", False))
                name      = str(t.get("name", ""))
                tracks.append(ppr.Track(pianoroll=pianoroll,
                                        program=program,
                                        is_drum=is_drum,
                                        name=name))
            beat_resolution = int(data.get("beat_resolution", 24))
            tempo    = np.array(data.get("tempo")) if "tempo" in keys else None
            downbeat = np.array(data.get("downbeat")) if "downbeat" in keys else None
            mt = ppr.Multitrack(tracks=tracks,
                                tempo=tempo,
                                downbeat=downbeat,
                                beat_resolution=beat_resolution)
            log("DEBUG", f"[load] {npz_path.name} via fallback 'tracks'", debug)
            return mt

        # Case B: stacked arrays (programs, is_drum, pianoroll)
        if {"pianoroll", "programs", "is_drum"} <= keys:
            pr = np.array(data["pianoroll"])  # (T,128) or (n_tracks,T,128)
            if pr.ndim == 2:
                pr = pr[None, ...]
            programs = np.array(data["programs"]).tolist()
            is_drums = np.array(data["is_drum"]).tolist()
            names    = data["names"].tolist() if "names" in keys else [""] * len(programs)
            tracks = []
            for i in range(len(programs)):
                tracks.append(ppr.Track(
                    pianoroll=pr[i],
                    program=int(programs[i]),
                    is_drum=bool(is_drums[i]),
                    name=str(names[i])
                ))
            beat_resolution = int(data.get("beat_resolution", 24))
            tempo    = np.array(data.get("tempo")) if "tempo" in keys else None
            downbeat = np.array(data.get("downbeat")) if "downbeat" in keys else None
            mt = ppr.Multitrack(tracks=tracks,
                                tempo=tempo,
                                downbeat=downbeat,
                                beat_resolution=beat_resolution)
            log("DEBUG", f"[load] {npz_path.name} via fallback 'stacked arrays'", debug)
            return mt

        # Case C: single pickled multitrack object
        if "multitrack" in keys:
            obj = data["multitrack"].item() if np.ndim(data["multitrack"]) else data["multitrack"]
            if isinstance(obj, ppr.Multitrack):
                log("DEBUG", f"[load] {npz_path.name} via embedded multitrack", debug)
                return obj

    except Exception as e:
        log("DEBUG", f"[load] np.load fallback failed for {npz_path.name}: {e}", debug)

    raise TypeError("Unsupported NPZ layout for pypianoroll.Multitrack")

# -------------------------
# Duration & quantization
# -------------------------

def write_temp_midi(mt: ppr.Multitrack, tmpdir: Path) -> Path:
    midi_path = tmpdir / "temp.mid"
    ppr.write(str(midi_path), mt)  # requires Multitrack instance
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
    """Compute symbolic duration (seconds) from tempo array and beat_resolution."""
    T = multitrack_max_length_steps(mt)
    res = int(getattr(mt, "beat_resolution", 24))
    tempo = getattr(mt, "tempo", None)

    if tempo is None:
        # assume constant 120 QPM
        sec_per_step = 60.0 / (120.0 * res)
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
    """
    mode:
      - 'none'     : always True (fastest).
      - 'symbolic' : cheap structural check (no audio render).
      - 'synth'    : render WAV with fluidsynth and compare seconds (slow).
    """
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

    # mode == 'synth' (slow, exact seconds)
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

def sixteen_grid_mask(T: int) -> np.ndarray:
    """Boolean mask: True on/near 16th grid (±TOL_STEPS), assuming 96 steps per bar."""
    g = np.zeros(T, dtype=bool)
    for t in range(T):
        r = t % SIXTEENTH_STEPS
        if r <= TOL_STEPS or r >= SIXTEENTH_STEPS - TOL_STEPS:
            g[t] = True
    return g

def quantize_drum_track_inplace(mt: ppr.Multitrack, drum_idx: int) -> Tuple[int, int]:
    """
    Keep only notes at time-steps near the 16th grid; zero the rest.
    Returns (kept_notes, removed_notes).
    """
    roll = mt.tracks[drum_idx].pianoroll
    T = roll.shape[0]
    on_near_grid = sixteen_grid_mask(T)

    cur = roll > 0
    total_notes = int(cur.sum())

    off_idx = ~on_near_grid
    removed_notes = int(cur[off_idx].sum())
    roll[off_idx, :] = 0

    kept_notes = total_notes - removed_notes
    return kept_notes, removed_notes

# -------------------------
# Pass 0: parallel drum counts
# -------------------------

def pass0_one(p: Path) -> Dict:
    try:
        mt = load_multitrack(p, debug=False)
        di = find_drum_track_idx(mt)
        if di is None:
            return {"path": p, "has_drum": False, "drum_notes": 0}
        return {"path": p, "has_drum": True, "drum_notes": drum_note_count(mt, di)}
    except Exception as e:
        return {"path": p, "has_drum": False, "drum_notes": 0, "error": str(e)}

def pass0_collect_counts(npz_files: List[Path], workers: int = 8) -> List[Dict]:
    if workers <= 1:
        out = []
        for p in tqdm(npz_files, desc="[scan] npz", unit="file"):
            out.append(pass0_one(p))
        return out
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(tqdm(ex.map(pass0_one, npz_files),
                         total=len(npz_files), desc="[scan] npz", unit="file"))

def compute_outlier_thresholds(counts: List[int]) -> Tuple[float, float, float, float]:
    """Returns (mean, std, low, high) where low=mean-2*std, high=mean+2*std."""
    if len(counts) == 0:
        return 0.0, 0.0, -math.inf, math.inf
    arr = np.asarray(counts, dtype=float)
    mu = float(arr.mean())
    sd = float(arr.std(ddof=0))
    lo = mu - 2.0 * sd
    hi = mu + 2.0 * sd
    return mu, sd, lo, hi

# -------------------------
# Cleaning pipeline
# -------------------------

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
):
    log("INFO",  f"[paths] DATA_ROOT={data_root}")
    log("INFO",  f"[paths] LPD_DIR={lpd_dir}")
    log("INFO",  f"[paths] TRASH_DIR={trash_dir}")
    log("INFO",  f"[conf ] duration_check={duration_check}  workers={workers}  write_clean={int(write_clean)}  resume={int(resume)}")
    if duration_check == "synth" and not SOUND_FONT.exists():
        log("WARN", f"SoundFont not found at {SOUND_FONT}. 'synth' duration check will fail.",)

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

    # ---- PASS 0: drum-note counts in parallel ----
    meta = pass0_collect_counts(all_npz, workers=workers)
    counts = [m["drum_notes"] for m in meta if m.get("has_drum") and m["drum_notes"] >= MIN_NOTES_HARD]
    mu, sd, lo, hi = compute_outlier_thresholds(counts)
    log("INFO", f"[stats] drum-note count: mean={mu:.1f} std={sd:.1f}  keep-range=[{max(lo,0):.1f}, {hi:.1f}]")

    # First, immediately trash the obvious rejects (no/empty drum or out-of-range)
    kept_stems = 0
    trashed_stems = 0

    candidates: List[Dict] = []
    for m in meta:
        p = m["path"]
        stem = p.stem

        # Rule A: must have a drum track with >= MIN_NOTES_HARD notes
        if not m.get("has_drum") or m["drum_notes"] < MIN_NOTES_HARD:
            log("INFO", f"[drop] {p.name} (no/empty drum track)")
            if not dry_run:
                move_to_trash(p, trash_dir)
                append_line(trash_file, stem)
            trashed_stems += 1
            continue

        # Rule B: within mean±2*std
        if not (lo <= m["drum_notes"] <= hi):
            log("INFO", f"[drop] {p.name} (drum notes={m['drum_notes']} out of [{lo:.1f},{hi:.1f}])")
            if not dry_run:
                move_to_trash(p, trash_dir)
                append_line(trash_file, stem)
            trashed_stems += 1
            continue

        candidates.append(m)

    log("INFO", f"[filter] drum-count candidates: {len(candidates)} / {len(meta)}")

    # Now apply (1) duration consistency and (3) 16th quantization on survivors
    for m in tqdm(candidates, desc="[clean] files", unit="file"):
        p = m["path"]
        stem = p.stem

        try:
            mt = load_multitrack(p, debug=debug)
            di = find_drum_track_idx(mt)
            if di is None:
                log("INFO", f"[drop] {p.name} (no/empty drum track)")
                if not dry_run:
                    move_to_trash(p, trash_dir)
                    append_line(trash_file, stem)
                trashed_stems += 1
                continue

            if not duration_consistency_ok(mt, mode=duration_check, debug=debug):
                log("INFO", f"[drop] {p.name} (inconsistent duration)")
                if not dry_run:
                    move_to_trash(p, trash_dir)
                    append_line(trash_file, stem)
                trashed_stems += 1
                continue

            kept_notes, removed_notes = quantize_drum_track_inplace(mt, di)
            total = kept_notes + removed_notes
            removed_pct = (100.0 * removed_notes / max(1, total))
            log("DEBUG", f"[quant] {p.name}: kept={kept_notes} removed={removed_notes} ({removed_pct:.2f}%)", debug)

            if not dry_run and write_clean:
                out_clean = p.with_suffix("").with_name(p.stem + ".clean.npz")
                try:
                    ppr.save(str(out_clean), mt)  # pypianoroll >= 1.0
                except Exception:
                    # fallback: pack the essentials in a compatible layout
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
            if not dry_run:
                append_line(kept_file, stem)

        except Exception as e:
            log("ERROR", f"[clean] {p.name}: {e}")
            if not dry_run:
                try:
                    move_to_trash(p, trash_dir)
                    append_line(trash_file, stem)
                except Exception:
                    pass
            trashed_stems += 1

    # Also count the already-done items to give a full summary
    total_all = len(all_npz_all)
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
    ap = argparse.ArgumentParser(description="Clean LPD dataset per Wei et al. (2019).")
    ap.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT,
                    help="Root directory containing 'lpd/' and where manifests will be written.")
    ap.add_argument("--lpd_subdir", type=str, default=LPD_SUBDIR,
                    help="Relative subfolder under data_root containing .npz files")
    ap.add_argument("--trash_subdir", type=str, default=TRASH_SUBDIR,
                    help="Relative subfolder under data_root for trashed files")
    ap.add_argument("--dry_run", type=parse_bool, default=False,
                    help="If True, skip file moves/writes; just log decisions.")
    ap.add_argument("--duration_check", choices=["synth", "symbolic", "none"], default="symbolic",
                    help="How to verify duration consistency (synth is slow; symbolic is fast; none skips).")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel workers for the initial scan (drum note counting).")
    ap.add_argument("--write_clean", type=parse_bool, default=False,
                    help="If True, write '<stem>.clean.npz' next to kept originals.")
    ap.add_argument("--resume", type=parse_bool, default=True,
                    help="If True, skip stems that already appear in kept/trash manifests.")
    ap.add_argument("--debug", type=parse_bool, default=False,
                    help="Verbose debug logs (loader path, quantization stats, etc.).")

    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    lpd_dir   = data_root / args.lpd_subdir
    trash_dir = data_root / args.trash_subdir

    if not lpd_dir.exists():
        log("ERROR", f"LPD directory not found: {lpd_dir}")
        sys.exit(1)

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
    )

if __name__ == "__main__":
    main()
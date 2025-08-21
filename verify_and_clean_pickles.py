#!/usr/bin/env python3
import argparse, os, sys, pickle, glob, time, traceback
from pathlib import Path

# Optional: tqdm for a nice progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

import numpy as np

def extract_mel_stem(p: Path) -> str:
    return p.stem.replace("song_barlv_ssm_", "")

def extract_drm_stem(p: Path) -> str:
    return p.stem.replace("song_barlv_drum_ssm_", "")

def is_good_array(x) -> bool:
    try:
        arr = np.asarray(x)
        return (arr.ndim == 2) and (arr.shape == (256, 256)) and np.issubdtype(arr.dtype, np.floating)
    except Exception:
        return False

def check_pickle(path: Path):
    """Return (ok, msg). ok=False if unpickling fails or array is malformed."""
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except Exception as e:
        return False, f"unpickling_error: {e.__class__.__name__}: {e}"
    if not is_good_array(obj):
        try:
            arr = np.asarray(obj)
            return False, f"bad_shape_or_dtype: shape={arr.shape}, dtype={arr.dtype}"
        except Exception:
            return False, "bad_object_type"
    return True, "ok"

def main():
    ap = argparse.ArgumentParser(description="Find & (optionally) delete corrupted/malformed SSM pickles.")
    ap.add_argument("--mel-dir", required=True, help="Folder with melodic SSM pickles (song_barlv_ssm_*.pkl)")
    ap.add_argument("--drum-dir", required=True, help="Folder with drum SSM pickles (song_barlv_drum_ssm_*.pkl)")
    ap.add_argument("--delete", action="store_true", help="Actually delete bad/corrupted files")
    ap.add_argument("--delete-unmatched", action="store_true", help="Also delete pickles that are missing its pair")
    ap.add_argument("--clean-tmp", action="store_true", help="Remove leftover *.pkl.tmp files")
    ap.add_argument("--show", type=int, default=20, help="Max items to list for each category")
    args = ap.parse_args()

    mel_dir = Path(args.mel_dir)
    drm_dir = Path(args.drum_dir)

    if not mel_dir.is_dir() or not drm_dir.is_dir():
        print(f"[ERR] One of the directories does not exist:\n  mel: {mel_dir}\n  drm: {drm_dir}")
        sys.exit(1)

    mel_pkls = sorted(mel_dir.glob("song_barlv_ssm_*.pkl"))
    drm_pkls = sorted(drm_dir.glob("song_barlv_drum_ssm_*.pkl"))

    # Build stem sets for pairing check
    mel_stems = {extract_mel_stem(p) for p in mel_pkls}
    drm_stems = {extract_drm_stem(p) for p in drm_pkls}
    common = mel_stems & drm_stems
    only_mel = sorted(mel_stems - drm_stems)
    only_drm = sorted(drm_stems - mel_stems)

    print(f"[info] mel files: {len(mel_pkls)}  drum files: {len(drm_pkls)}  matched stems: {len(common)}")
    if only_mel:
        print(f"[warn] mel-only stems: {len(only_mel)} (showing up to {args.show}) -> {only_mel[:args.show]}")
    if only_drm:
        print(f"[warn] drum-only stems: {len(only_drm)} (showing up to {args.show}) -> {only_drm[:args.show]}")

    bad_files = []
    def scan(files, label):
        iterator = files
        if tqdm:
            iterator = tqdm(files, desc=f"[scan {label}]", unit="file")
        for p in iterator:
            ok, msg = check_pickle(p)
            if not ok:
                bad_files.append((p, msg))
        return

    # Scan both dirs
    scan(mel_pkls, "mel")
    scan(drm_pkls, "drum")

    if bad_files:
        print(f"[BAD] corrupted/malformed pickles: {len(bad_files)} (showing up to {args.show})")
        for p, msg in bad_files[:args.show]:
            try:
                sz = p.stat().st_size
            except Exception:
                sz = -1
            print(f"  - {p}  ({sz/1e6:.2f} MB)  -> {msg}")
    else:
        print("[OK] No corrupted/malformed pickles found.")

    # Clean *.tmp leftovers if requested
    tmp_leftovers = []
    if args.clean_tmp:
        for d in [mel_dir, drm_dir]:
            tmp_leftovers.extend(sorted(d.glob("*.pkl.tmp")))
        if tmp_leftovers:
            print(f"[info] tmp leftovers: {len(tmp_leftovers)} (showing up to {args.show})")
            for p in tmp_leftovers[:args.show]:
                print(f"  - {p}")
            if args.delete:
                for p in tmp_leftovers:
                    try:
                        p.unlink()
                    except Exception as e:
                        print(f"[err] failed to remove tmp {p}: {e}")

    # Delete bad files (and optionally unmatched) if asked
    if args.delete:
        # Delete the corrupted/malformed ones
        for p, _ in bad_files:
            try:
                p.unlink()
                print(f"[del] removed {p}")
            except Exception as e:
                print(f"[err] failed to delete {p}: {e}")

        if args.delete_unmatched:
            # Delete unpaired mel/drum pickles (helps keep dataset consistent)
            for stem in only_mel:
                p = mel_dir / f"song_barlv_ssm_{stem}.pkl"
                if p.exists():
                    try:
                        p.unlink()
                        print(f"[del] removed unmatched mel {p}")
                    except Exception as e:
                        print(f"[err] failed to delete {p}: {e}")
            for stem in only_drm:
                p = drm_dir / f"song_barlv_drum_ssm_{stem}.pkl"
                if p.exists():
                    try:
                        p.unlink()
                        print(f"[del] removed unmatched drm {p}")
                    except Exception as e:
                        print(f"[err] failed to delete {p}: {e}")

    print("[done] sweep complete.")

if __name__ == "__main__":
    main()

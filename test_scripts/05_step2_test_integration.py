import os
import pickle
import importlib
import numpy as np
import torch

def _write_fake_pair(tmp_dir, idx: int):
    """Write one mel + drum pkl pair with correct filenames and shapes."""
    # 256x256 is important because your down/upsampling pipeline expects 8 downsamples to 1x1
    H = W = 256
    mel = np.random.rand(H, W).astype(np.float32)
    drum = np.random.rand(H, W).astype(np.float32)

    mel_name  = tmp_dir / f"song_barlv_ssm_{idx:04d}.pkl"
    drum_name = tmp_dir / f"song_barlv_drum_ssm_{idx:04d}.pkl"

    with open(mel_name, "wb") as f:
        pickle.dump(mel, f)
    with open(drum_name, "wb") as f:
        pickle.dump(drum, f)

def test_step2_end_to_end(tmp_path, monkeypatch):
    # Force CPU to make CI happy
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    # Create fake input dirs/files
    mel_dir  = tmp_path / "pre_processed_data" / "bar_level_cqt_ssm"
    drum_dir = tmp_path / "pre_processed_data" / "bar_level_drum_ssm"
    out_dir  = tmp_path / "pre_processed_data"
    mel_dir.mkdir(parents=True, exist_ok=True)
    drum_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_out_drum_ssm_pkg.pkl"

    # Two fake pairs
    _write_fake_pair(mel_dir, 1)
    _write_fake_pair(mel_dir, 2)
    # Copy names to drum_dir (already written there by _write_fake_pair if we pointed it there)
    # For clarity, create drum pairs directly in the drum_dir:
    for i in [1, 2]:
        src = mel_dir / f"song_barlv_ssm_{i:04d}.pkl"
        dst = drum_dir / f"song_barlv_drum_ssm_{i:04d}.pkl"
        with open(src, "rb") as f: mel = pickle.load(f)
        with open(dst, "wb") as f: pickle.dump(np.flipud(mel).astype(np.float32), f)  # different content

    # Build a tiny model and checkpoint in the tmp tree that matches your code path
    ckpt_dir = tmp_path / "checkpoints" / "ssm_generator"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "best.pt"

    models = importlib.import_module("models")
    if hasattr(models, "SSMEncoder") and hasattr(models, "SSMDecoder") and hasattr(models, "SSMVAE"):
        enc = models.SSMEncoder(in_channels=1, base_channels=8, latent_dim=8)
        dec = models.SSMDecoder(out_channels=1, base_channels=8, latent_dim=8)
        vae = models.SSMVAE(enc, dec).eval()
        torch.save({"vae": vae.state_dict()}, ckpt)
    elif hasattr(models, "SSMArranger"):
        # If your step_2 script is arranger-based, store its state dict under simple key
        arr = models.SSMArranger(in_channels=1, base=8, out_channels=1).eval()
        torch.save(arr.state_dict(), ckpt)
    else:
        raise RuntimeError("No compatible model classes found in models.py")

    # Import your step_2 script as a module
    step2 = importlib.import_module("step_2_generate_drum_ssm")

    # Call its main() with temp paths via monkeypatched argv
    import sys
    argv = [
        "prog",
        "--mel_dir", str(mel_dir),
        "--drum_dir", str(drum_dir),
        "--ckpt", str(ckpt),
        "--out", str(out_path),
        "--device", "cpu",
        "--batch_size", "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    step2.main()

    # Verify outputs
    assert out_path.exists(), "Output PKL not created"
    with open(out_path, "rb") as f:
        cqt, drum_gt, drum_pred = pickle.load(f)

    assert cqt.shape[0] == 2 and cqt.shape[1:] == (256, 256)
    assert drum_gt.shape == cqt.shape
    assert drum_pred.shape == cqt.shape
    # Light sanity: predictions are finite
    assert np.isfinite(drum_pred).all()

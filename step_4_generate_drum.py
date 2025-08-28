import os
import pickle
import numpy as np
from tqdm import tqdm
import argparse

import torch
from torch.utils.data import DataLoader
from models import DrumVAE, DrumEncoder, DrumDecoder
from utils import (
    load_bar_index_list,
    read_bar_feature_package,
    save_midi_from_drum_pattern,
    evaluate_note_score,
    downsample_256x256_to_46x16,
)

# ---------------------------------------------
# CONFIGURATION
# ---------------------------------------------
data_root = "/data/pre_processed_data"
out_root = "./generated_data"
ckpt_path = "./checkpoints/drum_generator/best.pt"

add_note_levels = [0, 3, 6, 12, 20]
batch_size = 64

bar_index_list_path = os.path.join(data_root, "abs_bar_idx_str_list.pkl")
bar_feature_dir = os.path.join(data_root, "bar_feature_data")
out_dir = os.path.join(out_root, "drum_generation_results")
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------
# LOAD MODEL
# ---------------------------------------------

encoder = DrumEncoder(in_channels=8, base_channels=64, latent_dim=32)
decoder = DrumDecoder(out_channels=1, base_channels=64, latent_dim=32)
model = DrumVAE(encoder, decoder)

# ---------------------------------------------
# MAIN GENERATION LOOP
# ---------------------------------------------
def run_generation(device=None):
    with open(bar_index_list_path, 'rb') as f:
        bar_idx_list = pickle.load(f)

    print(f"[info] Total {len(bar_idx_list)} bars to process")

    for add_note_val in add_note_levels:
        print(f"[info] Generating with +{add_note_val} note complexity")
        
        all_inputs, all_targets, all_outputs = [], [], []

        for idx_str in tqdm(bar_idx_list):
            # Load per-bar conditioning features
            cqt_tensor, note_ratio, tempo_vec, style_vec, progress_vec, note_count, target_drum = \
                read_bar_feature_package(bar_feature_dir, idx_str)

            # Modify note count with additive value
            note_count_mod = np.clip(note_count + add_note_val, 0.0, 200.0).reshape(1,1).astype(np.float32)

            # Prepare 8-channel spectrogram (B=1)
            # prepare input
            cqt_tensor = torch.tensor(cqt_tensor).unsqueeze(0).float().to(device)

            with torch.no_grad():
                mu, logvar, c_hat = model.encoder(cqt_tensor)
                # overwrite c_hat with modified one
                c_hat[...] = torch.tensor(note_count_mod, dtype=torch.float32).to(device)
                z = model.reparameterize(mu, logvar)
                pred = model.decoder(z, c_hat).squeeze(0).cpu().numpy()  # (46, 16)
                pred_downsampled = downsample_256x256_to_46x16(pred)

            # Binarize
            pred_bin = (pred_downsampled >= 0.5).astype(np.float32)  # (46, 16)

            # Record
            all_inputs.append(cqt_tensor.squeeze(0).cpu().numpy())
            all_targets.append(target_drum)
            all_outputs.append(pred_bin)

        # Save .mid
        out_mid_path = os.path.join(out_dir, f"drums_addnote_{add_note_val}.mid")
        save_midi_from_drum_pattern(np.array(all_outputs), out_mid_path)

        preds = np.array(all_outputs)
        targets = np.array(all_targets)
        print(f"[debug] preds.shape  = {preds.shape}")
        print(f"[debug] targets.shape = {targets.shape}")

        # Evaluate
        score = evaluate_note_score(np.array(all_outputs), np.array(all_targets))
        print(f"[info] AddNote {add_note_val} - Note Score: {score*100:.2f}%")


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu'")
    args = ap.parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["vae"])
    model.to(device)
    model.eval()
    
    run_generation(device=device)
import os
import pickle
import numpy as np
from pretty_midi import PrettyMIDI, Instrument, Note

def read_bar_feature_package(bar_feature_dir, index_str):
    """
    Loads 8-bar CQT + 1-hot attributes + drum ground truth for a given bar index.
    index_str: e.g. "2_030"
    Returns:
        (8, 84, 96) spectrogram stack
        (7,) similarity weights (not used here)
        (10,) tempo one-hot
        (15,) style one-hot
        (10,) song progress one-hot
        (1,) note count
        (46, 16) ground truth drum pattern
    """
    song_idx = int(index_str.split('_')[0])
    bar_idx = int(index_str.split('_')[1])

    fname = os.path.join(bar_feature_dir, f"{song_idx:05d}.pkl")
    with open(fname, 'rb') as f:
        song_data = pickle.load(f)

    bar_cqt = song_data['bar_cqt'][bar_idx]  # (84, 96)
    rel_idxs = song_data['bar_selection_index'][bar_idx][:7]  # 7 nearest
    rel_weights = song_data['bar_selection_weight'][bar_idx][:7]  # (7,)

    cqt_stack = [bar_cqt]
    for idx in rel_idxs:
        cqt_stack.append(song_data['bar_cqt'][idx])
    cqt_stack = np.stack(cqt_stack, axis=0).astype(np.float32)  # (8, 84, 96)

    tempo = song_data['tempo'].astype(np.float32)            # (10,)
    style = song_data['style'].astype(np.float32)            # (15,)
    song_prog = song_data['song_progress'][bar_idx].astype(np.float32)  # (10,)
    note_cnt = np.array([song_data['note_count'][bar_idx]]).astype(np.float32)  # (1,)
    target_drum = song_data['drum_roll'][bar_idx].astype(np.float32)   # (46, 16)

    return cqt_stack, rel_weights, tempo, style, song_prog, note_cnt, target_drum

def evaluate_note_score(preds, targets):
    """
    Compute mean accuracy score between binary predictions and targets.
    """
    assert preds.shape == targets.shape
    total = np.prod(preds.shape)
    mismatch = np.sum(np.abs(preds - targets))
    return 1.0 - (mismatch / total)

def save_midi_from_drum_pattern(drum_array, out_path):
    """
    Convert binary (N, 46, 16) drum patterns into MIDI file.
    """
    drum_array = np.squeeze(drum_array)  # handle accidental (N, 46, 16, 1)

    pm = PrettyMIDI()
    drum = Instrument(program=0, is_drum=True)
    ticks_per_step = 0.125  # assuming 120 bpm, 16 steps/bar

    for bar_idx, bar in enumerate(drum_array):  # bar: (46, 16)
        for pitch_idx in range(46):
            for step in range(16):
                value = bar[pitch_idx, step]
                if isinstance(value, np.ndarray):
                    value = value.item()  # safely extract scalar

                if value >= 0.5:
                    pitch = 35 + pitch_idx
                    start = (bar_idx * 16 + step) * ticks_per_step
                    end = start + 0.1
                    drum.notes.append(Note(velocity=100, pitch=pitch, start=start, end=end))

    pm.instruments.append(drum)
    pm.write(out_path)

def load_bar_index_list(path):
    """Load abs_bar_idx_str_list.pkl into a list of strings like '2_030'."""
    with open(path, 'rb') as f:
        return pickle.load(f)
    
# inside utils.py (for testing)
def read_bar_feature_package(bar_feature_dir, index_str):
    cqt_stack = np.random.rand(8, 84, 96).astype(np.float32)
    rel_weights = np.random.rand(7).astype(np.float32)
    tempo = np.eye(10)[np.random.choice(10)].astype(np.float32)
    style = np.eye(15)[np.random.choice(15)].astype(np.float32)
    song_prog = np.eye(10)[np.random.choice(10)].astype(np.float32)
    note_cnt = np.array([np.random.randint(1, 200)]).astype(np.float32)
    target_drum = np.random.randint(0, 2, size=(46, 16)).astype(np.float32)
    return cqt_stack, rel_weights, tempo, style, song_prog, note_cnt, target_drum

def load_bar_index_list(path):
    return ["00002_030", "00002_031", "00003_010"]

def downsample_256x256_to_46x16(image):
    """
    Downsamples a 256x256 image to a 46x16 binary matrix using average pooling.
    Assumes input shape is (1, 256, 256) or (256, 256).
    """
    import torch.nn.functional as F
    import torch

    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()

    if image.ndim == 3:
        image = image.unsqueeze(0)  # (1, 1, 256, 256)
    elif image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)

    image = F.adaptive_avg_pool2d(image, output_size=(46, 16))
    return image.squeeze().numpy()  # → (46, 16)
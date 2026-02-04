import os
import glob
from pathlib import Path
import math

from tqdm import tqdm
import torch
import librosa
import numpy as np
import sys

# Add custom module path
sys.path.append('path/to/GTsinger_tech_recognition')

from modules.pe.rmvpe import RMVPE
from utils.audio import get_wav_num_frames
from utils.commons.dataset_utils import batch_by_size, optimized_batch_by_size
from concurrent.futures import ThreadPoolExecutor

# --- Configuration ---
pe_ckpt = 'checkpoints/rmvpe_model.pt'
rmvpe = None

sr = 16000           # Sampling rate
hop_size = 160       # Hop size for feature extraction

wav_dir = 'suno70k/audio'
out_dir = 'suno70k/audio_F0'

# Collect WAV file paths and map them to item names
item_names = []
items = {}

# Iterate through sorted mp3 files
for wav_path in sorted(glob.glob(f"{wav_dir}/*.mp3")):
    item_name = Path(wav_path).stem
    item_names.append(item_name)
    items[item_name] = {'wav_path': wav_path}

# Initialize RMVPE (Robust Pitch Estimation) model
if rmvpe is None:
    rmvpe = RMVPE(pe_ckpt, device='cuda')

# Helper function to get audio frame count metadata
def get_wav_info(item_name, wav_path, sr, hop_size):
    total_frames = get_wav_num_frames(wav_path, sr)
    return item_name, round(total_frames / hop_size)

# Collect audio sizes for optimized batching
id_and_sizes = []
with ThreadPoolExecutor(max_workers=32) as executor:  # Parallel processing of metadata
    futures = [
        executor.submit(get_wav_info, item_name, items[item_name]['wav_path'], sr, hop_size)
        for idx, item_name in enumerate(item_names)
    ]
    for idx, future in enumerate(tqdm(futures, total=len(futures), desc="Collecting WAV metadata")):
        item_name, size = future.result()
        id_and_sizes.append((idx, size))

# Optimized batching configuration to manage GPU memory usage
get_size = lambda x: x[1]
max_tokens = 360000
max_sentences = 128  # Batch size limit per iteration
bs = optimized_batch_by_size(id_and_sizes, get_size, max_tokens=max_tokens, max_sentences=max_sentences)

# --- F0 Extraction and Persistence ---
for batch in tqdm(bs, total=len(bs), desc=f'| Processing F0 [max_tokens={max_tokens}; max_sentences={max_sentences}]'):
    wavs, lengths = [], []
    
    for idx in batch:
        item_name = item_names[idx]
        item = items[item_name]
        
        # Check for existing output to avoid redundant processing
        output_path = os.path.join(out_dir, f"{item_name.replace('_vocals', '')}.pt")
        if os.path.exists(output_path):
            continue
            
        wav_fn = item['wav_path']
        # Load audio (librosa loads as mono by default)
        wav, _ = librosa.core.load(wav_fn, sr=sr)
        wavs.append(wav)
        lengths.append(math.ceil((wav.shape[0]) / hop_size))

    if len(wavs) == 0:
        continue
        
    # Perform pitch estimation in batch mode
    with torch.no_grad():
        f0s, indexs, cents = rmvpe.get_pitch_batch(
            wavs, 
            sample_rate=sr,
            hop_size=hop_size,
            lengths=lengths,
            fmax=900,
            fmin=50
        )
    torch.cuda.empty_cache()

    # Save F0 features and cent information to disk
    os.makedirs(out_dir, exist_ok=True)
    for i in range(len(f0s)):
        item_name = item_names[batch[i]]
        output_path = os.path.join(out_dir, f"{item_name}.pt")
        torch.save({'f0': f0s[i], 'cent': cents[i]}, output_path)

# --- Resource Cleanup ---
if rmvpe is not None:
    rmvpe.release_cuda()
    torch.cuda.empty_cache()

print("Processing completed.")
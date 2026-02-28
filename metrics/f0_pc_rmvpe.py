# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import librosa

import numpy as np

from torchmetrics import PearsonCorrCoef
import sys
sys.path.append('/path/to/Songecho_code/metrics')
from utils.util import JsonHParams
from utils.f0 import get_f0_features_using_parselmouth, get_pitch_sub_median
from utils.rmvpe import RMVPE
import math
import mir_eval

pe_ckpt = "checkpoints/rmvpe_model.pt"
rmvpe = None
# sr = 48000
# hop_size = 256 #和原本24000-128最后的token数目相同，ROSVOT的处理
sr = 16000
hop_size = 160
if rmvpe is None:
    rmvpe = RMVPE(pe_ckpt, device='cuda')
def extract_fpc_v2(
    audio_ref,
    audio_deg,
    hop_size=160,
    f0_min=50,
    f0_max=1100,
    model=None,
    **kwargs,
):
    """Compute F0 Pearson Distance (FPC) between the predicted and the ground truth audio.
    audio_ref: path to the ground truth audio.
    audio_deg: path to the predicted audio.
    fs: sampling rate.
    hop_length: hop length.
    f0_min: lower limit for f0.
    f0_max: upper limit for f0.
    pitch_bin: number of bins for f0 quantization.
    pitch_max: upper limit for f0 quantization.
    pitch_min: lower limit for f0 quantization.
    need_mean: subtract the mean value from f0 if "True".
    method: "dtw" will use dtw algorithm to align the length of the ground truth and predicted audio.
            "cut" will cut both audios into a same length according to the one with the shorter length.
    """
    # Load hyperparameters
    kwargs = kwargs["kwargs"]
    fs = kwargs["fs"]
    method = kwargs["method"]
    need_mean = kwargs["need_mean"]

    # Initialize method
    pearson = PearsonCorrCoef()

    # Load audio
    if fs != None:
        audio_ref, _ = librosa.load(audio_ref, sr=fs)
        audio_deg, _ = librosa.load(audio_deg, sr=fs)
    else:
        audio_ref, fs = librosa.load(audio_ref)
        audio_deg, fs = librosa.load(audio_deg)

    f0_ref, uvres = model.get_pitch(
        audio_ref,
        sample_rate=fs,
        hop_size=hop_size,
        length=math.ceil((audio_ref.shape[0]) / hop_size),
        fmin=f0_min,
        fmax=f0_max,
    )

    f0_deg, uvres = model.get_pitch(
        audio_deg,
        sample_rate=fs,
        hop_size=hop_size,
        length=math.ceil((audio_deg.shape[0]) / hop_size),
        fmin=f0_min,
        fmax=f0_max,
    )
    # f0_deg = torch.load("/data/lisifei/SongEval/data/suno/audio_F0/c4995e4d-01cd-40a6-bfc7-e4a94e984fca.pt", weights_only=False)['f0']
    # breakpoint()
    # Subtract mean value from f0
    if need_mean:
        f0_ref = torch.from_numpy(f0_ref)
        f0_deg = torch.from_numpy(f0_deg)

        f0_ref = get_pitch_sub_median(f0_ref).numpy()
        f0_deg = get_pitch_sub_median(f0_deg).numpy()

    # Avoid silence
    min_length = min(len(f0_ref), len(f0_deg))
    if min_length <= 1:
        return 1
    # breakpoint()
    # F0 length alignment
    if method == "cut":
        length = min(len(f0_ref), len(f0_deg))
        f0_ref = f0_ref[:length]
        f0_deg = f0_deg[:length]
    elif method == "dtw":
        _, wp = librosa.sequence.dtw(f0_ref, f0_deg, backtrack=True)
        f0_gt_new = []
        f0_pred_new = []
        for i in range(wp.shape[0]):
            gt_index = wp[i][0]
            pred_index = wp[i][1]
            f0_gt_new.append(f0_ref[gt_index])
            f0_pred_new.append(f0_deg[pred_index])
        f0_ref = np.array(f0_gt_new)
        f0_deg = np.array(f0_pred_new)
        assert len(f0_ref) == len(f0_deg)

    # Convert to tensor
    f0_ref = torch.from_numpy(f0_ref)
    f0_deg = torch.from_numpy(f0_deg)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        f0_ref = f0_ref.to(device)
        f0_deg = f0_deg.to(device)
        pearson = pearson.to(device)
    #vrs, vfas, rpas, rcas, oas
    f0_gt = freq_gt = f0_ref.cpu().numpy()
    f0_pred = freq_pred = f0_deg.cpu().numpy()
    step = hop_size / sr
    t_gt = np.arange(0, step * len(f0_gt), step)
    t_pred = np.arange(0, step * len(f0_pred), step)

    if len(t_gt) > len(freq_gt):
        t_gt = t_gt[:len(freq_gt)]
    else:
        freq_gt = freq_gt[:len(t_gt)]
    if len(t_pred) > len(freq_pred):
        t_pred = t_pred[:len(freq_pred)]
    else:
        freq_pred = freq_pred[:len(t_pred)]

    ref_voicing, ref_cent, est_voicing, est_cent = mir_eval.melody.to_cent_voicing(t_gt, freq_gt,
                                                                                   t_pred, freq_pred)
    vr, vfa = mir_eval.melody.voicing_measures(ref_voicing,
                                               est_voicing)  # voicing recall, voicing false alarm
    rpa = mir_eval.melody.raw_pitch_accuracy(ref_voicing, ref_cent, est_voicing, est_cent)
    rca = mir_eval.melody.raw_chroma_accuracy(ref_voicing, ref_cent, est_voicing, est_cent)
    oa = mir_eval.melody.overall_accuracy(ref_voicing, ref_cent, est_voicing, est_cent)
    print(f"VR: {vr}, VFA: {vfa}, RPA: {rpa}, RCA: {rca}, OA: {oa}")

    return {
        "RPA": rpa,
        "RCA": rca,
        "OA": oa
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract FPC")
    parser.add_argument("--audio_ref", type=str, required=True, help="Path to reference audio")
    parser.add_argument("--audio_deg", type=str, required=True, help="Path to degraded audio")
    parser.add_argument("--hop_size", type=int, default=160, help="Hop length for f0 extraction")
    parser.add_argument("--f0_min", type=int, default=50, help="Minimum f0 value")
    parser.add_argument("--f0_max", type=int, default=900, help="Maximum f0 value")
    args = parser.parse_args()

    fpc = extract_fpc(
        args.audio_ref,
        args.audio_deg,
        hop_size=args.hop_size,
        f0_min=args.f0_min,
        f0_max=args.f0_max,
        model=rmvpe,
        kwargs={"fs": 16000, "method": "cut", "need_mean": False, "kwargs": {}},
    )
    print(f"FPC: {fpc}")
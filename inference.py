
import torch
import torchaudio
import argparse
import os
import random
from safetensors.torch import load_file
from acestep.pipeline_ace_step import ACEStepPipeline
from torch.nn import functional as F
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from acestep.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from acestep.apg_guidance import apg_forward, MomentumBuffer
from torch.utils.data import DataLoader
from acestep.text2music_dataset import Text2MusicDataset
from tqdm import tqdm
import json
from peft import LoraConfig
import torch.nn as nn
import numpy as np
from metrics.utils.rmvpe import RMVPE
from metrics.f0_pc_rmvpe import extract_fpc_v2



def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class MelodyEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, kernel_size=5):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding='same'),
            nn.ReLU()
        )
        for param in self.conv_stack.parameters():
            param.requires_grad = True

    def forward(self, normalized_f0):
        # f0 shape: (B, T, 1) -> (B, 1, T)
        f0_transposed = normalized_f0.transpose(1, 2)
        # conv_out shape: (B, hidden_dim, T)
        conv_out = self.conv_stack(f0_transposed)
        return conv_out
    
class Pipeline(torch.nn.Module):
    def __init__(
        self,
        num_workers: int = 4,
        T: int = 1000,
        shift: float = 3.0,
        checkpoint_dir=None,
        dataset_path: str = "./data/your_dataset_path",
        lora_config_path: str = None,
        adapter_name: str = "lora_adapter",
        pt_path: str = None,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()

        self.T = T
        self.shift = shift
        self.dataset_path = dataset_path
        self.num_workers = num_workers
        self.device = device

        # breakpoint()
        # step 1: load model
        acestep_pipeline = ACEStepPipeline(checkpoint_dir, cpu_offload=True)
        acestep_pipeline.load_checkpoint(acestep_pipeline.checkpoint_dir)
        # breakpoint()
        transformers = acestep_pipeline.ace_step_transformer.float().cpu()

        transformers.requires_grad_(False)
        for block in transformers.transformer_blocks:
            block.ff.proj_c = nn.Linear(256, 256)
            block.ff.proj_x = nn.Linear(2560, 256)
            block.ff.cond_project = nn.Sequential(
                nn.SiLU(),
                zero_module(nn.Linear(256, 2560 * 2, bias=False))
            )
        
        if pt_path is not None:
            # breakpoint()
            if pt_path.endswith(".safetensors"):
                melody_state_dict = load_file(pt_path, device="cpu")
                transformers.load_state_dict(melody_state_dict)
                print(f"Loaded melody conditioner parameters from {pt_path}")
            else:
                melody_state_dict = torch.load(pt_path, map_location="cpu")
                state_dict_to_load = {}
                for name, param in melody_state_dict.items():
                    if "cond_project" in name or "proj_c" in name or "proj_x" in name:
                        state_dict_to_load[name] = param
                missing, unexpected = transformers.load_state_dict(state_dict_to_load, strict=False)
                print(f"Loaded melody conditioner parameters from {pt_path}")
                if unexpected:
                    print(f"Unexpected keys: {unexpected}")

        if lora_config_path is not None:
            with open(lora_config_path, encoding="utf-8") as f:
                import json
                lora_config = json.load(f)
            lora_config = LoraConfig(**lora_config)
            transformers.add_adapter(adapter_config=lora_config, adapter_name=adapter_name)
            self.adapter_name = adapter_name

        self.transformers = transformers

        self.dcae = acestep_pipeline.music_dcae.float().cpu()
        self.dcae.requires_grad_(False)

        self.text_encoder_model = acestep_pipeline.text_encoder_model.float().cpu()
        self.text_encoder_model.requires_grad_(False)
        self.text_tokenizer = acestep_pipeline.text_tokenizer

        self.melody_encoder = MelodyEncoder().float().cpu()
        if pt_path is not None:
            melody_encoder_path = pt_path.replace("melody.pt", "melody_encoder.pt")
            if os.path.exists(melody_encoder_path):
                melody_encoder_state_dict = torch.load(melody_encoder_path, map_location="cpu")
                self.melody_encoder.load_state_dict(melody_encoder_state_dict)
                print(f"Loaded melody encoder parameters from {melody_encoder_path}")

    def get_text_embeddings(self, texts, device, text_max_length=256):
        inputs = self.text_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=text_max_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        if self.text_encoder_model.device != device:
            self.text_encoder_model.to(device)
        with torch.no_grad():
            outputs = self.text_encoder_model(**inputs)
            last_hidden_states = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]

        return last_hidden_states, attention_mask
    def melody_preprocess(self, melodys):

        uv_flag = (melodys> 0).float()  
        log_f0 = torch.zeros_like(melodys)
        voiced_mask = melodys > 0
        log_f0[voiced_mask] = torch.log(melodys[voiced_mask])

        f0_min_log = 3.912023005 #50hz
        f0_max_log = 6.802394763 #900hz
        normalized_f0 = (log_f0 - f0_min_log) / (f0_max_log - f0_min_log)
        normalized_f0[~voiced_mask] = 0.0
        melody_condition = torch.cat([normalized_f0, uv_flag], dim=-1) # shape: (B, T, 2)
        return melody_condition
    
    def preprocess(self, batch):
        target_wavs = batch["target_wavs"].to(self.device)
        wav_lengths = batch["wav_lengths"].to(self.device)

        dtype = target_wavs.dtype
        bs = target_wavs.shape[0]
        device = target_wavs.device


        # 1: text embedding
        texts = batch["prompts"]
        encoder_text_hidden_states, text_attention_mask = self.get_text_embeddings(
            texts, device
        )
        encoder_text_hidden_states = encoder_text_hidden_states.to(dtype)

        target_latents, _ = self.dcae.encode(target_wavs, wav_lengths)
        attention_mask = torch.ones(
            bs, target_latents.shape[-1], device=device, dtype=dtype
        )

        #melody condition
        if len(batch["melodys"]) != 0:
            melodys = batch["melodys"]
            melodys = melodys.to(device).to(dtype)
            # breakpoint()
            melodys = melodys.transpose(1,2)
            melodys = self.melody_preprocess(melodys)
            melodys = self.melody_encoder(melodys)
            # melodys = melodys * mask
            melodys = F.interpolate(melodys, size=target_latents.shape[-1], mode="linear", align_corners=False)
            melodys = melodys.transpose(1, 2)  # (B, C, T) -> (B, T, C)
            
        # breakpoint()
        speaker_embds = batch["speaker_embs"].to(device).to(dtype)
        keys = batch["keys"]
        lyric_token_ids = batch["lyric_token_ids"].to(device)
        lyric_mask = batch["lyric_masks"].to(device)

        return (
            keys,
            target_latents,
            attention_mask,
            encoder_text_hidden_states,
            text_attention_mask,
            speaker_embds,
            lyric_token_ids,
            lyric_mask,
            melodys if len(batch["melodys"]) != 0 else None,
        )

    @torch.no_grad()
    def diffusion_process(
        self,
        duration,
        encoder_text_hidden_states,
        text_attention_mask,
        speaker_embds,
        lyric_token_ids,
        lyric_mask,
        random_generators=None,
        infer_steps=60,
        guidance_scale=15.0,
        omega_scale=10.0,
        encoder_hidden_states_con=None,
    ):
        seed_num = 1234
        torch.manual_seed(seed_num)
        do_classifier_free_guidance = True
        if guidance_scale == 0.0 or guidance_scale == 1.0:
            do_classifier_free_guidance = False

        device = encoder_text_hidden_states.device
        dtype = encoder_text_hidden_states.dtype
        bsz = encoder_text_hidden_states.shape[0]

        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=3.0,
        )
        if duration is None:
            frame_length = encoder_hidden_states_con.shape[1]
        else:
            frame_length = int(duration * 44100 / 512 / 8)
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler, num_inference_steps=infer_steps, device=device, timesteps=None
        )

        target_latents = randn_tensor(
            shape=(bsz, 8, 16, frame_length),
            generator=random_generators,
            device=device,
            dtype=dtype,
        )
        attention_mask = torch.ones(bsz, frame_length, device=device, dtype=dtype)
        if do_classifier_free_guidance:
            attention_mask = torch.cat([attention_mask] * 3, dim=0)
            encoder_text_hidden_states = torch.cat(
                [
                    encoder_text_hidden_states,
                    encoder_text_hidden_states,
                    # torch.zeros_like(encoder_text_hidden_states),
                    torch.zeros_like(encoder_text_hidden_states),
                ],
                0,
            )
            text_attention_mask = torch.cat([text_attention_mask] * 3, dim=0)

            speaker_embds = torch.cat(
                [speaker_embds, torch.zeros_like(speaker_embds), torch.zeros_like(speaker_embds)], 0
            )

            lyric_token_ids = torch.cat(
                [torch.zeros_like(lyric_token_ids), lyric_token_ids,  torch.zeros_like(lyric_token_ids)], 0
            )
            lyric_mask = torch.cat([torch.zeros_like(lyric_mask), lyric_mask, torch.zeros_like(lyric_mask)], 0)

            encoder_hidden_states_con = (
                torch.cat(
                    [   
                        torch.zeros_like(encoder_hidden_states_con),
                        encoder_hidden_states_con,
                        torch.zeros_like(encoder_hidden_states_con),
                    ],
                    dim=0,
                )
                if encoder_hidden_states_con is not None
                else None
            )
        momentum_buffer = MomentumBuffer()

        for i, t in tqdm(enumerate(timesteps), total=num_inference_steps):
            # expand the latents if we are doing classifier free guidance
            latents = target_latents
            latent_model_input = (
                torch.cat([latents] * 3) if do_classifier_free_guidance else latents
            )
            timestep = t.expand(latent_model_input.shape[0])
            noise_pred = self.transformers(
                hidden_states=latent_model_input,
                attention_mask=attention_mask,
                encoder_text_hidden_states=encoder_text_hidden_states,
                text_attention_mask=text_attention_mask,
                speaker_embeds=speaker_embds,
                lyric_token_idx=lyric_token_ids,
                lyric_mask=lyric_mask,
                timestep=timestep,
                encoder_hidden_states_con=encoder_hidden_states_con,
            ).sample

            if do_classifier_free_guidance:
                noise_pred_with_text, noise_pred_melody, noise_pred_uncond = noise_pred.chunk(3)
                noise_pred_with_cond = 0.5*noise_pred_with_text + 0.5*noise_pred_melody 
                noise_pred = apg_forward(
                    pred_cond=noise_pred_with_cond,
                    pred_uncond=noise_pred_uncond,
                    guidance_scale=guidance_scale,
                    momentum_buffer=momentum_buffer,
                )

            target_latents = scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=target_latents,
                return_dict=False,
                omega=omega_scale,
            )[0]

        return target_latents

    def predict_step(self, batch):
        (
            keys,
            target_latents,
            attention_mask,
            encoder_text_hidden_states,
            text_attention_mask,
            speaker_embds,
            lyric_token_ids,
            lyric_mask,
            melodys,
        ) = self.preprocess(batch)

        infer_steps = 60
        guidance_scale = 15.0
        omega_scale = 10.0
        seed_num = 1234
        torch.manual_seed(seed_num)
        seeds = [seed_num]

        if melodys is not None:
            duration = None
        else:
            duration = 240
        pred_latents = self.diffusion_process(
            duration=duration,
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embds=speaker_embds,
            lyric_token_ids=lyric_token_ids,
            lyric_mask=lyric_mask,
            random_generators=None,
            infer_steps=infer_steps,
            guidance_scale=guidance_scale,
            omega_scale=omega_scale,
            encoder_hidden_states_con=melodys,
        )

        audio_lengths = batch["wav_lengths"]
        sr, pred_wavs = self.dcae.decode(
            pred_latents, audio_lengths=audio_lengths, sr=48000
        )
        return {
            "target_wavs": batch["target_wavs"],
            "pred_wavs": pred_wavs,
            "keys": keys,
            "prompts": batch["prompts"],
            "candidate_lyric_chunks": batch["candidate_lyric_chunks"],
            "sr": sr,
            "seeds": seeds,
        }
    def val_dataloader(self):
        dataset = Text2MusicDataset(
            train=False,
            train_dataset_path=self.dataset_path,
            load_melody=True,
            sample_size=90,
            minibatch_size=1,
        )
        return DataLoader(
            dataset,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            batch_size=1,
        )
    def construct_lyrics(self, candidate_lyric_chunk):
        lyrics = []
        for chunk in candidate_lyric_chunk:
            lyrics.append(chunk["lyric"])

        lyrics = "\n".join(lyrics)
        return lyrics
            
def main(args):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(42)
    np.random.seed(42)
     # Initialize model
    model = Pipeline(
        checkpoint_dir=args.checkpoint_dir,
        pt_path=args.pt_path,
        lora_config_path=args.lora_config_path,
        dataset_path=args.dataset_path,
        num_workers=args.num_workers,
    )
    model.to(device=args.device)
    model.eval()
    rmvpe = RMVPE("checkpoints/rmvpe_model.pt", device=f'cuda')

    # Load validation dataset
    dataloader = model.val_dataloader()
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    fpcs = []
    key_fpc_dict = {}  # [ADDED]
    metrics_dict = {}
    metric_lists = { "RPA": [], "RCA": [], "OA": []}
    target_dir = os.path.join(output_dir,"target_wav")
    pred_dir = os.path.join(output_dir,"pred_wav")
    lyric_save_dir = os.path.join(output_dir,"lyrics")
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(lyric_save_dir, exist_ok=True)
    # Process each batch
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        results = model.predict_step(batch)

        target_wavs = results["target_wavs"]
        pred_wavs = results["pred_wavs"]
        keys = results["keys"]
        prompts = results["prompts"]
        candidate_lyric_chunks = results["candidate_lyric_chunks"]
        sr = results["sr"]
        seeds = results["seeds"]
        
        for i, (key, target_wav, pred_wav, prompt, candidate_lyric_chunk, seed) in enumerate(
            zip(keys, target_wavs, pred_wavs, prompts, candidate_lyric_chunks, seeds)
        ):
            lyric = model.construct_lyrics(candidate_lyric_chunk)
            key_prompt_lyric = f"# KEY\n\n{key}\n\n\n# PROMPT\n\n{prompt}\n\n\n# LYRIC\n\n{lyric}\n\n# SEED\n\n{seed}\n\n"


            # Save audio files
            torchaudio.save(
                os.path.join(target_dir, f"{key}_{i}.mp3"),
                target_wav.float().cpu(),
                sr,
            )
            torchaudio.save(
                os.path.join(pred_dir, f"{key}_{i}.mp3"),
                pred_wav.float().cpu(),
                sr,
            )

            # Save metadata
            with open(
                os.path.join(lyric_save_dir, f"key_prompt_lyric_{key}_{i}.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(key_prompt_lyric)
            metrics = extract_fpc_v2(
                    audio_ref=f"{target_dir}/{key}_{i}.mp3",
                    audio_deg=f"{pred_dir}/{key}_{i}.mp3",
                    f0_min=50,
                    f0_max=900,
                    model=rmvpe,
                    kwargs={"fs": 16000, "method": "cut", "need_mean": False, "kwargs": {}},
                )
            metrics_dict[f"{key}_{i}"] = metrics
            for metric_name in metric_lists:
                metric_lists[metric_name].append(metrics[metric_name])
    
        mean_metrics = {key: np.mean(np.array(values)) if np.array(values).size > 0 else 0.0 for key, values in metric_lists.items()}

            
        for metric_name, mean_value in mean_metrics.items():
            print(f"eval/{metric_name.lower()}_mean: {mean_value}")


    print(f"Saved results to {output_dir}")
    metrics_output_path = os.path.join(output_dir, "key_metrics.json")
    with open(metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Saved metrics to {metrics_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Base checkpoint dir for ACEStepPipeline")
    parser.add_argument("--pt_path", type=str, default="exps/logs/lightning_logs/2025-08-09_08-33-36suno/checkpoints/epoch=0-step=32000/melody.pt", help="Path to melody.pt for melody conditioner parameters")
    parser.add_argument("--lora_config_path", type=str, default=None, help="Path to LoRA config JSON if used")
    parser.add_argument("--dataset_path", type=str, default="suno_dataset_prompt2_val", help="Path to Text2MusicDataset")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Directory to save generated audio and metadata")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--infer_steps", type=int, default=60, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=15.0, help="Guidance scale")
    parser.add_argument("--omega_scale", type=float, default=10.0, help="Omega scale")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    args = parser.parse_args()
    main(args)
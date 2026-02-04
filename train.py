
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from datetime import datetime
import argparse
import torch
import json
import matplotlib
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning.core import LightningModule
from torch.utils.data import DataLoader
from acestep.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from acestep.text2music_dataset import Text2MusicDataset
from loguru import logger
from transformers import AutoModel, Wav2Vec2FeatureExtractor
import torchaudio
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)
from diffusers.utils.torch_utils import randn_tensor
from acestep.apg_guidance import apg_forward, MomentumBuffer
from tqdm import tqdm
import random
import os
from acestep.pipeline_ace_step import ACEStepPipeline
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from safetensors.torch import save_file, load_file
import itertools
from metrics.utils.rmvpe import RMVPE
from metrics.f0_pc_rmvpe import extract_fpc_v2
import numpy as np
from pytorch_lightning.utilities.model_summary import summarize
import math
import torch.distributed as dist

matplotlib.use("Agg")
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")

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
    
class Pipeline(LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-4,
        num_workers: int = 4,
        train: bool = True,
        T: int = 1000,
        weight_decay: float = 1e-2,
        every_plot_step: int = 2000,
        shift: float = 3.0,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        timestep_densities_type: str = "logit_normal",
        ssl_coeff: float = 1.0,
        checkpoint_dir=None,
        max_steps: int = 200000,
        warmup_steps: int = 10,
        train_dataset_path: str = "./data/your_dataset_path",
        val_dataset_path: str = "./data/your_dataset_path",
        lora_config_path: str = None,
        adapter_name: str = "lora_adapter",
        pt_path: str = None,
        batch_size: int = 1,
        sample_size: int = 20,
        output_dir: str = "./exps/logs/outputs/",
    ):
        super().__init__()

        self.save_hyperparameters()
        self.is_train = train
        self.T = T

        # Initialize scheduler
        self.scheduler = self.get_scheduler()
        # breakpoint()
        # step 1: load model
        acestep_pipeline = ACEStepPipeline(checkpoint_dir, cpu_offload=True)
        acestep_pipeline.load_checkpoint(acestep_pipeline.checkpoint_dir)
        # breakpoint()
        transformers = acestep_pipeline.ace_step_transformer.float().cpu()
        transformers.enable_gradient_checkpointing()

        transformers.requires_grad_(False)
        for block in transformers.transformer_blocks:
            block.ff.proj_c = nn.Linear(256, 256)
            block.ff.proj_x = nn.Linear(2560, 256)
            block.ff.cond_project = nn.Sequential(
                nn.SiLU(),
                zero_module(nn.Linear(256, 2560 * 2, bias=False))
            )
            block.ff.proj_c.requires_grad_(True)
            block.ff.proj_x.requires_grad_(True)
            block.ff.cond_project.requires_grad_(True)
        
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
                # if missing:
                #     print(f"Missing keys: {missing}")
                if unexpected:
                    print(f"Unexpected keys: {unexpected}")

        if lora_config_path is not None:
            try:
                from peft import LoraConfig
            except ImportError:
                raise ImportError("Please install peft library to use LoRA training")
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


        self.rmvpe = RMVPE("checkpoints/melody_encoder.pt", device=f'cuda:{self.local_rank}')

        self.melody_encoder = MelodyEncoder().float().cpu()
        if pt_path is not None:
            melody_encoder_path = pt_path.replace("melody.pt", "melody_encoder.pt")
            if os.path.exists(melody_encoder_path):
                melody_encoder_state_dict = torch.load(melody_encoder_path, map_location="cpu")
                self.melody_encoder.load_state_dict(melody_encoder_state_dict)
                print(f"Loaded melody encoder parameters from {melody_encoder_path}")

        if self.is_train:
            self.transformers.train()
            self.melody_encoder.train()
            self.ssl_coeff = ssl_coeff

        # Print model summary if local_rank == 0
        # if self.local_rank == 0:
        #     print(summarize(self))

        # if self.local_rank == 1 or torch.distributed.get_rank() == 1 or torch.cuda.current_device() == 1:
        num_train = 0
        for name, param in self.transformers.named_parameters():
            if param.requires_grad:
                num_train += 1
                print(f"Parameter: {name}, requires_grad: {param.requires_grad}, shape: {param.shape}")
        
        for name, param in self.melody_encoder.named_parameters():
            if param.requires_grad:
                num_train += 1
                print(f"Parameter: {name}, requires_grad: {param.requires_grad}, shape: {param.shape}")
                # if not param.requires_grad:
                #     print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
        print(f"Total trainable parameters: {num_train}")

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

        uv_flag = (melodys> 0).float()  # (B, T, 1)

        log_f0 = torch.zeros_like(melodys)
        voiced_mask = melodys > 0
        log_f0[voiced_mask] = torch.log(melodys[voiced_mask])

        f0_min_log = 3.912023005 #50hz
        f0_max_log = 6.802394763 #900hz
        normalized_f0 = (log_f0 - f0_min_log) / (f0_max_log - f0_min_log)
        normalized_f0[~voiced_mask] = 0.0
        melody_condition = torch.cat([normalized_f0, uv_flag], dim=-1) # shape: (B, T, 2)
        return melody_condition
    
    def preprocess(self, batch, train=True):
        target_wavs = batch["target_wavs"]
        wav_lengths = batch["wav_lengths"]

        dtype = target_wavs.dtype
        bs = target_wavs.shape[0]
        device = target_wavs.device

        # SSL constraints
        mert_ssl_hidden_states = None
        mhubert_ssl_hidden_states = None

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
            # breakpoint()
            # print(f"melody shape:{melodys.shape}")
            melodys = melodys.transpose(1,2)
            melodys = self.melody_preprocess(melodys)
            melodys = self.melody_encoder(melodys)
            melodys = F.interpolate(melodys, size=target_latents.shape[-1], mode="linear", align_corners=False)
            melodys = melodys.transpose(1, 2)  # (B, C, T) -> (B, T, C)
            
        # breakpoint()
        speaker_embds = batch["speaker_embs"].to(dtype)
        keys = batch["keys"]
        lyric_token_ids = batch["lyric_token_ids"]
        lyric_mask = batch["lyric_masks"]

        return (
            keys,
            target_latents,
            attention_mask,
            encoder_text_hidden_states,
            text_attention_mask,
            speaker_embds,
            lyric_token_ids,
            lyric_mask,
            mert_ssl_hidden_states,
            mhubert_ssl_hidden_states,
            melodys if batch.get("melodys") is not None else None,
        )

    def get_scheduler(self):
        return FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.T,
            shift=self.hparams.shift,
        )

    def configure_optimizers(self):
        # Collect trainable parameters from transformers
        transformer_params = [
            p for name, p in self.transformers.named_parameters() if p.requires_grad
        ]
        # Collect trainable parameters from melody_encoder
        melody_encoder_params = [
            p for name, p in self.melody_encoder.named_parameters() if p.requires_grad
        ]
        # Combine all trainable parameters
        trainable_params = transformer_params + melody_encoder_params

        optimizer = torch.optim.AdamW(
            params=[
                {"params": trainable_params},
            ],
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95),
        )
        max_steps = self.hparams.max_steps
        warmup_steps = self.hparams.warmup_steps  # New hyperparameter for warmup steps

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup from 0 to initial_lr
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine annealing after warmup
                progress = (current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
                return 0.5 * (1.0 + math.cos(math.pi * progress))  # eta_min=0

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def train_dataloader(self):
        self.train_dataset = Text2MusicDataset(
            train=True,
            train_dataset_path=self.hparams.train_dataset_path,
            load_melody=True,
            minibatch_size=1
        
        )
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.collate_fn,
            batch_size=self.hparams.batch_size,  
        )
    def val_dataloader(self):
        self.val_dataset = Text2MusicDataset(
            train=False,  
            train_dataset_path=self.hparams.val_dataset_path,
            load_melody=True,
            sample_size=1,
            minibatch_size=1
        )
        return DataLoader(
            self.val_dataset,
            shuffle=False,  
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self.val_dataset.collate_fn,
            batch_size=1,
        )
      
    def real_val_dataloader(self):
        self.real_val_dataset = Text2MusicDataset(
            train=False,
            train_dataset_path=self.hparams.val_dataset_path,
            load_melody=True,
            sample_size=self.hparams.sample_size,  # 默认 20
            minibatch_size=1
        )
        return DataLoader(
            self.real_val_dataset,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self.real_val_dataset.collate_fn,
            batch_size=1,
            sampler=DistributedSampler(self.real_val_dataset, shuffle=False) if self.trainer.num_devices > 1 else None
        )
    def get_sd3_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):
        sigmas = self.scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def get_timestep(self, bsz, device):
        if self.hparams.timestep_densities_type == "logit_normal":
            # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
            # In practice, we sample the random variable u from a normal distribution u ∼ N (u; m, s)
            # and map it through the standard logistic function
            u = torch.normal(
                mean=self.hparams.logit_mean,
                std=self.hparams.logit_std,
                size=(bsz,),
                device="cpu",
            )
            u = torch.nn.functional.sigmoid(u)
            indices = (u * self.scheduler.config.num_train_timesteps).long()
            indices = torch.clamp(
                indices, 0, self.scheduler.config.num_train_timesteps - 1
            )
            timesteps = self.scheduler.timesteps[indices].to(device)

        return timesteps

    def run_step(self, batch, batch_idx):
        # breakpoint()
        # self.plot_step(batch, batch_idx)
        (
            keys,
            target_latents,
            attention_mask,
            encoder_text_hidden_states,
            text_attention_mask,
            speaker_embds,
            lyric_token_ids,
            lyric_mask,
            mert_ssl_hidden_states,
            mhubert_ssl_hidden_states,
            melodys,
        ) = self.preprocess(batch)

        target_image = target_latents
        device = target_image.device
        dtype = target_image.dtype
        # Step 1: Generate random noise, initialize settings
        noise = torch.randn_like(target_image, device=device)
        bsz = target_image.shape[0]
        timesteps = self.get_timestep(bsz, device)

        # Add noise according to flow matching.
        sigmas = self.get_sd3_sigmas(
            timesteps=timesteps, device=device, n_dim=target_image.ndim, dtype=dtype
        )
        noisy_image = sigmas * noise + (1.0 - sigmas) * target_image

        # This is the flow-matching target for vanilla SD3.
        target = target_image

        # SSL constraints for CLAP and vocal_latent_channel2
        all_ssl_hiden_states = []
        if mert_ssl_hidden_states is not None:
            all_ssl_hiden_states.append(mert_ssl_hidden_states)
        if mhubert_ssl_hidden_states is not None:
            all_ssl_hiden_states.append(mhubert_ssl_hidden_states)

        # N x H -> N x c x W x H
        x = noisy_image
        # Step 5: Predict noise
        transformer_output = self.transformers(
            hidden_states=x,
            attention_mask=attention_mask,
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embeds=speaker_embds,
            lyric_token_idx=lyric_token_ids,
            lyric_mask=lyric_mask,
            timestep=timesteps.to(device).to(dtype),
            ssl_hidden_states=all_ssl_hiden_states,
            encoder_hidden_states_con=melodys,
        )
        model_pred = transformer_output.sample
        proj_losses = transformer_output.proj_losses

        # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
        # Preconditioning of the model outputs.
        model_pred = model_pred * (-sigmas) + noisy_image

        # Compute loss. Only calculate loss where chunk_mask is 1 and there is no padding
        # N x T x 64
        # N x T -> N x c x W x T
        mask = (
            attention_mask.unsqueeze(1)
            .unsqueeze(1)
            .expand(-1, target_image.shape[1], target_image.shape[2], -1)
        )

        selected_model_pred = (model_pred * mask).reshape(bsz, -1).contiguous()
        selected_target = (target * mask).reshape(bsz, -1).contiguous()

        loss = F.mse_loss(selected_model_pred, selected_target, reduction="none")
        loss = loss.mean(1)
        loss = loss * mask.reshape(bsz, -1).mean(1)
        loss = loss.mean()

        prefix = "train"

        self.log(
            f"{prefix}/denoising_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,  
            rank_zero_only=True,
        )

        total_proj_loss = 0.0
        for k, v in proj_losses:
            self.log(
                f"{prefix}/{k}_loss", v, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True, rank_zero_only=True
            )
            total_proj_loss += v

        if len(proj_losses) > 0:
            total_proj_loss = total_proj_loss / len(proj_losses)

        loss = loss + total_proj_loss * self.ssl_coeff
        self.log(f"{prefix}/loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True, rank_zero_only=True)
        # print(f"Step {self.global_step}, {prefix} loss: {loss.item():.4f}")
        # Log learning rate if scheduler exists
        if self.lr_schedulers() is not None:
            learning_rate = self.lr_schedulers().get_last_lr()[0]
            self.log(
                f"{prefix}/learning_rate",
                learning_rate,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
        # del noise, noisy_image, model_pred, selected_model_pred, selected_target, transformer_output
        # torch.cuda.empty_cache()
        return loss

    def training_step(self, batch, batch_idx):
        return self.run_step(batch, batch_idx)

    def on_save_checkpoint(self, checkpoint):
        # breakpoint()
        state = {}
        if (self.local_rank != 0
            or torch.distributed.get_rank() != 0
            or torch.cuda.current_device() != 0):
            return state
        
        log_dir = os.path.join(self.hparams.output_dir, self.hparams.adapter_name)
        epoch = self.current_epoch
        step = self.global_step
        checkpoint_name = f"epoch={epoch}-step={step}"
        checkpoint_dir = os.path.join(log_dir, "checkpoints", checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        os.makedirs(os.path.join(self.logger.log_dir, "checkpoints", "nockpt"), exist_ok=True)
        
        
        grad_params = {
            name: param for name, param in self.transformers.named_parameters() if param.requires_grad
        }
        grad_params_path = os.path.join(checkpoint_dir, "melody.pt")
        torch.save(grad_params, grad_params_path)
        print(f"Saved {len(grad_params)} grad_params to {grad_params_path}")
        
        # Save melody encoder parameters separately
        melody_encoder_params = {
            name: param for name, param in self.melody_encoder.named_parameters() if param.requires_grad
        }
        melody_encoder_path = os.path.join(checkpoint_dir, "melody_encoder.pt")
        torch.save(melody_encoder_params, melody_encoder_path)
        print(f"Saved {len(melody_encoder_params)} melody encoder params to {melody_encoder_path}")
        
        state["grad_params_path"] = grad_params_path
        state["melody_encoder_path"] = melody_encoder_path
        # torch.cuda.empty_cache()
        return state

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
                    torch.zeros_like(encoder_text_hidden_states),
                ],
                0,
            )
            text_attention_mask = torch.cat([text_attention_mask] * 3, dim=0)

            speaker_embds = torch.cat(
                [speaker_embds, torch.zeros_like(speaker_embds), torch.zeros_like(speaker_embds)], 0
            )

            lyric_token_ids = torch.cat(
                [torch.zeros_like(lyric_token_ids), lyric_token_ids, torch.zeros_like(lyric_token_ids)], 0
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
            mert_ssl_hidden_states,
            mhubert_ssl_hidden_states,
            melodys,
        ) = self.preprocess(batch, train=False)

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

    def construct_lyrics(self, candidate_lyric_chunk):
        lyrics = []
        for chunk in candidate_lyric_chunk:
            lyrics.append(chunk["lyric"])

        lyrics = "\n".join(lyrics)
        return lyrics
    
    def plot_step(self, batch, batch_idx):
        # breakpoint()
        log_dir = os.path.join(self.hparams.output_dir, self.hparams.adapter_name)
        os.makedirs(log_dir, exist_ok=True)
        save_dir = f"{log_dir}/eval_results/step_{self.global_step}"
        os.makedirs(save_dir, exist_ok=True)
        val_dataloader = self.real_val_dataloader()
        fpcs = []
        iterator = tqdm(val_dataloader, desc="Processing batches", total=len(val_dataloader)) if self.local_rank == 0 else val_dataloader

        metrics_dict = {}
        metric_lists = { "RPA": [], "RCA": [], "OA": []}

        for batch_idx, batch in enumerate(iterator):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            result = self.predict_step(batch)

            target_wavs = result["target_wavs"]
            pred_wavs = result["pred_wavs"]
            keys = result["keys"]
            prompts = result["prompts"]
            candidate_lyric_chunks = result["candidate_lyric_chunks"]
            sr = result["sr"]
            seeds = result["seeds"]

            for i, (key, target_wav, pred_wav, prompt, candidate_lyric_chunk, seed) in enumerate(zip(
                keys, target_wavs, pred_wavs, prompts, candidate_lyric_chunks, seeds
            )):
                lyric = self.construct_lyrics(candidate_lyric_chunk)
                key_prompt_lyric = f"# KEY\n\n{key}\n\n\n# PROMPT\n\n{prompt}\n\n\n# LYRIC\n\n{lyric}\n\n# SEED\n\n{seed}\n\n"
                
                # torchaudio.save(f"{save_dir}/target_wav_{key}_{i}.wav", target_wav.float().cpu(), sr)
                # torchaudio.save(f"{save_dir}/pred_wav_{key}_{i}.wav", pred_wav.float().cpu(), sr)
                target_dir = os.path.join(save_dir, f"target_wav")
                pred_dir = os.path.join(save_dir, f"pred_wav")
                key_prompt_lyric_dir = os.path.join(save_dir, f"key_prompt_lyric")
                os.makedirs(target_dir, exist_ok=True)
                os.makedirs(pred_dir, exist_ok=True)
                os.makedirs(key_prompt_lyric_dir, exist_ok=True)
                torchaudio.save(f"{target_dir}/{key}_{i}.mp3", target_wav.float().cpu(), sr)
                torchaudio.save(f"{pred_dir}/{key}_{i}.mp3", pred_wav.float().cpu(), sr)
                with open(f"{key_prompt_lyric_dir}/{key}_{i}.txt", "w", encoding="utf-8") as f:
                    f.write(key_prompt_lyric)
                print(f"Rank {self.local_rank}: Saved eval result for key {key}_{batch_idx}_{i} in {save_dir}")

                metrics = extract_fpc_v2(
                    audio_ref=f"{target_dir}/{key}_{i}.mp3",
                    audio_deg=f"{pred_dir}/{key}_{i}.mp3",
                    f0_min=50,
                    f0_max=900,
                    model=self.rmvpe,
                    kwargs={"fs": 16000, "method": "cut", "need_mean": False, "kwargs": {}},
                )
                metrics_dict[f"{key}_{i}"] = metrics
                for metric_name in metric_lists:
                    metric_lists[metric_name].append(metrics[metric_name])
                if self.local_rank == 0:
                    print(f"Rank {self.local_rank}: Key {key}_{batch_idx}_{i}, Metrics: {metrics}")
            
        if self.trainer.num_devices > 1:
            metric_tensors = {
                key: torch.tensor(values, device=self.device) for key, values in metric_lists.items()
            }
            gathered_metrics = {
                key: [torch.zeros_like(tensor) for _ in range(self.trainer.num_devices)]
                for key, tensor in metric_tensors.items()
            }
            for key in metric_tensors:
                torch.distributed.all_gather(gathered_metrics[key], metric_tensors[key])
                gathered_metrics[key] = torch.cat(gathered_metrics[key]).cpu().numpy()
        else:
            gathered_metrics = {key: np.array(values) for key, values in metric_lists.items()}

        if self.local_rank == 0:
            mean_metrics = {key: np.mean(values) if values.size > 0 else 0.0 for key, values in gathered_metrics.items()}
            print(f"Rank {self.local_rank}: Mean Metrics: {mean_metrics}")
            
            for metric_name, mean_value in mean_metrics.items():
                self.log(
                    f"eval/{metric_name.lower()}_mean",
                    mean_value,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    sync_dist=False,  # 已手动汇总
                    rank_zero_only=True,
                )

    def validation_step(self, batch, batch_idx):
        self.transformers.eval()
        self.plot_step(batch, batch_idx)
        self.transformers.train()
        torch.cuda.empty_cache()

            
def main(args):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(42)
    np.random.seed(42)
    model = Pipeline(
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        shift=args.shift,
        max_steps=args.max_steps,
        every_plot_step=args.every_plot_step,
        # dataset_path=args.dataset_path,
        train_dataset_path=args.train_dataset_path,
        val_dataset_path=args.val_dataset_path,
        checkpoint_dir=args.checkpoint_dir,
        adapter_name=args.exp_name,
        lora_config_path=args.lora_config_path,
        pt_path=args.pt_path,
        batch_size=args.batch_size,
        sample_size=args.sample_size,
        output_dir=args.output_dir,
        warmup_steps=args.warmup_steps,
    )
    # breakpoint()
    checkpoint_callback = ModelCheckpoint(
        monitor=None,
        every_n_train_steps=args.every_n_train_steps,
        save_top_k=-1,
    )
    # add datetime str to version
    logger_callback = TensorBoardLogger(
        version=datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + args.exp_name,
        save_dir=args.logger_dir,
    )
    trainer = Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        strategy="ddp_find_unused_parameters_true",
        # strategy="ddp",
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        log_every_n_steps=1,
        logger=logger_callback,
        callbacks=[checkpoint_callback],
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        reload_dataloaders_every_n_epochs=args.reload_dataloaders_every_n_epochs,
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=None,
        # limit_val_batches=0,
    )


    trainer.fit(
        model,
        ckpt_path=args.ckpt_path,
    )

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--num_nodes", type=int, default=1)
    args.add_argument("--shift", type=float, default=3.0)
    args.add_argument("--learning_rate", type=float, default=1e-4)
    args.add_argument("--num_workers", type=int, default=2)
    args.add_argument("--epochs", type=int, default=-1)
    args.add_argument("--max_steps", type=int, default=40000)
    args.add_argument("--every_n_train_steps", type=int, default=2000)
    args.add_argument("--train_dataset_path", type=str, default="./zh_lora_dataset")
    args.add_argument("--val_dataset_path", type=str, default="./zh_lora_dataset")
    args.add_argument("--exp_name", type=str, default="chinese_rap_lora")
    args.add_argument("--precision", type=str, default="32")
    args.add_argument("--accumulate_grad_batches", type=int, default=1)
    args.add_argument("--devices", type=int, default=1)
    args.add_argument("--logger_dir", type=str, default="./exps/logs/")
    args.add_argument("--ckpt_path", type=str, default=None)
    args.add_argument("--checkpoint_dir", type=str, default=None)
    args.add_argument("--gradient_clip_val", type=float, default=0.5)
    args.add_argument("--gradient_clip_algorithm", type=str, default="norm")
    args.add_argument("--reload_dataloaders_every_n_epochs", type=int, default=1)
    args.add_argument("--every_plot_step", type=int, default=2000)
    args.add_argument("--val_check_interval", type=int, default=2000)
    args.add_argument("--lora_config_path", type=str, default=None)
    args.add_argument("--pt_path", type=str, default=None)
    args.add_argument("--batch_size", type=int, default=2)
    args.add_argument("--sample_size", type=int, default=20)
    args.add_argument("--output_dir", type=str, default="./exps/logs/outputs/")
    args.add_argument("--warmup_steps", type=int, default=500)
    # args.add_argument("--lora_config_path", type=str, default="config/zh_rap_lora_config.json")
    args = args.parse_args()
    main(args)

from dataclasses import dataclass
from packaging import version
import os

from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as ttf

from utils import set_attn_processors, compute_snr


class Pipeline:

    def __init__(self, 
                 pretrained_model='CompVis/stable-diffusion-v1-4',
                 self_attn_only=False,
                 disable_prompts=False,
                 onepass=False,
                 prediction_type='v_prediction',
                 use_oracle_ddim=False,
                 lora_ckpt=None,
                 enable_xformers=False,
                 device='cuda',
                 mixed_precision='',
    ):
        self.pretrained_model = pretrained_model
        self.self_attn_only = self_attn_only
        self.disable_prompts = disable_prompts
        self.use_oracle_ddim = use_oracle_ddim
        self.enable_xformers = enable_xformers
        self.onepass = onepass
        self.device = device

        # Load scheduler, tokenizer and models.
        self.tokenizer = CLIPTokenizer.from_pretrained(
                pretrained_model, subfolder='tokenizer')
        self.text_encoder = CLIPTextModel.from_pretrained(
                pretrained_model, subfolder='text_encoder')
        self.vae = AutoencoderKL.from_pretrained(
                pretrained_model, subfolder='vae')
        self.unet = UNet2DConditionModel.from_pretrained(
                pretrained_model, subfolder='unet')
        self.noise_scheduler = DDPMScheduler.from_pretrained(
                pretrained_model, subfolder='scheduler')
        if prediction_type:
            if onepass:
                assert prediction_type == 'sample'
            # set prediction_type of scheduler if defined
            self.noise_scheduler.register_to_config(prediction_type=prediction_type)

        # Freeze parameters of models to save more memory
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype = torch.float32
        if mixed_precision == 'fp16':
            self.weight_dtype = torch.float16
        elif mixed_precision == 'bf16':
            self.weight_dtype = torch.bfloat16

        # Move unet, vae and text_encoder to device and cast to weight_dtype
        self.unet.to(device, dtype=self.weight_dtype)
        self.vae.to(device, dtype=self.weight_dtype)
        self.text_encoder.to(device, dtype=self.weight_dtype)

        # Convert the original attention processors to xformer
        if self.enable_xformers:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse('0.0.16'):
                    logger.warn(
                        'xFormers 0.0.16 cannot be used for training in some GPUs. '
                        'If you observe problems during training, please update xFormers to at least 0.0.17. '
                        'See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details.'
                    )
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError('xformers is not available. Make sure it is installed correctly')
        self.orig_attn_procs = self.unet.attn_processors
        self.define_and_load_lora(lora_ckpt, reset_first=False)
        self.has_prepared_infer = False

    def define_and_load_lora(self, ckpt, reset_first=False, self_attn_only=None):
        if self_attn_only is not None:
            self.self_attn_only = self_attn_only
        else:
            self_attn_only = self.self_attn_only
        if reset_first:
            set_attn_processors(self.unet, self.orig_attn_procs)

        if ckpt and os.path.isdir(ckpt): # automatically define lora from the state dict
            self.unet.load_attn_procs(ckpt)
        else: # add new LoRA weights to the attention layers
            # It's important to realize here how many attention weights will be added and of which sizes
            # The sizes of the attention layers consist only of two different variables:
            # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
            # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

            # Let's first see how many attention processors we will have to set.
            # For Stable Diffusion, it should be equal to:
            # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
            # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
            # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
            # => 32 layers

            # Set correct lora layers
            self.lora_attn_procs = {}
            for name in self.unet.attn_processors.keys():
                if not self_attn_only or name.endswith('attn1.processor'):
                    cross_attention_dim = None if name.endswith('attn1.processor') else self.unet.config.cross_attention_dim
                    if name.startswith('mid_block'):
                        hidden_size = self.unet.config.block_out_channels[-1]
                    elif name.startswith('up_blocks'):
                        block_id = int(name[len('up_blocks.')])
                        hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
                    elif name.startswith('down_blocks'):
                        block_id = int(name[len('down_blocks.')])
                        hidden_size = self.unet.config.block_out_channels[block_id]

                    self.lora_attn_procs[name] = LoRAAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        rank=4,
                    )
            set_attn_processors(self.unet, self.lora_attn_procs)
            if ckpt is not None:
                miss, unexp = self.unet.load_state_dict(torch.load(ckpt, map_location=self.device), strict=False)
                if len(unexp):
                    print('Unexpected:', unexp)

        # Convert the lora attention processors to xformers
        if self.enable_xformers:
            self.unet.enable_xformers_memory_efficient_attention()

        self.unet.to(self.device)
        self.lora_attn_procs = self.unet.attn_processors

        self.lora_layers = AttnProcsLayers(
                {n: p for n, p in self.unet.attn_processors.items() \
                if not self_attn_only or n.endswith('attn1.processor')})

    def trainable_parameters(self):
        return self.lora_layers

    def prepare_infer(self):
        if self.use_oracle_ddim:
            from ddim import DDIMScheduler
        else:
            from diffusers import DDIMScheduler
        self.scheduler = DDIMScheduler.from_pretrained(
                self.pretrained_model, subfolder='scheduler')

        # Encode unconditional text
        with torch.no_grad():
            uncond_input = self.tokenizer([''], padding='max_length',
                max_length=self.tokenizer.model_max_length, return_tensors='pt').input_ids
            self.uncond_embeds = self.text_encoder(uncond_input.to(self.device))[0]
        self.has_prepared_infer = True

    @torch.no_grad()
    def infer_batch(
            self,
            text_inputs=None,
            init_latents=None,
            src_imgs=None,
            src_inference_steps=50,
            trg_inference_steps=10,
            src_guidance_scale=7.5,
            trg_guidance_scale=2,
            generator=None,
            return_mid_latents=False,
    ):
        if init_latents is None and src_imgs is None and text_inputs is None:
            raise ValueError('cannot generate images')

        if not self.has_prepared_infer:
            self.prepare_infer()

        if text_inputs is not None:
            text_embeds = self.text_encoder(text_inputs.to(self.device))[0]
            uc_embeds = self.uncond_embeds.repeat(text_embeds.size(0), 1, 1)
            cond_embeds = torch.cat([text_embeds, uc_embeds])

        if src_imgs is not None:
            latents = self.vae.encode(src_imgs.to(device=self.device, dtype=self.weight_dtype)).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            src_latents = latents
        else: # generate source images
            if init_latents is not None:
                latents = init_latents.to(device=self.device, dtype=self.weight_dtype)
            else:
                latents = torch.randn(text_embeds.size(0), 4, 64, 64, generator=generator)
            latents = latents.to(device=self.device, dtype=self.weight_dtype) * self.scheduler.init_noise_sigma

            set_attn_processors(self.unet, self.orig_attn_procs)
            self.scheduler.set_timesteps(src_inference_steps)
            self.scheduler.register_to_config(prediction_type='epsilon')
            for t in self.scheduler.timesteps:
                latent_model_input = latents.repeat(2, 1, 1, 1)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=cond_embeds).sample
                noise, noise_uncond = noise_pred.chunk(2)
                noise = noise_uncond + src_guidance_scale * (noise - noise_uncond)
                latents = self.scheduler.step(noise, t, latents).prev_sample
            src_latents = latents

        # Generate target images
        set_attn_processors(self.unet, self.lora_attn_procs)
        self.scheduler.set_timesteps(trg_inference_steps)
        self.scheduler.register_to_config(prediction_type=self.noise_scheduler.config.prediction_type)

        if self.onepass:
            uc_embeds = self.uncond_embeds.repeat(src_latents.size(0), 1, 1)
            cond_embeds = uc_embeds if self.disable_prompts else text_embeds
            trg_latents = self.unet(src_latents, 1, encoder_hidden_states=cond_embeds).sample
            return InferStepOutput(
                    src_latents=src_latents,
                    trg_latents=trg_latents,
            )

        mid_latents = [] if return_mid_latents else None
        for t in self.scheduler.timesteps:
            if self.disable_prompts:
                uc_embeds = self.uncond_embeds.repeat(latents.size(0), 1, 1)
                latent_model_input = latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)
                noise = self.unet(latent_model_input, t, encoder_hidden_states=uc_embeds).sample
            else:
                latent_model_input = latents.repeat(2, 1, 1, 1)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=cond_embeds).sample
                noise, noise_uncond = noise_pred.chunk(2)
                noise = noise_uncond + trg_guidance_scale * (noise - noise_uncond)

            if self.use_oracle_ddim:
                scheduler_output = self.scheduler.step(noise, t, latents, src_latents)
            else:
                scheduler_output = self.scheduler.step(noise, t, latents)

            latents = scheduler_output.prev_sample
            if return_mid_latents:
                mid_latents.append(scheduler_output.prev_sample)

        return InferStepOutput(
                src_latents=src_latents, 
                trg_latents=latents, 
                mid_latents=mid_latents
        )

    def train_batch(self, src, trg, text_ids, 
                    snr_trunc=None, snr_gamma=None):
        # Convert images to latent space
        src = self.vae.encode(src.to(dtype=self.weight_dtype)).latent_dist.sample()
        src = src * self.vae.config.scaling_factor

        trg = self.vae.encode(trg.to(dtype=self.weight_dtype)).latent_dist.sample()
        trg = trg * self.vae.config.scaling_factor

        bsz = src.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=src.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(trg, src, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(text_ids)[0]

        # Predict the noise residual and compute loss
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == 'epsilon':
            target = src
        elif self.noise_scheduler.config.prediction_type == 'sample':
            target = trg
        elif self.noise_scheduler.config.prediction_type == 'v_prediction':
            target = self.noise_scheduler.get_velocity(trg, src, timesteps)
        else:
            raise ValueError(f'Unknown prediction type {self.noise_scheduler.config.prediction_type}')

        if snr_trunc is None and snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction='mean')
        else:
            snr = compute_snr(timesteps)
            mse_loss_weights = snr
            if snr_trunc is not None:
                mse_loss_weights = (
                    torch.stack([mse_loss_weights, snr_trunc * torch.ones_like(timesteps)], dim=1).max(dim=1)[0]
                )
            if snr_gamma is not None:
                mse_loss_weights = (
                    torch.stack([mse_loss_weights, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                )
            if self.noise_scheduler.config.prediction_type == 'epsilon':
                mse_loss_weights = mse_loss_weights / snr
            elif self.noise_scheduler.config.prediction_type == 'v_predction':
                mse_loss_weights = mse_loss_weights / (snr + 1)
            # We first calculate the original loss. Then we mean over the non-batch dimensions and
            # rebalance the sample-wise losses with their respective loss weights.
            # Finally, we take the mean of the rebalanced loss.
            loss = F.mse_loss(model_pred.float(), target.float(), reduction='none')
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        return loss

    def decode_latents(self, latents, target_mode='RGB', target_scale=1, return_preview=True):
        latents = latents / self.vae.config.scaling_factor
        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        if target_mode == 'RGB':
            imgs = (imgs / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1) * 255
            imgs = imgs.round().detach().cpu().numpy().astype('uint8')
            imgs = [Image.fromarray(img) for img in imgs]
        elif target_mode == 'F':
            imgs = (imgs.float().mean(dim=1) / 2 + 0.5).clamp(min=0)
            if target_scale > 0:
                arrs = imgs * target_scale
            else:
                m = imgs.amin(dim=(1,2), keepdim=True)
                M = imgs.amax(dim=(1,2), keepdim=True)
                arrs = (imgs - m) / (M - m)
            if return_preview:
                imgs = arrs.div(target_scale).clamp(0, 1) if target_scale > 0 else arrs
                imgs = imgs.mul(65535).round().cpu().numpy().astype('uint16')
                imgs = [(Image.fromarray(img), arr.cpu().numpy()) for img, arr in zip(imgs, arrs)]
            else:
                return arrs.cpu().numpy()
        else:
            raise NotImplementedError()
        return imgs

    @torch.no_grad()
    def caption(self, img):
        if not hasattr(self, 'blip_processor'):
            from transformers import Blip2Processor
            self.blip_processor = Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b')
        if not hasattr(self, 'blip_model'):
            from transformers import Blip2ForConditionalGeneration
            self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
                    'Salesforce/blip2-opt-2.7b', 
                    load_in_8bit=True, 
                    device_map={'': 0}, 
                    torch_dtype=torch.float16
            )
        inputs = self.blip_processor(images=img, return_tensors='pt').to(self.device, torch.float16)
        generated_ids = self.blip_model.generate(**inputs)
        generated_text = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(generated_text)

        input_ids = self.tokenizer([generated_text], padding='max_length',
            max_length=self.tokenizer.model_max_length, return_tensors='pt').input_ids
        return input_ids

    @torch.no_grad()
    def __call__(self, source_img, inference_steps, target_mode, guidance_scale=2, resize=True):
        """Inference function for 1 sample
        Params:
            source_img: PIL.Image with RGB mode
            inference_step: int
            target_mode: str = `RGB` or `F`
        """
        if not self.has_prepared_infer:
            self.prepare_infer()
        if not self.disable_prompts:
            text_inputs = self.caption(source_img)
            text_embeds = self.text_encoder(text_inputs.to(self.device))[0]
            uc_embeds = self.uncond_embeds.repeat(text_embeds.size(0), 1, 1)
            cond_embeds = torch.cat([text_embeds, uc_embeds])

        src = ttf.to_tensor(source_img)
        src = ttf.normalize(src, mean=0.5, std=0.5)
        if resize: # match smaller edge
            src = ttf.resize(src, 512, interpolation=ttf.InterpolationMode.BICUBIC)
        src = src[None]

        latents = self.vae.encode(src.to(device=self.device, dtype=self.weight_dtype)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        src_latents = latents

        # Generate target images
        set_attn_processors(self.unet, self.lora_attn_procs)
        self.scheduler.set_timesteps(inference_steps)
        self.scheduler.register_to_config(prediction_type=self.noise_scheduler.config.prediction_type)

        for t in self.scheduler.timesteps:
            if self.disable_prompts:
                uc_embeds = self.uncond_embeds.repeat(latents.size(0), 1, 1)
                latent_model_input = latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)
                noise = self.unet(latent_model_input, t, encoder_hidden_states=uc_embeds).sample
            else:
                latent_model_input = latents.repeat(2, 1, 1, 1)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=cond_embeds).sample
                noise, noise_uncond = noise_pred.chunk(2)
                noise = noise_uncond + guidance_scale * (noise - noise_uncond)

            if self.use_oracle_ddim:
                scheduler_output = self.scheduler.step(noise, t, latents, src_latents)
            else:
                scheduler_output = self.scheduler.step(noise, t, latents)
            latents = scheduler_output.prev_sample

        img = self.decode_latents(latents, target_mode, return_preview=False)[0]
        if resize: # restore the original size
            W, H = source_img.size
            if target_mode == 'RGB':
                img = img.resize((W, H), resample=Image.Resampling.BICUBIC)
            elif target_mode == 'F':
                img = ttf.resize(torch.tensor(img)[None], (H, W), interpolation=ttf.InterpolationMode.BICUBIC)[0]
                img = img.numpy()
        return img


@dataclass
class InferStepOutput:
    src_latents: torch.FloatTensor = None
    trg_latents: torch.FloatTensor = None
    mid_latents: torch.FloatTensor = None

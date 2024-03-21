from argparse import ArgumentParser
from itertools import islice
import os

from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler
from tqdm.auto import tqdm
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
import torch


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--prompt')
    parser.add_argument('--from-file')
    parser.add_argument('--output', required=True)
    parser.add_argument('--pretrained', default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--num-inference-steps', type=int, default=50)
    parser.add_argument('--num-iter', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--guidance-scale', type=int, default=7.5)
    parser.add_argument('--plms', action='store_true')
    parser.add_argument('--seed', type=int, default=2147483647)
    parser.add_argument('--read-xt')
    parser.add_argument('--save-xt', action='store_true')
    args = parser.parse_args()

    if args.prompt is None and args.from_file is None:
        raise ValueError('No prompt.')
    if args.prompt:
        args.prompt = [args.batch_size * [args.prompt]]
        args.num_iter = (args.num_iter + args.batch_size - 1) // args.batch_size
    else:
        args.prompt = list(chunk(open(args.from_file).read().splitlines(), args.batch_size))
        if args.num_iter > 1:
            print("WARNING: --num-iter > 1. Are you sure?")

    if args.save_xt and args.read_xt:
        raise ValueError('Read and save at the same time?')
    if args.read_xt and not os.path.isdir(args.read_xt):
        raise ValueError(f'{args.read_xt} is not a dir.')

    os.makedirs(args.output, exist_ok=True)

    return args


@torch.no_grad()
def main():
    args = parse_args()

    vae = AutoencoderKL.from_pretrained(args.pretrained, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained, subfolder="unet")
    if args.plms:
        print('Use PNDM sampler')
        scheduler = PNDMScheduler.from_pretrained(args.pretrained, subfolder="scheduler")
    else:
        print('Use DDIM sampler')
        scheduler = DDIMScheduler.from_pretrained(args.pretrained, subfolder="scheduler")

    device = "cuda"
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    unet.enable_xformers_memory_efficient_attention()

    generator = torch.manual_seed(args.seed)

    img_cnt = 0
    xt_cnt = 0

    for i in range(args.num_iter):
        for j, prompts in enumerate(args.prompt):

            # Encode text
            text_input = tokenizer(
                prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
            )
            bsz, seq = text_input.input_ids.shape
            uncond_input = tokenizer([""] * bsz, padding="max_length", max_length=seq, return_tensors="pt")
            with torch.autocast(device):
                text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
                uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

            # Prepare latents
            if args.read_xt:
                latents = []
                for _ in range(len(prompts)):
                    xt_path = os.path.join(args.read_xt, f'{xt_cnt:05}.npz')
                    xt = torch.tensor(np.load(xt_path)['x'], dtype=torch.float)
                    latents.append(xt.unsqueeze(0))
                    xt_cnt += 1
                latents = torch.cat(latents)
            else:
                latents = torch.randn(len(prompts), unet.config.in_channels, args.height//8, args.width//8, generator=generator)
                if args.save_xt:
                    for b in range(len(prompts)):
                        np.savez_compressed(os.path.join(args.output, f'{xt_cnt:05}.npz'), x=latents[b].cpu().numpy())
                        xt_cnt += 1

            latents = latents.to(device)
            latents = latents * scheduler.init_noise_sigma

            # Denoise
            scheduler.set_timesteps(args.num_inference_steps)
            with torch.autocast(device):
                for t in tqdm(scheduler.timesteps, desc=f'Iter {i+1}/{args.num_iter} Prompt {j+1}/{len(args.prompt)}'):
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

                    with torch.no_grad():
                        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    latents = scheduler.step(noise_pred, t, latents).prev_sample

            # Decode images
            latents = 1 / 0.18215 * latents
            with torch.autocast(device):
                images = vae.decode(latents).sample
            images = (images / 2 + 0.5).clamp(0, 1)
            images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (images * 255).round().astype("uint8")

            for img in images:
                Image.fromarray(img).save(os.path.join(args.output, f'{img_cnt:05}.png'))
                img_cnt += 1


if __name__ == "__main__":
    main()

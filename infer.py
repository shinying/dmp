import os

from argparse import ArgumentParser
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import torch

from data import InferDataset
from pipeline import Pipeline
from utils import seed_everything


@torch.no_grad()
def main():
    parser = ArgumentParser()
    parser.add_argument('--prompts', default='')
    parser.add_argument('--lora-ckpt', required=True)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--output', default='output')
    parser.add_argument('--latents')
    parser.add_argument('--src')
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--save-mid', action='store_true')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--target-mode', default='RGB', choices=['RGB', 'F'])
    parser.add_argument('--target-scale', default=1, type=float)
    parser.add_argument('--target-pred-type', default='v_prediction', choices=['epsilon', 'sample', 'v_prediction'])
    parser.add_argument('--self-attn-only', action='store_true')
    parser.add_argument('--disable-prompts', action='store_true')
    parser.add_argument('--use-oracle-ddim', action='store_true')
    parser.add_argument('--onepass', action='store_true')
    parser.add_argument('--seed', type=int, default=666666)
    parser.add_argument('--num-workers', type=int, default=1)
    args = parser.parse_args()

    if not args.src and not args.prompts:
        raise ValueError('at least provide --prompts or --src.')
    if not args.disable_prompts and args.prompts == '':
        raise ValueError('either provide --prompts or set --disable-prompts.')

    cfg = OmegaConf.load(args.config)
    if not args.src:
        os.makedirs(os.path.join(args.output, 'src'), exist_ok=True)
    os.makedirs(os.path.join(args.output), exist_ok=True)
    seed_everything(args.seed)
    device = 'cuda'

    pipeline = Pipeline(
            cfg.pretrained,
            self_attn_only=args.self_attn_only,
            disable_prompts=args.disable_prompts,
            onepass=args.onepass,
            prediction_type=args.target_pred_type,
            use_oracle_ddim=args.use_oracle_ddim,
            lora_ckpt=args.lora_ckpt,
            enable_xformers=True,
            device=device,
            mixed_precision='fp16',
    )
    generator = torch.Generator().manual_seed(args.seed)
    dataset = InferDataset(args.prompts, pipeline.tokenizer, args.latents, args.src, args.num_samples, generator)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # Denoise
    for i, batch in enumerate(tqdm(dataloader, ncols=100)):
        # Check if some images have been processed
        s = batch['key'][0]
        e = batch['key'][-1]
        if (os.path.isfile(os.path.join(args.output, 'trg', f'{s}.png')) and
            os.path.isfile(os.path.join(args.output, 'trg', f'{e}.png'))):
            continue
        
        if args.save_mid:
            for key in batch['key']:
                os.makedirs(f'mid/{args.output}/{key}', exist_ok=True)
        
        output = pipeline.infer_batch(
                batch['text_ids'],
                batch.get('latents', None),
                batch.get('src_img', None),
                cfg.num_inference_steps,
                cfg.target_inference_steps,
                cfg.guidance_scale,
                cfg.target_guidance_scale,
                return_mid_latents=args.save_mid,
        )

        if args.save_mid:
            for ti, t in enumerate(pipeline.scheduler.timesteps):
                mid = pipeline.decode_latents(output.mid_latents[ti])
                for img, key in zip(mid, batch['key']):
                    img.save(f'mid/{args.output}/{key}/{t:03}.png')

        if not args.src:
            imgs = pipeline.decode_latents(output.src_latents)
            for img, key in zip(imgs, batch['key']):
                img.save(os.path.join(args.output, 'src', f'{key}.png'))

        trgs = pipeline.decode_latents(output.trg_latents, args.target_mode, args.target_scale)
        for trg, key in zip(trgs, batch['key']):
            if args.target_mode == 'RGB':
                trg.save(os.path.join(args.output, f'{key}.png'))
            elif args.target_mode == 'F':
                visual, arr = trg
                visual.save(os.path.join(args.output, f'{key}.png'))
                np.savez_compressed(os.path.join(args.output, f'{key}.npz'), x=arr)


if __name__ == '__main__':
    main()

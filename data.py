import glob
import io
import json
import os
import random

from PIL import Image
from torchvision import transforms
import lmdb
import numpy as np
import torch
import torchvision.transforms.functional as F
import albumentations as A
import cv2


def tokenize_caption(tokenizer, caption, is_train=True):
    if isinstance(caption, (list, tuple)):
        caption = random.choice(caption) if is_train else caption[0]
    inputs = tokenizer(
        caption,
        max_length=tokenizer.model_max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return inputs.input_ids[0]


class TrainDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_root,
        target_db,
        tokenizer,
        target_mode='RGB',
        target_scale=None,
        target_extra_key=None,
        random_flip=False,
        more_augment=False,
        disable_prompts=False,
    ):
        self.data_root = data_root
        self.env = lmdb.open(target_db, map_size=1024**3*100, readonly=True)
        self.txn = self.env.begin()
        self.tokenizer = tokenizer
        self.target_mode = target_mode
        self.target_scale = target_scale
        self.target_extra_key = target_extra_key
        self.random_flip = random_flip
        self.disable_prompts = disable_prompts 

        self.metadata = [json.loads(l) for l in open(os.path.join(self.data_root, 'metadata.jsonl')).readlines()]
        self._length = len(self.metadata)

        self.do_augment = random_flip or more_augment
        tf = [A.HorizontalFlip()] if random_flip else []
        if more_augment:
            tf.extend([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.9),
                A.ImageCompression(quality_lower=70),
            ])
        self.aug_transform = A.Compose(tf)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        entry = self.metadata[i]
        example = {}

        image = Image.open(os.path.join(self.data_root, entry['file_name']))
        if not image.mode == 'RGB':
            image = image.convert('RGB')

        trg_key = entry['file_name'].split('.')[0]
        if self.target_extra_key:
            trg_key += f'-{self.target_extra_key}'
        if self.target_mode == 'RGB':
            target = Image.open(io.BytesIO(self.txn.get(trg_key.encode())))
            if not target.mode == 'RGB':
                target = target.convert('RGB')
        elif self.target_mode == 'F':
            target = np.load(io.BytesIO(self.txn.get(trg_key.encode())))['x']
            if self.target_scale is not None:
                if self.target_scale == -1:
                    target = (target - target.min()) / (target.max() - target.min())
                else:
                    target = target / self.target_scale
        else:
            raise NotImplementedError

        if self.do_augment:
            x = self.aug_transform(image=np.array(image), mask=np.array(target))
            image, target = x['image'], x['mask']

        example['rgb'] = self.transform(image)
        example['pixel_values'] = self.transform(target)
        if self.target_mode == 'F':
            example['pixel_values'] = example['pixel_values'].repeat(3, 1, 1)

        if self.disable_prompts:
            example['input_ids'] = tokenize_caption(self.tokenizer, '')
        else:
            example['input_ids'] = tokenize_caption(self.tokenizer, entry['text'])
        return example


class InferDataset(torch.utils.data.Dataset):

    def __init__(self, prompts, tokenizer, latents=None, src_imgs=None, num_samples=None, generator=None):
        if isinstance(prompts, str):
            if os.path.isfile(prompts):
                print('Reading prompts from', prompts)
                self.prompts = open(prompts).read().splitlines()
            else:
                self.prompts = [prompts]
        elif hasattr(prompts, '__iter__'):
            self.prompts = prompts
        else:
            raise NotImplementedError('unsupported prompts', type(prompts))
        self.num_prompts = len(self.prompts)

        self.tokenizer = tokenizer

        self.latents = latents
        self.src_imgs = src_imgs
        self.generator = generator

        self.num_latents = 0
        if src_imgs and os.path.isdir(src_imgs):
            print('Using source images from', src_imgs)
            if os.path.isdir(src_imgs):
                self.src_imgs = glob.glob(os.path.join(src_imgs, '*.png')) + \
                        glob.glob(os.path.join(src_imgs, '*.jpg'))
                self.src_imgs.sort()
            elif os.path.isfile(src_imgs):
                self.src_imgs = [src_imgs]
            num_samples = num_samples or len(self.src_imgs)
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ])
        else:
            if latents is None:
                if generator is None:
                    print('WARNING: no generator for latents')
            else:
                if os.path.isdir(latents):
                    print('Using latents from', latents)
                    self.latents = glob.glob(os.path.join(latents, '*.npz'))
                    self.latents.sort()
                elif os.path.isfile(latents):
                    print('Using latents from', latents)
                    self.latents = [latents]
                self.num_latents = len(self.latents)

        self.num_samples = num_samples or max(self.num_prompts, self.num_latents)
        print(f'Genrating {self.num_samples} images')

    def __len__(self):
        return self.num_samples

    def __getitem__(self, i):
        sample = {}
        sample['key'] = f'{i:05}'
        prompt = self.prompts[i%self.num_prompts]
        sample['text_ids'] = self.tokenizer(
            prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors='pt').input_ids[0]

        if self.src_imgs is not None:
            path = self.src_imgs[i]
            src_img = Image.open(path).convert('RGB')
            sample['src_img'] = self.transform(src_img)
            sample['key'] = os.path.basename(path).split('.')[0]
        elif self.latents is None:
            sample['latents'] = torch.randn(4, 64, 64, generator=self.generator)
        else:
            path = self.latents[i%self.num_latents]
            latents = torch.tensor(np.load(path)['x'])
            if latents.size() == 4:
                latents.squeeze(0)
            sample['latents'] = latents
            sample['key'] = os.path.basename(path).split('.')[0]

        return sample

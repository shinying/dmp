import argparse
import glob
import io
import multiprocessing as mp
import os

from PIL import Image
from tqdm import tqdm
import lmdb
import numpy as np
import torch
import torchvision.transforms.functional as tf

from models import DPTRegressionModel
from models.models import TrainableModel, WrapperModel, DataParallelModel
from models.utils import *


def build_model():
    ckpt_path = './tools/models/rgb2normal_omni_dpt_2d3daug.pth'

    model = WrapperModel(DataParallelModel(
        DPTRegressionModel(num_channels=3, backbone='vitb_rn50_384', non_negative=False)))
    model_state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(model_state_dict["('rgb', '"+'normal'+"')"])
    model.eval()
    model = model.to("cuda")
    return model

def process(model, imgs):
    output = model(imgs.to("cuda")).clamp(min=0, max=1).cpu().numpy().transpose(0,2,3,1)
    output = (output * 255).astype(np.uint8)
    return output

def encode(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='png')
    return buf.getvalue()


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, img_dir, index=None, ngpu=None):
        self.imgs = glob.glob(os.path.join(img_dir, '*.png'))
        self.imgs.sort()
        if index is not None and ngpu is not None:
            if index < 0 or index > ngpu - 1:
                raise ValueError(f"Invalid index: {index}")
            chunk_size = len(self.imgs) // ngpu
            s = chunk_size * index
            e = chunk_size * (index + 1) if index < ngpu - 1 else len(self.imgs)
            print(f"Processing imgs from {s} to {e-1}")
            self.imgs = self.imgs[s:e]
        else:
            print(f"Processing all {len(self.imgs)} imgs")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        img = (tf.to_tensor(img) - 0.5) / 0.5
        return img, os.path.basename(img_path).split('.')[0]


class DBDataset(torch.utils.data.Dataset):

    def __init__(self, db, key_list=None, index=None, ngpu=None):
        self.env = lmdb.open(db, map_size=1024**3*100, readonly=True)
        self.txn = self.env.begin()

        if key_list is not None:
            self.keys = open(key_list).read().splitlines()
        else:
            self.keys = [key.decode() for key in self.txn.cursor().iternext(values=False)]
            self.keys.sort()

        if index is not None and ngpu is not None:
            if index < 0 or index > ngpu - 1:
                raise ValueError(f"Invalid index: {index}")
            chunk_size = len(self.keys) // ngpu
            s = chunk_size * index
            e = chunk_size * (index + 1) if index < ngpu - 1 else len(self.keys)
            print(f"Processing imgs from {s} to {e-1}")
            self.keys = self.keys[s:e]
        else:
            print(f"Processing all {len(self.keys)} imgs")

    def __del__(self):
        if hasattr(self, "env"):
            self.env.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        img = Image.open(io.BytesIO(self.txn.get(key.encode())))
        img = (tf.to_tensor(img) - 0.5) / 0.5
        return img, key


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-k", "--keys")
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--index", type=int)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--save-files", action="store_true")
    args = parser.parse_args()

    if os.path.isfile(os.path.join(args.input, "data.mdb")):
        dataset = DBDataset(args.input, args.keys, args.index, args.world_size)
    else:
        assert args.keys is None
        dataset = ImageDataset(args.input, args.index, args.world_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, num_workers=8)

    model = build_model()
    pool = mp.Pool(8)

    with torch.no_grad():
        if args.save_files:
            os.makedirs(args.output, exist_ok=True)
            for imgs, keys in tqdm(dataloader):
                output = process(model, imgs)
                output = pool.map(encode, output)
                for out, key in zip(output, keys):
                    open(os.path.join(args.output, key+".png"), 'wb').write(out)
        else:
            env = lmdb.open(args.output, map_size=1024**3*100)
            for imgs, keys in tqdm(dataloader):
                output = process(model, imgs)
                txn = env.begin(write=True)
                output = pool.map(encode, output)
                for out, key in zip(output, keys):
                    txn.put(key.encode(), out)
                txn.commit()
            env.close()

    pool.close()
    pool.join()

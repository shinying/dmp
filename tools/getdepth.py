import argparse
import glob
import io
import os

from PIL import Image
from tqdm import tqdm
import lmdb
import numpy as np
import torch
import torchvision.transforms.functional as tf


def build_model():
    torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)
    repo = "isl-org/ZoeDepth"
    model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)
    zoe = model_zoe_n.to("cuda")
    zoe.eval()
    return zoe

def process(model, imgs):
    return model.infer(imgs.to("cuda")).squeeze(dim=1).cpu().numpy()

def encode(output):
    output = output.astype(np.float16)
    buf = io.BytesIO()
    np.savez_compressed(buf, x=output)
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
        return tf.to_tensor(img), os.path.basename(img_path).split('.')[0]


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
        return tf.to_tensor(img), key


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-k", "--keys")
    parser.add_argument("--bs", type=int, default=8)
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

    with torch.no_grad():
        if args.save_files:
            os.makedirs(args.output, exist_ok=True)
            for imgs, keys in tqdm(dataloader):
                output = process(model, imgs)
                for out, key in zip(output, keys):
                    buf = encode(out)
                    open(os.path.join(args.output, key+'.npz'), 'wb').write(buf)
        else:
            env = lmdb.open(args.output, map_size=1024**3*100)
            for imgs, keys in tqdm(dataloader):
                output = process(model, imgs)
                txn = env.begin(write=True)
                for out, key in zip(output, keys):
                    buf = encode(out)
                    txn.put(key.encode(), buf)
                txn.commit()
            env.close()

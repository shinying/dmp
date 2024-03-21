import argparse
import glob
import io
import multiprocessing as mp
import os

from PIL import Image
from tqdm import tqdm
import cv2
import lmdb
import numpy as np
import torch
import torchvision.transforms.functional as F

from Network import DecScaleClampedIllumEdgeGuidedNetworkBatchNorm
from Utils import mor_utils

torch.backends.cudnn.benchmark = True


def build_model():
    modelSaveLoc = 'model/real_world_model.t7'
    device = 'cuda'
    done = u'\u2713'
    print('[I] STATUS: Create utils instances...', end='')
    support = mor_utils(device)
    print(done)

    print('[I] STATUS: Load Network and transfer to device...', end='')
    net = DecScaleClampedIllumEdgeGuidedNetworkBatchNorm()
    net, _, _ = support.loadModels(net, modelSaveLoc)
    net.to(device)
    net.eval()
    print(done)
    return net

def process(model, imgs):
    output = model(imgs.to('cuda'))
    for key in ('reflectance', 'shading'):
        x = output[key].permute(0, 2, 3, 1) 
        amin, amax = x.amin(dim=(1,2,3), keepdim=True), x.amax(dim=(1,2,3), keepdim=True)
        x = (x - amin) / (amax - amin)
        x = x.mul(255).round().cpu()
        if x.shape[3] == 1:
            x = x.squeeze(3)
        output[key] = x.numpy().astype(np.uint8)
    return output

def encode(arr, **kwargs):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, **kwargs)
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
        img = np.array(img)
        return F.to_tensor(img), os.path.basename(img_path).split('.')[0]


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
        img = np.array(img)
        if img.shape[2] > 3:
            img = img[:,:,:3]
        return F.to_tensor(img), key


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-k", "--keys")
    parser.add_argument("--index", type=int)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--save-files", action="store_true")
    args = parser.parse_args()

    if os.path.isfile(os.path.join(args.input, "data.mdb")):
        dataset = DBDataset(args.input, args.keys, args.index, args.world_size)
    else:
        assert args.keys is None
        dataset = ImageDataset(args.input, args.index, args.world_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=8)

    model = build_model()

    with torch.no_grad():
        if args.save_files:
            os.makedirs(os.path.join(args.output, 'ref'), exist_ok=True)
            os.makedirs(os.path.join(args.output, 'sha'), exist_ok=True)
            for imgs, keys in tqdm(dataloader):
                output = process(model, imgs)
                for out, key in zip(output['reflectance'], keys):
                    out = encode(out, format='png')
                    open(os.path.join(args.output, 'ref', key+".png"), 'wb').write(out)
                for out, key in zip(output['shading'], keys):
                    out = encode(out, format='png')
                    open(os.path.join(args.output, 'sha', key+".png"), 'wb').write(out)
        else:
            env = lmdb.open(args.output, map_size=1024**3*100)
            for imgs, keys in tqdm(dataloader):
                output = process(model, imgs)
                txn = env.begin(write=True)

                for out, key in zip(output['reflectance'], keys):
                    out = encode(out, format='png')
                    txn.put(f"{key}-r".encode(), out)

                for out, key in zip(output['shading'], keys):
                    out = encode(out, format='png')
                    txn.put(f"{key}-s".encode(), out)

                txn.commit()
            env.close()

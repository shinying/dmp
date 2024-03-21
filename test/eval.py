from argparse import ArgumentParser
from pathlib import Path

from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def l1(pred, true):
    return (pred - true).abs().mean()

def rel(pred, true):
    return ((pred - true).abs() / true).mean()

def mse(pred, true):
    return ((pred - true) ** 2).mean()

def rmse(pred, true):
    return mse(pred, true)**0.5

def delta(pred, true, n=1, eps=1e-12):
    pred = pred + eps
    true = true + eps
    return (torch.maximum(pred/true, true/pred) < 1.25**n).float().mean()

def angular(pred, true):
    sim = F.cosine_similarity(pred, true, dim=2).clamp(0, 1)
    return torch.arccos(sim).mean()


def evaluate(pred, truth_root, metrics, absolute=False, normalize=False):
    if absolute:
        x_pred = np.load(pred)['x']
    else:
        x_pred = cv2.imread(str(pred))

    if x_pred.dtype == np.uint8:
        x_pred = x_pred / 255
    elif x_pred.dtype == np.uint16:
        x_pred = x_pred / 65535

    if absolute:
        x_true = np.load(truth_root/pred.name)['x']
    else:
        x_true = cv2.imread(str(truth_root/pred.name))
        if x_true.dtype == np.uint8:
            x_true = x_true / 255

    if x_pred.shape[0] != x_true.shape[0] or x_pred.shape[1] != x_true.shape[1]:
        x_true = cv2.resize(x_true.astype('float32'), x_pred.shape[:2], interpolation=cv2.INTER_AREA)
    if normalize:
        x_pred = (x_pred - x_pred.min()) / (x_pred.max() - x_pred.min())
        x_true = (x_true - x_true.min()) / (x_true.max() - x_true.min())

    device = 'cuda'
    x_pred = torch.tensor(x_pred, device=device, dtype=torch.float)
    x_true = torch.tensor(x_true, device=device, dtype=torch.float)
    return torch.stack([metric(x_pred, x_true) for metric in metrics])


def main():
    parser = ArgumentParser()
    parser.add_argument('pred', type=Path)
    parser.add_argument('truth', type=Path)
    parser.add_argument('--metrics', nargs='+', required=True)
    parser.add_argument('--ext', default='png')
    parser.add_argument('--abs', action='store_true', help='predictions are absolute values')
    parser.add_argument('--norm', action='store_true', help='normalize')
    args = parser.parse_args()

    preds = list(args.pred.glob('*.'+args.ext))
    truths = list(args.truth.glob('*.'+args.ext))
    if len(preds) == 0:
        raise ValueError(f'No {args.ext} files found in {args.pred}')
    if len(preds) > len(truths):
        raise ValueError('Not enough ground truth data')

    supported_metrics = ['l1', 'rel', 'mse', 'rmse', 'delta', 'angular']
    metric_funcs = []
    for metric in args.metrics:
        if metric not in supported_metrics:
            raise NotImplementedError(metric)
        metric_funcs.append(eval(metric))

    result = torch.stack([evaluate(pred, args.truth, metric_funcs, args.abs, args.norm) for pred in tqdm(preds, ncols=100)]).mean(dim=0)

    for name, value in zip(args.metrics, result):
        print(f'{name} loss: {value.item():.6f}')

main()

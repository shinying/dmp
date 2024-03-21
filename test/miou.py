from argparse import ArgumentParser
import glob
import os

# from mmseg.core.evaluation import mean_iou
from metrics import mean_iou


CLASSES = (
        'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ',
        'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
        'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
        'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug',
        'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
        'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
        'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
        'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
        'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
        'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
        'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
        'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
        'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
        'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
        'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
        'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
        'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
        'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
        'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
        'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
        'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
        'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
        'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
        'clock', 'flag')


def main():
    parser = ArgumentParser()
    parser.add_argument('pred')
    parser.add_argument('truth')
    parser.add_argument('-N', '--num-classes', type=int, default=len(CLASSES))
    parser.add_argument('--ignore', type=int, default=255)
    parser.add_argument('--reduce-zero', action='store_true')
    args = parser.parse_args()

    pred = glob.glob(os.path.join(args.pred, '*.npy'))
    pred.sort()
    true = glob.glob(os.path.join(args.truth, '*.png'))
    true.sort()

    result = mean_iou(pred, true, num_classes=args.num_classes, ignore_index=args.ignore, reduce_zero_label=args.reduce_zero)
    print('mAcc:', result['aAcc'])
    for i in range(args.num_classes):
        print(f"{CLASSES[i]:<16}", f"acc:{result['Acc'][i]:.4f}\t", f"iou:{result['IoU'][i]:.4f}")


main()

from argparse import ArgumentParser
import json
import os

EXTENSIONS = ('png', 'jpg')

parser = ArgumentParser()
parser.add_argument('-i', '--imgs', required=True)
parser.add_argument('-c', '--captions')
args = parser.parse_args()

imgs = [f for f in os.listdir(args.imgs) if any(f.endswith(ext) for ext in EXTENSIONS)]
imgs.sort()
captions = open(args.captions).read().splitlines() if args.captions else [None] * len(imgs)

if len(imgs) != len(captions):
    print('[WARNING] the size of images and captions does not match')

with open(os.path.join(args.imgs, 'metadata.jsonl'), 'w') as meta:
    for fn, caption in zip(imgs, captions):
        entry = {'file_name': fn}
        if args.captions:
            entry['text'] = caption
        print(json.dumps(entry), file=meta)

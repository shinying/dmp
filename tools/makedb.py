import argparse
import os

from tqdm import tqdm
import lmdb


parser = argparse.ArgumentParser()
parser.add_argument('dir')
parser.add_argument('db')
args = parser.parse_args()

if not os.path.isdir(args.dir):
    raise ValueError(f"expects {args.dir} a directory.")

files = os.listdir(args.dir)
print(f"Found {len(files)} files.")

with lmdb.open(args.db, map_size=1024**3*100) as env:
    with env.begin(write=True) as txn:
        for f in tqdm(files):
            key = f.split('.')[0]
            txn.put(key.encode(), open(os.path.join(args.dir, f), 'rb').read())
        txn.commit()


import os
import shutil

SRC_DIR = os.path.join('src', 'data')
DST_DIR = 'data'

os.makedirs(DST_DIR, exist_ok=True)

for fname in ['S3.csv', 'sink.csv', 'LAKE.csv']:
    src = os.path.join(SRC_DIR, fname)
    dst = os.path.join(DST_DIR, fname)
    if os.path.exists(src):
        shutil.copyfile(src, dst)
        print(f'Copied {src} -> {dst}')
    else:
        print(f'[WARN] Missing {src}')


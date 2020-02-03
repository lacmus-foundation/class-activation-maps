from multiprocessing import Pool
import json
import random

import cv2

from sklearn.model_selection import train_test_split

from tqdm import tqdm

import numpy as np

from utils import load_dataset, get_mask
from config import DATASET_DIR, CROP_SIZE


DATASET = load_dataset(DATASET_DIR)
TILES_DIR = DATASET_DIR.with_suffix('.tiles')

person_dir = TILES_DIR / 'Person'
person_dir.mkdir(parents=True, exist_ok=True)

noperson_dir = TILES_DIR / 'NoPerson'
noperson_dir.mkdir(parents=True, exist_ok=True)


def bbox_size(bbox):
    x0, y0, x1, y1 = bbox
    return x1 - x0, y1 - y0


def crop_center(image, bbox, size):
    x0, y0, x1, y1 = bbox
    x, y = (x0 + x1) // 2, (y0 + y1) // 2
    r = size // 2

    # skip boxes near the image borders
    if any([
        x - r < 0,
        y - r < 0,
        x + r > image.shape[1],
        y + r > image.shape[0],
    ]):
        return None, None

    dx = (x1 - x0) // 2
    dy = (y1 - y0) // 2
    new_bbox = (r - dx, r - dy, r + dx, r + dy)
    return image[y - r:y + r, x - r:x + r], new_bbox


def process_file_person(args):
    filename, bboxes, out_dir, size = args
    ret = []
    image = cv2.imread(str(filename), cv2.IMREAD_UNCHANGED)
    for n, bbox in enumerate(bboxes):
        cropped, new_bbox = crop_center(image, bbox, size)
        if cropped is None:
            continue
        fname = f'{filename.stem}_{n}.jpg'
        cv2.imwrite(str(out_dir / fname), cropped)
        ret.append((fname, new_bbox))
    return ret


def create_crop_person(dataset, out_dir, size):
    results = []
    print(f"Populating {out_dir}")
    pool = Pool()
    try:
        tasks = [(filename, bboxes, out_dir, size) for filename, bboxes in dataset]
        for i in tqdm(pool.imap_unordered(process_file_person, tasks), total=len(dataset)):
            results.extend(i)
    except KeyboardInterrupt:
        pass
    finally:
        pool.terminate()
        pool.join()
        with (out_dir.parent / (out_dir.name + '.json')).open('w') as f:
            json.dump(dict(results), f)


def process_file_noperson(args):
    filename, bboxes, out_dir, num_samples_per_image, size, seed = args
    rnd = random.Random(seed)
    source_image = cv2.imread(str(filename), cv2.IMREAD_UNCHANGED)
    h, w, _ = source_image.shape
    mask = get_mask(w, h, bboxes)
    num_samples_generated = 0
    max_attempts = num_samples_per_image * 100
    for i in range(max_attempts):
        x = rnd.randint(0, w - size - 1)
        y = rnd.randint(0, h - size - 1)
        if not np.any(mask[y:y + size, x:x + size]):
            out_image = source_image[y:y + size, x:x + size]
            # don't create several images near the same area,
            # but use random to have a chance to have an image near the bbox
            x -= x % 200
            y -= y % 200
            fname = f'{filename.stem}_{x}_{y}.jpg'
            path = out_dir / fname
            if not path.exists():
                cv2.imwrite(str(path), out_image)
                num_samples_generated += 1
                if num_samples_generated == num_samples_per_image:
                    return


def create_crop_noperson(dataset, out_dir, num_samples_per_image, size):
    print(f"Populating {out_dir}")
    pool = Pool()
    rnd = random.Random(42)
    try:
        tasks = [i + (out_dir, num_samples_per_image, size, rnd.random()) for i in dataset]
        for i in tqdm(pool.imap_unordered(process_file_noperson, tasks), total=len(tasks)):
            pass
    finally:
        pool.terminate()
        pool.join()


if __name__ == "__main__":

    create_crop_person(DATASET, person_dir, size=CROP_SIZE)
    create_crop_noperson(DATASET, noperson_dir, num_samples_per_image=50, size=CROP_SIZE)

    for i in ['train', 'val']:
        p = TILES_DIR / i
        p.mkdir(exist_ok=True)
        for j in ['Person', 'NoPerson']:
            (p / j).mkdir(exist_ok=True)

    # load files list and split with default 25% going to val
    files = list(TILES_DIR.glob('*/*.jpg'))
    train, val = train_test_split(files)

    # move train images
    for i in train:
        i.rename(TILES_DIR / 'train' / i.name)

    # move test images
    for i in val:
        i.rename(TILES_DIR / 'val' / i.name)

    # `tiles/{Person,NoPerson}` are now empty and unneeded, remove them:
    person_dir.rmdir()
    noperson_dir.rmdir()

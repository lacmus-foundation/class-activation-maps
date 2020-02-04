import os
from pathlib import Path


_default_dir = os.path.join('data/', os.environ['SEASON'])

DATASET_DIR = Path(os.getenv('DATASET_DIR', _default_dir))
TILES_DIR = DATASET_DIR.with_suffix('.tiles')
CROP_SIZE = 512

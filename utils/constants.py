import os.path as osp


DATA_DIR = "data"
MOLWENI = 'MOLWENI'
STAC = 'STAC'
MINECRAFT = 'MINECRAFT'
NUM_RELATIONS = 12

DATASETS = {
    MOLWENI: osp.join(DATA_DIR, MOLWENI),
    STAC: osp.join(DATA_DIR, STAC),
    MINECRAFT: osp.join(DATA_DIR, MINECRAFT),
}

SENTIMENTS = {
    'NEU': 0,
    'POS': 1,
    'NEG': -1
}

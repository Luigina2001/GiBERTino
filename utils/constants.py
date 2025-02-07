import os.path as osp


DATA_DIR = "data"
MOLWENI = 'MOLWENI'
STAC = 'STAC'
MINECRAFT = 'MINECRAFT'

DATASETS = {
    MOLWENI: osp.join(DATA_DIR, MOLWENI),
    STAC: osp.join(DATA_DIR, STAC),
    MINECRAFT: osp.join(DATA_DIR, MINECRAFT),
}

SENTIMENTS = {
    'NEUTRAL': 0,
    'POSITIVE': 1,
    'NEGATIVE': -1
}

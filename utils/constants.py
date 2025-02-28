import os.path as osp

from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, BinaryAUROC, AUROC

DATA_DIR = "data"
MOLWENI = 'MOLWENI'
STAC = 'STAC'
MINECRAFT = 'MINECRAFT'
NUM_RELATIONS = 12
MAX_SENTENCE_LEN = 39

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

EDGE_TYPES = [
    'Clarification_question', 'Explanation', 'Contrast', 'Comment',
    'Elaboration', 'Result', 'QAP', 'Correction', 'Narration',
    'Acknowledgement', 'Q-Elab', 'Continuation'
]

NEGATIVE_SAMPLES_RATIO = 50
BATCH_SIZE = 32
NUM_WORKERS = 4

METRICS = {
    "link": {
        "accuracy": Accuracy(task='binary'),
        "precision": Precision(task='binary'),
        "recall": Recall(task='binary'),
        "f1": F1Score(task='binary'),
        "roc": BinaryAUROC(),
    },
    "rel": {
        "accuracy": Accuracy,
        "precision": Precision,
        "recall": Recall,
        "f1": F1Score,
        "roc": AUROC,
    }
}

import os.path as osp

from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, BinaryAUROC, AUROC

DATA_DIR = "data"
MOLWENI = 'MOLWENI'
STAC = 'STAC'
MINECRAFT = 'MINECRAFT'
BALANCED = 'BALANCED'

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

NEGATIVE_SAMPLES_RATIO = 50
BATCH_SIZE = 32
NUM_WORKERS = 4
VAL_SPLIT_RATIO = 0.1

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

RELATIONS = {
    "BALANCED": [
        'Clarification_question', 'Explanation', 'Contrast', 'Comment',
        'Elaboration', 'Result', 'QAP', 'Correction', 'Narration',
        'Acknowledgement', 'Q-Elab', 'Continuation'
    ],
    "MINECRAFT": ["Acknowledgement",
                  "Continuation",
                  "Elaboration",
                  "Comment",
                  "Result",
                  "Confirmation_question",
                  "QAP",
                  "Clarification_question",
                  "Contrast",
                  "Correction",
                  "Narration",
                  "Alternation",
                  "Sequence",
                  "Q-Elab",
                  "Conditional",
                  "Explanation"],
    "MOLWENI": ["QAP",
                "Comment",
                "Clarification_question",
                "Continuation",
                "Acknowledgement",
                "Conditional",
                "Contrast",
                "Explanation",
                "Elaboration",
                "Result",
                "Correction",
                "Q-Elab",
                "Parallel",
                "Background",
                "Alternation",
                "Narration"],
    "STAC": [
        "Comment",
        "Elaboration",
        "QAP",
        "Q-Elab",
        "Explanation",
        "Result",
        "Continuation",
        "Acknowledgement",
        "Contrast",
        "Conditional",
        "Correction",
        "Background",
        "Parallel",
        "Alternation",
        "Clarification_question",
        "Narration",
    ]
}

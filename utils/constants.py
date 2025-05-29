import os.path as osp

from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AUROC

DATA_DIR = "data"
MOLWENI = "MOLWENI"
STAC = "STAC"
MINECRAFT = "MINECRAFT"
BALANCED = "BALANCED"

MAX_SENTENCE_LEN = 39

DATASETS = {
    MOLWENI: osp.join(DATA_DIR, MOLWENI),
    STAC: osp.join(DATA_DIR, STAC),
    MINECRAFT: osp.join(DATA_DIR, MINECRAFT),
}

SENTIMENTS = {
    "NEU": 0,
    "POS": 1,
    "NEG": -1
}

NEGATIVE_SAMPLES_RATIO = 50
BATCH_SIZE = 32
NUM_WORKERS = 4
VAL_SPLIT_RATIO = 0.1

METRICS = {
    "link": {
        "accuracy": Accuracy(task="binary"),
        "precision": Precision(task="binary"),
        "recall": Recall(task="binary"),
        "f1": F1Score(task="binary"),
        "roc": AUROC(task="binary"),
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
        "Clarification_question",
        "Explanation",
        "Contrast",
        "Comment",
        "Elaboration",
        "Result",
        "QAP",
        "Correction",
        "Narration",
        "Acknowledgement",
        "Q-Elab",
        "Continuation"
    ],

    "MINECRAFT": [
        "Clarification_question",
        "Explanation",
        "Contrast",
        "Comment",
        "Elaboration",
        "Result",
        "QAP",
        "Correction",
        "Narration",
        "Acknowledgement",
        "Q-Elab",
        "Continuation",
        "Alternation",
        "Conditional",
        "Sequence",
        "Confirmation_question"
    ],

    "MOLWENI": [
        "Clarification_question",
        "Explanation",
        "Contrast",
        "Comment",
        "Elaboration",
        "Result",
        "QAP",
        "Correction",
        "Narration",
        "Acknowledgement",
        "Q-Elab",
        "Continuation",
        "Alternation",
        "Conditional",
        "Background",
        "Parallel"
    ],

    "STAC": [
        "Clarification_question",
        "Explanation",
        "Contrast",
        "Comment",
        "Elaboration",
        "Result",
        "QAP",
        "Correction",
        "Narration",
        "Acknowledgement",
        "Q-Elab",
        "Continuation",
        "Alternation",
        "Conditional",
        "Background",
        "Parallel"
    ],

    "UNIFIED": [
        "Clarification_question",
        "Explanation",
        "Contrast",
        "Comment",
        "Elaboration",
        "Result",
        "QAP",
        "Correction",
        "Narration",
        "Acknowledgement",
        "Q-Elab",
        "Continuation",
        "Alternation",
        "Conditional",
        "Sequence",
        "Confirmation_question",
        "Background",
        "Parallel"
    ]

}

RELATIONS_COLOR_MAPS = {
    "Comment": "#e6194b",             # Strong Red
    "Elaboration": "#3cb44b",         # Vivid Green
    "QAP": "#ffe119",                 # Bright Yellow
    "Q-Elab": "#0082c8",              # Medium Blue
    "Explanation": "#f58231",         # Orange
    "Result": "#911eb4",              # Deep Purple
    "Continuation": "#46f0f0",        # Cyan
    "Acknowledgement": "#f032e6",     # Magenta
    "Contrast": "#d2f53c",            # Lime
    "Conditional": "#fabebe",         # Pink
    "Correction": "#008080",          # Teal
    "Background": "#e6beff",          # Lavender
    "Parallel": "#aa6e28",            # Brown
    "Alternation": "#fffac8",         # Pale Yellow
    "Clarification_question": "#800000", # Maroon
    "Narration": "#aaffc3",           # Mint
    "Confirmation_question": "#808000", # Olive
    "Sequence": "#ffd8b1",            # Apricot
}

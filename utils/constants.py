import os.path as osp
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, BinaryAUROC, AUROC

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
    "Comment": "#FF5733",  # Reddish Orange
    "Elaboration": "#33A1FF",  # Light Blue
    "QAP": "#33FF57",  # Green
    "Q-Elab": "#FFC300",  # Yellow
    "Explanation": "#C70039",  # Deep Red
    "Result": "#900C3F",  # Dark Purple
    "Continuation": "#581845",  # Dark Violet
    "Acknowledgement": "#8E44AD",  # Purple
    "Contrast": "#2E86C1",  # Blue
    "Conditional": "#27AE60",  # Dark Green
    "Correction": "#E74C3C",  # Red
    "Background": "#1ABC9C",  # Teal
    "Parallel": "#D35400",  # Orange
    "Alternation": "#F39C12",  # Gold
    "Clarification_question": "#3498DB",  # Sky Blue
    "Narration": "#2C3E50",  # Dark Blue
    "Confirmation_question": "#FF33A8",  # Pink
    "Sequence": "#76448A",  # Deep Purple
}

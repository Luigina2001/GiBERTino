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

RELATIONS_COUNTS = {
    "STAC": {
        "Clarification_question": 386,
        "Explanation": 444,
        "Contrast": 450,
        "Comment": 1725,
        "Elaboration": 873,
        "Result": 360,
        "QAP": 2437,
        "Correction": 194,
        "Narration": 85,
        "Acknowledgement": 1254,
        "Q-Elab": 518,
        "Continuation": 975,
        "Alternation": 100,
        "Conditional": 117,
        "Background": 88,
        "Parallel": 176
    },
    "MINECRAFT": {
        "Clarification_question": 569,
        "Explanation": 31,
        "Contrast": 225,
        "Comment": 1027,
        "Elaboration": 2259,
        "Result": 5928,
        "QAP": 1119,
        "Correction": 1229,
        "Narration": 2577,
        "Acknowledgement": 2528,
        "Q-Elab": 135,
        "Continuation": 1250,
        "Alternation": 109,
        "Conditional": 56,
        "Sequence": 19,
        "Confirmation_question": 560
    },
    "MOLWENI": {
        "Clarification_question": 16949,
        "Explanation": 1061,
        "Contrast": 880,
        "Comment": 22511,
        "Elaboration": 1624,
        "Result": 1747,
        "QAP": 14172,
        "Correction": 916,
        "Narration": 204,
        "Acknowledgement": 2281,
        "Q-Elab": 2122,
        "Continuation": 4680,
        "Alternation": 178,
        "Conditional": 703,
        "Background": 260,
        "Parallel": 166
    },
    "BALANCED": {
        "Clarification_question": 15589,
        "Explanation": 1246,
        "Contrast":1154,
        "Comment": 21794,
        "Elaboration": 2488,
        "Result": 3123,
        "QAP": 15612,
        "Correction": 1158,
        "Narration": 788,
        "Acknowledgement": 3512,
        "Q-Elab": 2455,
        "Continuation": 4998
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

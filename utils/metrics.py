import torch
from typing import List
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.loggers import TensorBoardLogger
from sentence_transformers import SentenceTransformer, util
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score

from .utils import get_device


class Metrics:
    def __init__(self,
                 num_classes: int = 12,
                 sentence_model: str = "Alibaba-NLP/gte-modernbert-base",
                 log_dir: str = "tb_logs",
                 logger_name: str = "gan_model_v0"):

        self.logger = TensorBoardLogger(log_dir, name=logger_name)
        self.num_classes = num_classes
        self.device = get_device()
        self.sentence_model = sentence_model
        self.writer = SummaryWriter(log_dir=log_dir)

        # Link prediction (Binary)
        self.link_accuracy = Accuracy(task="binary", threshold=0.5).to(self.device)
        self.link_precision = Precision(task="binary").to(self.device)
        self.link_recall = Recall(task="binary").to(self.device)
        self.link_f1_score = F1Score(task="binary").to(self.device)

        # Relation prediction (Multiclass)
        self.relation_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes,
                                          average="macro").to(self.device)
        self.relation_precision = Precision(task="multiclass", num_classes=self.num_classes,
                                            average="macro").to(self.device)
        self.relation_recall = Recall(task="multiclass", num_classes=self.num_classes,
                                      average="macro").to(self.device)
        self.relation_f1_score = F1Score(task="multiclass", num_classes=self.num_classes,
                                         average="macro").to(self.device)

        self.sbert_model = SentenceTransformer(self.sentence_model).to(self.device)
        self.sbert_model.eval()

    def log_metrics(self, metrics: dict, stage: str, step: int):
        if self.logger:
            for key, value in metrics.items():
                self.logger.experiment.add_scalar(f"{stage}/{key}", value, step)
                self.writer.add_scalar(f"{stage}/{key}", value, step)

    def compute_link_metrics(self, predictions: torch.Tensor, labels: torch.Tensor,
                             stage: str, step: int, reset: bool = False) -> dict:
        predictions = predictions.flatten()
        labels = labels.flatten().long()

        metrics = {
            "link_accuracy": self.link_accuracy(predictions, labels),
            "link_precision": self.link_precision(predictions, labels),
            "link_recall": self.link_recall(predictions, labels),
            "link_f1_score": self.link_f1_score(predictions, labels)
        }

        self.log_metrics(metrics, stage, step)

        if reset:
            self.reset_link_metrics()

        return metrics

    def compute_relation_metrics(self, predictions: torch.Tensor, labels: torch.Tensor,
                                 stage: str, step: int, reset: bool = False) -> dict:
        if predictions.ndim == 1:  # if they are classes
            predictions = predictions.long()
        else:  # if they are logits/probabilities
            predictions = predictions.argmax(dim=1)
        metrics = {
            "relation_accuracy": self.relation_accuracy(predictions, labels),
            "relation_precision": self.relation_precision(predictions, labels),
            "relation_recall": self.relation_recall(predictions, labels),
            "relation_f1_score": self.relation_f1_score(predictions, labels)
        }

        self.log_metrics(metrics, stage, step)

        if reset:
            self.reset_relation_metrics()

        return metrics

    def compute_sentence_bert_similarity(self, real_sentences: List[str], fake_sentences: List[str],
                                         stage: str, step: int, batch_size: int = 8) -> float:
        with torch.no_grad():
            embedding_real = self.sbert_model.encode(
                real_sentences,
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=str(self.device)
            )

            embedding_fake = self.sbert_model.encode(
                fake_sentences,
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=str(self.device)
            )

        cosine_sim = util.pytorch_cos_sim(embedding_real, embedding_fake).diag().mean()
        self.log_metrics({"sentence_bert_similarity": cosine_sim}, stage, step)

        return cosine_sim.item()

    def reset_link_metrics(self):
        self.link_accuracy.reset()
        self.link_precision.reset()
        self.link_recall.reset()
        self.link_f1_score.reset()

    def reset_relation_metrics(self):
        self.relation_accuracy.reset()
        self.relation_precision.reset()
        self.relation_precision.reset()
        self.relation_f1_score.reset()

    def reset_all_metrics(self):
        self.reset_link_metrics()
        self.reset_relation_metrics()

    def close_writer(self):
        self.writer.close()

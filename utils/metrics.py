import torch

from typing import List, Literal

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from torch.utils.tensorboard import SummaryWriter
from lightning.pytorch.loggers import TensorBoardLogger

from .constants import NUM_RELATIONS, BATCH_SIZE, METRICS
from .utils import get_device


class Metrics:
    def __init__(self, num_classes: int = NUM_RELATIONS + 1,
                 sentence_model: str = 'Alibaba-NLP/gte-modernbert-base',
                 log_dir: str = "lightning_logs", logger_name: str = "GiBERTino"):

        self.logger = TensorBoardLogger(name=logger_name, save_dir=log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.num_classes = num_classes
        self.device = get_device()

        for metric in METRICS["link"]:
            setattr(self, f"link_{metric}", METRICS["link"][metric].to(self.device))

        for metric in METRICS["rel"]:
            setattr(self, f"rel_{metric}",
                    METRICS["rel"][metric](task='multiclass', num_classes=self.num_classes, average='macro').to(
                        self.device))

        self.sbert_model = SentenceTransformer(sentence_model).to(self.device)
        self.sbert_model.eval()

    def log(self, metrics: dict, stage: str, step: int):
        for key, value in metrics.items():
            self.logger.experiment.add_scalar(f"{stage}/{key}", value, step)
            self.writer.add_scalar(f"{stage}/{key}", value, step)

    def compute_metrics(self, preds: torch.Tensor, target: torch.Tensor, metric_type: Literal['link', 'rel'],
                        stage: str, step: int) -> dict:
        metrics = {}
        for metric in METRICS[metric_type]:
            metrics[f"{metric_type}_{metric}"] = getattr(self, f"{metric_type}_{metric}")(preds, target).item()

        self.log(metrics, stage, step)

        return metrics

    def compute_sbert_similarity(self, preds: List[str], target: List[str], stage: str, step: int,
                                 batch_size: int = BATCH_SIZE):
        # compute sentence embeddings
        pred_embeddings = self.sbert_model.encode(preds, batch_size=batch_size, convert_to_tensor=True,
                                                  show_progress_bar=False).to(self.device)
        target_embeddings = self.sbert_model.encode(target, batch_size=batch_size, convert_to_tensor=True,
                                                    show_progress_bar=False).to(self.device)

        # compute cosine similarity
        similarity = cos_sim(pred_embeddings, target_embeddings).diag().mean().item()
        self.log({f"{stage}/sbert_similarity": similarity}, stage, step)

        return similarity

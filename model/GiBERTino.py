from typing import Literal, Optional

import torch
import torch.nn as nn
import torch_geometric
import lightning as L
import torch.nn.functional as F

from utils import print_metrics
from utils.metrics import Metrics
from utils.constants import NUM_RELATIONS


class GiBERTino(L.LightningModule):
    def __init__(self, model: str, in_channels: int, hidden_channels: int, num_layers: int,
                 checkpoint_path: Optional[str] = None):
        super().__init__()

        self.model = getattr(torch_geometric.nn, model)(in_channels=in_channels, hidden_channels=hidden_channels,
                                                        num_layers=num_layers)

        self.link_classifier = nn.Sequential(
            # Multiply by 2 to account for both source (src) and destination (dst) node embeddings,
            # and add 1 to include the cosine similarity, which is concatenated to the embeddings.
            nn.Linear(hidden_channels * 2 + 1, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

        self.rel_classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, NUM_RELATIONS + 1),
            nn.Softmax(dim=-1)
        )

        if checkpoint_path is not None:
            self.load_state_dict(torch.load(checkpoint_path))

        self.link_loss = torch.nn.BCEWithLogitsLoss()
        self.rel_loss = torch.nn.CrossEntropyLoss()

        self.metrics = Metrics()

        # save hyperparameters when saving checkpoint
        self.save_hyperparameters()

    def _predict(self, node_embeddings, edge_index, predict: Literal['link', 'rel']):
        src, dst = edge_index
        embeddings = torch.cat([node_embeddings[src], node_embeddings[dst]], dim=-1)

        src_emb = F.normalize(node_embeddings[src], dim=-1)
        dst_emb = F.normalize(node_embeddings[dst], dim=-1)
        cosine_sim = F.cosine_similarity(src_emb, dst_emb, dim=-1, eps=1e-8).unsqueeze(-1)

        if predict == 'link':
            embeddings = torch.cat([embeddings, cosine_sim], dim=-1)
            return self.link_classifier(embeddings).squeeze(dim=-1)
        return self.rel_classifier(embeddings)

    def forward(self, batch):
        x, edge_index = batch["edu"].x, batch["edu", "to", "edu"].edge_index
        embeddings = self.model(x, edge_index)

        link_logits = self._predict(embeddings, edge_index, 'link')
        rel_probs = self._predict(embeddings, edge_index, 'rel')

        return link_logits, rel_probs

    def compute_loss(self, batch, stage):
        link_logits, rel_probs = self.forward(batch)
        link_labels = batch["edu", "to", "edu"].link_labels.float()
        rel_labels = batch["edu", "to", "edu"].rel_labels

        link_metrics = self.metrics.compute_metrics(link_logits, link_labels, 'link', stage, self.global_step)
        rel_metrics = self.metrics.compute_metrics(rel_probs, rel_labels, 'rel', stage, self.global_step)

        self.metrics.log({'link_accuracy': link_metrics['link_accuracy'], 'rel_accuracy': rel_metrics['rel_accuracy']},
                         'train', self.global_step)

        link_loss = self.link_loss(link_logits, link_labels)
        rel_loss = self.rel_loss(rel_probs, rel_labels)

        loss = link_loss + rel_loss

        self.log_dict({f'{stage}/loss': loss, f'{stage}/link_accuracy': link_metrics['link_accuracy'],
                       f'{stage}/rel_accuracy': rel_metrics['rel_accuracy']}, on_step=True, on_epoch=True,
                      batch_size=batch.batch_size, prog_bar=True, logger=True)

        if self.global_step % self.trainer.log_every_n_steps == 0:  # noqa
            print_metrics(self.global_step, {
                f"{stage}/link_loss": link_loss.item(),
                f"{stage}/rel_loss": rel_loss.item(),
                f"{stage}/total_loss": loss.item(),
                f"{stage}/link_accuracy": link_metrics["link_accuracy"],
                f"{stage}/link_precision": link_metrics["link_precision"],
                f"{stage}/link_recall": link_metrics["link_recall"],
                f"{stage}/rel_accuracy": rel_metrics["rel_accuracy"],
                f"{stage}/rel_precision": rel_metrics["rel_precision"],
                f"{stage}/rel_recall": rel_metrics["rel_recall"]
            })

        return loss

    def training_step(self, batch, batch_idx):
        return self.compute_loss(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.compute_loss(batch, 'val')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.parameters())
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler
            }
        }

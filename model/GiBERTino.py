from typing import Literal, Optional

import lightning as L
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric
from transformers import AutoModel, AutoTokenizer

from utils import print_metrics, log_inverse_freq_weights
from utils.constants import RELATIONS, RELATIONS_COUNTS
from utils.metrics import Metrics

REL_EMBEDDING_DIM = 64


class GiBERTino(L.LightningModule):
    def __init__(
        self,
        gnn_model: Literal["GCN", "GAT", "GraphSAGE"],
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dataset_name: Literal["STAC", "MOLWENI", "MINECRAFT", "BALANCED"],
        num_relations: int,
        alpha: float = 0.5,
        tokenizer: str = "Alibaba-NLP/gte-modernbert-base",
        bert_model: str = "Alibaba-NLP/gte-modernbert-base",
        lr: float = 1e-3,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()

        # save hyperparameters when saving checkpoint
        self.save_hyperparameters()

        self.gnn_model = getattr(torch_geometric.nn, gnn_model)(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
        )

        self.num_relations = num_relations

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.bert_model = AutoModel.from_pretrained(bert_model)
        self.bert_model.eval()

        # Embedding layer for relations
        # NUM_RELATIONS = len(RELATIONS["UNIFIED"]) + 1
        self.relation_embeddings = nn.Embedding(self.num_relations, REL_EMBEDDING_DIM)
        # Initialize relation embeddings to promote stable training start while
        # allowing gradual specialization.
        # Matches BERT-style initialization practices for embedding layers.
        nn.init.normal_(self.relation_embeddings.weight, mean=0.0, std=0.02)

        # Link prediction classifier
        self.link_classifier = nn.Sequential(
            # Input size: hidden_channels * 2 (src, dst, diff, multi) + 1 (cosine similarity)
            nn.Linear(hidden_channels * 4 + 1, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

        # Relation prediction classifier
        self.rel_classifier = nn.Sequential(
            nn.Linear(hidden_channels * 4, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, self.num_relations)
        )

        if checkpoint_path is not None:
            self.load_state_dict(torch.load(checkpoint_path))

        class_counts = [0] * self.num_relations
        for rel, count in RELATIONS_COUNTS[dataset_name].items():
            class_counts[RELATIONS["UNIFIED"].index(rel)] = count

        self.class_weights = torch.tensor(log_inverse_freq_weights(class_counts), dtype=torch.float32)
        self.alpha = alpha
        self.link_loss = torch.nn.BCEWithLogitsLoss()
        self.rel_loss = torch.nn.CrossEntropyLoss(weight=self.class_weights)

        self.metrics = Metrics(self.num_relations)

        self.lr = lr

    def _predict(
        self, node_embeddings, edge_index, predict: Literal["link", "rel"]
    ):
        src, dst = edge_index
        h_src, h_dst = node_embeddings[src], node_embeddings[dst]
        embeddings = torch.cat([h_src, h_dst, torch.abs(h_src - h_dst), h_src * h_dst], dim=-1)

        if predict == "link":
            cosine_sim = F.cosine_similarity(h_src, h_dst, dim=-1, eps=1e-8).unsqueeze(-1)
            embeddings = torch.cat([embeddings, cosine_sim], dim=-1)
            return self.link_classifier(embeddings).squeeze(dim=-1)

        # teacher-forcing on relation classification
        return self.rel_classifier(embeddings)

    def forward(self, batch):
        # tokenize raw edus
        # AutoTokenizer expects a flat list of texts but the dataloader returns
        # a list of lists of texts
        # keep track of which edus correspond to which graph
        outputs = self.bert_model(**batch["edu"].edus)
        # the last hidden state contains token-level contextualized embeddings:
        # for each edu, we'll have an embedding for each token in the edu
        # token-level embeddings correspond to local embeddings, in order to
        # get global-level embeddings we need to perform a sort of pooling
        # operation
        flat_local_embeddings = outputs.last_hidden_state.mean(dim=1)
        x, edge_index = batch["edu"].x, batch["edu", "to", "edu"].edge_index
        x = torch.cat((x, flat_local_embeddings), dim=-1)
        # use speaker and turn + text embeddings
        embeddings = self.gnn_model(x.float(), edge_index)

        link_logits = self._predict(embeddings, edge_index, "link")
        link_labels = batch["edu", "to", "edu"].link_labels
        edge_index_gold = edge_index[:, link_labels == 1] # get the original graph structure - teacher forcing
        rel_logits = self._predict(embeddings, edge_index_gold, "rel")

        return link_logits, rel_logits

    def compute_loss(self, batch, stage):
        link_logits, rel_logits = self.forward(batch)
        link_labels = batch["edu", "to", "edu"].link_labels.float()
        rel_labels = batch["edu", "to", "edu"].rel_labels

        link_metrics = self.metrics.compute_metrics(
            link_logits, link_labels, "link", stage, self.global_step
        )
        rel_metrics = self.metrics.compute_metrics(
            rel_logits, rel_labels, "rel", stage, self.global_step
        )

        self.metrics.log(
            {
                "link_accuracy": link_metrics["link_accuracy"],
                "rel_accuracy": rel_metrics["rel_accuracy"],
            },
            "train",
            self.global_step,
        )

        link_loss = self.link_loss(link_logits.float(), link_labels)
        rel_loss = self.rel_loss(rel_logits, rel_labels)

        loss = self.alpha * link_loss + (1 - self.alpha) * rel_loss

        self.log_dict(
            {
                f"{stage}_loss": loss,
                f"{stage}_link_accuracy": link_metrics["link_accuracy"],
                f"{stage}_rel_accuracy": rel_metrics["rel_accuracy"],
            },
            on_step=True,
            on_epoch=True,
            batch_size=batch.batch_size,
            prog_bar=True,
            logger=True,
        )

        if self.global_step % self.trainer.log_every_n_steps == 0:  # noqa
            print_metrics(
                self.global_step,
                {
                    f"{stage}_link_loss": link_loss.item(),
                    f"{stage}_rel_loss": rel_loss.item(),
                    f"{stage}_total_loss": loss.item(),
                    f"{stage}_link_accuracy": link_metrics["link_accuracy"],
                    f"{stage}_link_precision": link_metrics["link_precision"],
                    f"{stage}_link_recall": link_metrics["link_recall"],
                    f"{stage}_rel_accuracy": rel_metrics["rel_accuracy"],
                    f"{stage}_rel_precision": rel_metrics["rel_precision"],
                    f"{stage}_rel_recall": rel_metrics["rel_recall"],
                },
            )

        return loss

    def training_step(self, batch, batch_idx):
        return self.compute_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.compute_loss(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(lr=self.lr, params=self.parameters())
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler}}

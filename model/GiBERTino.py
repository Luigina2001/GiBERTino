from typing import Literal, Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from transformers import AutoTokenizer, AutoModel

from utils import print_metrics
from utils.constants import BALANCED, RELATIONS
from utils.metrics import Metrics


class GiBERTino(L.LightningModule):
    def __init__(self, gnn_model: Literal['GCN', 'GAT', 'GraphSAGE'], in_channels: int,
                 hidden_channels: int, num_layers: int,
                 alpha: float = 0.5,
                 tokenizer: str = 'Alibaba-NLP/gte-modernbert-base',
                 bert_model: str = 'Alibaba-NLP/gte-modernbert-base',
                 lr: float = 1e-3,
                 relations: str = BALANCED,
                 checkpoint_path: Optional[str] = None):
        super().__init__()

        self.gnn_model = getattr(torch_geometric.nn, gnn_model)(
            in_channels=in_channels, hidden_channels=hidden_channels,
            num_layers=num_layers)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.bert_model = AutoModel.from_pretrained(bert_model)

        # Embedding layer for relations
        NUM_RELATIONS = len(RELATIONS[relations]) + 1
        self.relation_embeddings = nn.Parameter(torch.randn(NUM_RELATIONS))

        # Link prediction classifier
        self.link_classifier = nn.Sequential(
            # Input size: hidden_channels * 2 (src + dst) + 1 (cosine similarity)
            nn.Linear(hidden_channels * 2 + 1, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

        # Relation prediction classifier
        self.rel_classifier = nn.Sequential(
            # Input size: hidden_channels * 2 (src + dst) + hidden_channels (relation embedding)
            nn.Linear(hidden_channels * 2 + hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, NUM_RELATIONS),
            nn.Softmax(dim=-1)
        )

        if checkpoint_path is not None:
            self.load_state_dict(torch.load(checkpoint_path))

        self.alpha = alpha
        self.link_loss = torch.nn.BCEWithLogitsLoss()
        self.rel_loss = torch.nn.CrossEntropyLoss()

        self.metrics = Metrics(NUM_RELATIONS)

        self.lr = lr

        # save hyperparameters when saving checkpoint
        self.save_hyperparameters()

    def _predict(self, node_embeddings, edge_index, edge_rel,
                 predict: Literal['link', 'rel']):
        src, dst = edge_index
        embeddings = torch.cat([node_embeddings[src], node_embeddings[dst]],
                               dim=-1)

        if predict == 'link':
            # src_emb = F.normalize(node_embeddings[src], dim=-1)
            # dst_emb = F.normalize(node_embeddings[dst], dim=-1)
            cosine_sim = F.cosine_similarity(node_embeddings[src], node_embeddings[dst],
                                             dim=-1, eps=1e-8).unsqueeze(-1)
            embeddings = torch.cat([embeddings, cosine_sim], dim=-1)
            return self.link_classifier(embeddings).squeeze(dim=-1)

        rel_emb = self.relation_embeddings(edge_rel)
        embeddings = torch.cat([embeddings, rel_emb], dim=-1)
        return self.rel_classifier(embeddings)

    def forward(self, batch):
        # tokenize raw edus
        # AutoTokenizer expects a flat list of texts but the dataloader returns
        # a list of lists of texts
        # keep track of which edus correspond to which graph
        # edu_indices = [len(edus) for edus in batch["edu"].edus]
        # flat_edus = [edu for edus in batch["edu"].edus for edu in edus]
        edge_rel = batch["edu", "to", "edu"].rel_labels

        outputs = self.bert_model(**batch["edu"].edus)
        # the last hidden state contains token-level contextualized embeddings:
        # for each edu, we'll have an embedding for each token in the edu
        # token-level embeddings correspond to local embeddings, in order to
        # get global-level embeddings we need to perform a sort of pooling
        # operation
        flat_local_embeddings = outputs.last_hidden_state

        # get local embeddings per graph, returned as a tuple
        # local_embeddings = torch.split(flat_local_embeddings, edu_indices)
        # local_attention_masks = torch.split(tokenized_edus["attention_mask"], edu_indices)

        # node_embeddings = []

        # for i in range(batch.batch_size):
        #     # the attention mask indices indicate which tokens are actual words (1)
        #     # and which are padding tokens (0)
        #     input_mask_expanded = local_attention_masks[i].unsqueeze(-1).expand(local_embeddings[i].shape).float()
        #     valid_local_tokens = local_embeddings[i] * input_mask_expanded
        #
        #     cumulative_attention = input_mask_expanded.cumsum(dim=1)
        #     contextualized_global_embeddings = torch.cumsum(valid_local_tokens, dim=1) / cumulative_attention
        #     node_embeddings.append(torch.cat((local_embeddings[i], contextualized_global_embeddings), dim=1))

        # node_embeddings = torch.cat(node_embeddings, dim=0)
        flat_local_embeddings = flat_local_embeddings.mean(dim=-1)
        x, edge_index = batch["edu"].x, batch["edu", "to", "edu"].edge_index
        x = torch.cat((x, flat_local_embeddings), dim=-1)
        embeddings = self.gnn_model(x, edge_index)

        link_logits = self._predict(embeddings, edge_index, edge_rel, 'link')
        rel_probs = self._predict(embeddings, edge_index, edge_rel, 'rel')

        return link_logits, rel_probs

    def compute_loss(self, batch, stage):
        link_logits, rel_probs = self.forward(batch)
        link_labels = batch["edu", "to", "edu"].link_labels.float()
        rel_labels = batch["edu", "to", "edu"].rel_labels

        link_metrics = self.metrics.compute_metrics(link_logits, link_labels, 'link', stage, self.global_step)
        rel_metrics = self.metrics.compute_metrics(rel_probs, rel_labels, 'rel', stage, self.global_step)

        self.metrics.log({'link_accuracy': link_metrics['link_accuracy'],
                          'rel_accuracy': rel_metrics['rel_accuracy']},
                         'train', self.global_step)

        link_loss = self.link_loss(link_logits, link_labels)
        rel_loss = self.rel_loss(rel_probs, rel_labels)

        loss = self.alpha * link_loss + (1 - self.alpha) * rel_loss

        self.log_dict({f'{stage}/loss': loss,
                       f'{stage}/link_accuracy': link_metrics['link_accuracy'],
                       f'{stage}/rel_accuracy': rel_metrics['rel_accuracy']},
                      on_step=True, on_epoch=True,
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
        optimizer = torch.optim.AdamW(lr=self.lr, params=self.parameters())
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler
            }
        }

from typing import Literal
import pytorch_lightning as pl
import torch.optim
import torch_geometric
from torch import nn
from torch.optim.lr_scheduler import LinearLR

from dataset.sub_dialogue_datamodule import SubDialogueDataModule
from utils import get_device
from utils.metrics import Metrics
from utils.constants import NUM_RELATIONS
from utils import print_metrics


class GiBERTino(pl.LightningModule):
    def __init__(self, model_name: Literal['GCN', 'GAT'], **model_kwargs):
        super().__init__()

        self.save_hyperparameters()
        self.model = getattr(torch_geometric.nn, model_name)(**model_kwargs)
        hidden_channels = model_kwargs.get("hidden_channels")

        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

        self.relation_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 13),
            nn.Softmax(dim=-1)
        )

        self.link_pl = nn.BCEWithLogitsLoss()  # Binary classification loss for link prediction
        self.rel_pl = nn.CrossEntropyLoss()  # Loss for relation classification
        self.metrics = Metrics(num_classes=NUM_RELATIONS + 1)

    def forward(self, data):
        x, edge_index = data["edu"].x, data["edu", "to", "edu"].edge_index
        node_embeddings = self.model(x, edge_index)

        link_logits = self._predict_links(node_embeddings, edge_index)
        relation_logits = self._predict_relations(node_embeddings, edge_index)

        return link_logits, relation_logits

    def _predict_links(self, node_embeddings, edge_index):
        # candidate_edge_index: tensor di shape [2, num_candidate_edges]
        src = edge_index[0]  # Source nodes
        dst = edge_index[1]  # Destination nodes

        pair_embeddings = torch.cat([node_embeddings[src], node_embeddings[dst]], dim=-1)
        link_logits = self.link_predictor(pair_embeddings)  # shape [num_candidate_edges]
        return link_logits.squeeze()

    def _predict_relations(self, node_embeddings, edge_index):
        # candidate_edge_index: tensor di shape [2, num_candidate_edges]
        src = edge_index[0]  # Source nodes
        dst = edge_index[1]  # Destination nodes

        pair_embeddings = torch.cat([node_embeddings[src], node_embeddings[dst]], dim=-1)
        relation_probs = self.relation_predictor(pair_embeddings)  # shape [num_candidate_edges]
        return relation_probs

    def training_step(self, batch, batch_idx):
        link_logits, relation_probs = self(batch)
        link_labels = batch["edu", "to", "edu"].link_labels.float()
        link_loss = self.link_pl(link_logits, link_labels)

        relation_labels = batch["edu", "to", "edu"].relation_labels
        relation_loss = self.rel_pl(relation_probs, relation_labels)

        link_metrics = self.metrics.compute_link_metrics(
            predictions=link_logits,
            labels=link_labels,
            stage="train",
            step=self.global_step
        )

        relation_metrics = self.metrics.compute_relation_metrics(
            predictions=relation_probs,
            labels=relation_labels,
            stage="train",
            step=self.global_step
        )

        total_loss = link_loss + relation_loss
        losses = {
            "link_loss": link_loss,
            "relation_loss": relation_loss,
            "total_loss": total_loss
        }
        self.metrics.log_losses(losses, stage="train", step=self.global_step)

        losses.update(link_metrics)
        losses.update(relation_metrics)

        batch_size = batch["edu", "to", "edu"].link_labels.shape[0]
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/link_accuracy", link_metrics["link_accuracy"], on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=batch_size)
        self.log("train/relation_accuracy", relation_metrics["relation_accuracy"], on_step=True, on_epoch=True,
                 prog_bar=True, batch_size=batch_size)

        if self.global_step % 10 == 0:
            metrics_to_print = {
                "link_loss": link_loss.item(),
                "relation_loss": relation_loss.item(),
                "total_loss": total_loss.item(),
                "link_accuracy": link_metrics["link_accuracy"].item(),
                "link_precision": link_metrics["link_precision"].item(),
                "link_recall": link_metrics["link_recall"].item(),
                "relation_accuracy": relation_metrics["relation_accuracy"].item(),
                "relation_precision": relation_metrics["relation_precision"].item(),
                "relation_recall": relation_metrics["relation_recall"].item()
            }
            print_metrics(self.global_step, metrics_to_print)

        return total_loss

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.parameters())
        lr_scheduler = LinearLR(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler
            }
        }


if __name__ == "__main__":
    model = GiBERTino('GCN', in_channels=2304, hidden_channels=10, num_layers=3)
    data_module = SubDialogueDataModule("../data/BALANCED/graphs/test")
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()

    trainer = pl.Trainer(
        max_epochs=2,
        log_every_n_steps=10,
        accelerator=str(get_device()),
        logger=model.metrics.logger
    )
    trainer.fit(model=model, train_dataloaders=train_loader)

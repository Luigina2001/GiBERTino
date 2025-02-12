from typing import Literal
import pytorch_lightning as pl
import torch.optim
import torch_geometric
from torch import nn
from torch.optim.lr_scheduler import LinearLR

from dataset.sub_dialogue_datamodule import SubDialogueDataModule
from utils.metrics import Metrics
from utils.constants import NUM_RELATIONS


class GiBERTino(pl.LightningModule):
    def __init__(self, model_name: Literal['GCN', 'GAT'], **model_kwargs):
        super().__init__()

        self.save_hyperparameters()
        self.model = getattr(torch_geometric.nn, model_name)(**model_kwargs)
        hidden_channels = model_kwargs.get("hidden_channels")

        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 2),
            nn.LogSoftmax(dim=1)
        )

        self.link_pl = nn.NLLLoss()
        self.rel_pl = nn.CrossEntropyLoss()
        self.metrics = Metrics(num_classes=NUM_RELATIONS)

    def forward(self, data):
        x, edge_index = data["edu"].x, data["edu", "to", "edu"].edge_index
        node_embeddings = self.model(x, edge_index)

        link_probs = self._predict_links(node_embeddings, edge_index)
        # relation_probs = self._predict_relations(node_embeddings, edge_index)

        return link_probs

    def _predict_links(self, node_embeddings, edge_index):
        src, dst = edge_index
        src_emb = node_embeddings[src]
        dst_emb = node_embeddings[dst]

        pair_embeddings = torch.cat([src_emb, dst_emb], dim=1)
        link_probs = self.link_predictor(pair_embeddings)

        return link_probs

    def training_step(self, batch, batch_idx):
        print(batch)

        link_probs = self(batch)
        link_labels = batch["edu", "to", "edu"].link_labels
        link_loss = self.link_pl(link_probs, link_labels)

        relation_loss = 0
        # relation_labels = batch["edu", "to", "edu"].relation_labels
        # relation_loss = self.rel_pl(relation_probs, relation_labels)

        self.metrics.compute_link_metrics(
            predictions=link_probs.argmax(dim=1),
            labels=link_labels,
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

    trainer = pl.Trainer()
    trainer.fit(model=model, train_dataloaders=train_loader)

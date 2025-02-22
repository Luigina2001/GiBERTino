import argparse
from typing import Literal, Optional

import lightning as L
import torch
import torch.nn as nn
import torch_geometric

from dataset.dialogue_graph_datamodule import SubDialogueDataModule
from utils import get_device, print_metrics
from utils.metrics import Metrics
from utils.constants import NUM_RELATIONS


class GiBERTino(L.LightningModule):
    def __init__(self, model: str, in_channels: int, hidden_channels: int, num_layers: int,
                 checkpoint_path: Optional[str] = None):
        super().__init__()

        self.model = getattr(torch_geometric.nn, model)(in_channels=in_channels, hidden_channels=hidden_channels,
                                                        num_layers=num_layers)

        self.link_classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
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

        if predict == 'link':
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

        # if self.global_step % self.trainer.log_every_n_steps == 0:  # noqa
        #     print_metrics(self.global_step, {
        #         f"{stage}/link_loss": link_loss.item(),
        #         f"{stage}/rel_loss": rel_loss.item(),
        #         f"{stage}/total_loss": loss.item(),
        #         f"{stage}/link_accuracy": link_metrics["link_accuracy"],
        #         f"{stage}/link_precision": link_metrics["link_precision"],
        #         f"{stage}/link_recall": link_metrics["link_recall"],
        #         f"{stage}/rel_accuracy": rel_metrics["rel_accuracy"],
        #         f"{stage}/rel_precision": rel_metrics["rel_precision"],
        #         f"{stage}/rel_recall": rel_metrics["rel_recall"]
        #     })

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


def train(args):
    L.seed_everything(args.seed)

    model = GiBERTino(
        model=args.model_name,
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
    )

    data_module = SubDialogueDataModule(args.data_path)
    data_module.setup(stage="fit")

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        log_every_n_steps=args.log_every_n_steps,
        accelerator=str(get_device()),
        logger=model.metrics.logger,
        deterministic=True,
        enable_progress_bar=True,
    )

    trainer.fit(model=model, train_dataloaders=data_module.train_dataloader())


def argument_parser():
    parser = argparse.ArgumentParser(
        description="GiBERTino Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # General settings
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--log_every_n_steps", type=int, default=10, help="Log frequency in steps")
    parser.add_argument("--data_path", type=str, default="./data/BALANCED/graphs/", help="Path to training data")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to model checkpoint (optional)")

    # Model hyperparameters
    parser.add_argument("--model_name", type=str, choices=["GCN", "GAT"], default="GCN", help="Graph model type")
    parser.add_argument("--in_channels", type=int, default=2304, help="Input feature dimension")
    parser.add_argument("--hidden_channels", type=int, default=10, help="Hidden layer dimension")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of graph model layers")

    return parser


def main(args):
    train(args)


if __name__ == '__main__':
    args = argument_parser().parse_args()
    main(args)

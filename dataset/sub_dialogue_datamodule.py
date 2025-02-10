from typing import Optional
import pytorch_lightning as pl
from utils.constants import DATA_DIR
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, HeteroData
from dialogue_graph_dataset import DialogueGraphDataset


class SubDialogueDataModule(pl.LightningDataModule):

    def __init__(self, root: str = DATA_DIR, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_data = DialogueGraphDataset(self.root, "train")
            self.val_data = DialogueGraphDataset(self.root, "val")

        if stage == "test" or stage is None:
            self.test_data = DialogueGraphDataset(self.root, "test")

    @staticmethod
    def collate_fn(batch):
        """Custom collate to handle heterogeneous graphs."""
        return Batch.from_data_list([
            HeteroData(
                x=item["x"],
                edge_indices=item["edge_indices"],
                link_labels=item["link_labels"],
                relation_labels=item["relation_labels"]
            ) for item in batch
        ])

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers
        )

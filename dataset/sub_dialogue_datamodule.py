from typing import Optional
import pytorch_lightning as pl
from utils.constants import DATA_DIR
from torch_geometric.loader import DataLoader
from .dialogue_graph_dataset import DialogueGraphDataset


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
        if stage in ("fit", None):
            self.train_data = DialogueGraphDataset(self.root, "train")
            self.val_data = DialogueGraphDataset(self.root, "val")

        if stage in ("test", None):
            self.test_data = DialogueGraphDataset(self.root, "test")

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

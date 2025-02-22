import os

from lightning import LightningDataModule
from torch_geometric.loader import DataLoader

from .dialouge_graph_dataset import DialogueGraphDataset
from utils.constants import BATCH_SIZE, NUM_WORKERS, NEGATIVE_SAMPLES_RATIO


class SubDialogueDataModule(LightningDataModule):
    def __init__(self, root: str, batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS,
                 negative_sampling_ratio: float = NEGATIVE_SAMPLES_RATIO):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.negative_sampling_ratio = negative_sampling_ratio

        self.train_data = None
        self.val_data = None
        self.test_data = None

    def setup(self, stage: str):
        if stage == "fit":
            self.train_data = DialogueGraphDataset(root=os.path.join(self.root, 'train'),
                                                   negative_sampling_ratio=self.negative_sampling_ratio)
            self.val_data = DialogueGraphDataset(root=os.path.join(self.root, 'val'),
                                                 negative_sampling_ratio=self.negative_sampling_ratio)

        if stage == "test":
            self.test_data = DialogueGraphDataset(root=os.path.join(self.root, 'test'))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False, persistent_workers=True)

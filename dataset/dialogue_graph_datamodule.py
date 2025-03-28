import os
import random
import shutil

from lightning import LightningDataModule
from torch_geometric.loader import DataLoader

from utils.constants import BATCH_SIZE, NUM_WORKERS, NEGATIVE_SAMPLES_RATIO, VAL_SPLIT_RATIO
from .dialouge_graph_dataset import DialogueGraphDataset


class SubDialogueDataModule(LightningDataModule):
    def __init__(self, root: str, dataset_name: str, batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS,
                 negative_sampling_ratio: float = NEGATIVE_SAMPLES_RATIO,
                 val_split_ratio: float = VAL_SPLIT_RATIO):
        super().__init__()
        self.root = root
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.negative_sampling_ratio = negative_sampling_ratio
        self.val_split_ratio = val_split_ratio

        self.train_data = None
        self.val_data = None
        self.test_data = None

    def move_val_samples(self):
        train_path = os.path.join(self.root, "train")
        val_path = os.path.join(self.root, "val")

        if not os.path.exists(val_path):
            os.makedirs(val_path, exist_ok=True)
            all_train_files = [f for f in os.listdir(train_path) if f.endswith(".pt")]

            val_size = int(len(all_train_files) * self.val_split_ratio)
            val_files = random.sample(all_train_files, val_size)

            for file in val_files:
                src = os.path.join(train_path, file)
                dst = os.path.join(val_path, file)
                shutil.move(src, dst)

            print(f"Moved {len(val_files)} files from train to val.")

    def setup(self, stage: str):
        if stage == "fit":
            self.move_val_samples()
            self.train_data = DialogueGraphDataset(root=os.path.join(self.root, 'train'),
                                                   dataset_name=self.dataset_name,
                                                   negative_sampling_ratio=self.negative_sampling_ratio)
            self.val_data = DialogueGraphDataset(root=os.path.join(self.root, 'val'), dataset_name=self.dataset_name,
                                                 negative_sampling_ratio=self.negative_sampling_ratio)
        if stage == "test":
            self.test_data = DialogueGraphDataset(root=os.path.join(self.root, 'test'),
                                                  dataset_name=self.dataset_name, )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False)

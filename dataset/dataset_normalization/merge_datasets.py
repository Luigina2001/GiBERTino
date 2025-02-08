from utils import load_dataset, save_dataset
from utils.constants import MINECRAFT, MOLWENI, STAC
import random
import os.path as osp
from pathlib import Path

DATA_DIR = Path("../../data")
OUTPUT_DIR = DATA_DIR / "MERGED"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_files = [
    osp.join(DATA_DIR, MINECRAFT, "TRAIN_307_bert.json"),
    osp.join(DATA_DIR, MOLWENI, "train.json"),
    osp.join(DATA_DIR, STAC, "train_subindex.json")
]

test_files = [
    osp.join(DATA_DIR, MINECRAFT, "TEST_101_bert.json"),
    osp.join(DATA_DIR, MINECRAFT, "TEST_133.json"),
    osp.join(DATA_DIR, MOLWENI, "test.json"),
    osp.join(DATA_DIR, STAC, "test_subindex.json")
]

val_files = [
    osp.join(DATA_DIR, MINECRAFT, "DEV_32_bert.json"),
    osp.join(DATA_DIR, MINECRAFT, "VAL_100_bert.json"),
    osp.join(DATA_DIR, MOLWENI, "dev.json")
]

train_data = [item for f in train_files for item in load_dataset(f)]
test_data = [item for f in test_files for item in load_dataset(f)]
val_data = [item for f in val_files for item in load_dataset(f)]

# Take 10% of the STAC train to use as validation
stac_train_path = osp.join(DATA_DIR, STAC, "train_subindex.json")
stac_train = load_dataset(stac_train_path)
random.shuffle(stac_train)

split_idx = int(0.1 * len(stac_train))
stac_val = stac_train[:split_idx]
stac_train = stac_train[split_idx:]

# Add the new STAC data to the corresponding sets
train_data.extend(stac_train)
val_data.extend(stac_val)

random.shuffle(train_data)
random.shuffle(test_data)
random.shuffle(val_data)


save_dataset(train_data, str(osp.join(OUTPUT_DIR, "train.json")))
save_dataset(test_data, str(osp.join(OUTPUT_DIR, "test.json")))
save_dataset(val_data, str(osp.join(OUTPUT_DIR, "val.json")))

print(f"Data merged and saved in {OUTPUT_DIR}")

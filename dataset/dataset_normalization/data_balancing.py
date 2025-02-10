import json

from utils import load_dataset, save_dataset
import os


class DataBalancer:
    def __init__(self, dataset_paths, output_dir: str): # noqa
        self.dataset_paths = dataset_paths
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def balance_datasets(dataset):
        """
        Removes dialogues that contain relationships categorized as low-frequency.
        These relationships have been identified in explore_features.py as occurring
        in less than 1% of the total dataset.
        """

        low_freq_types = {"Confirmation_question", "Conditional", "Alternation",
                          "Parallel", "Background", "Sequence"}

        print("Relations removed:", low_freq_types)

        filtered_entries = [
            entry for entry in dataset
            if not any(rel["type"] in low_freq_types for rel in entry.get("relations", []))
        ]

        return filtered_entries

    def process_datasets(self):
        for name, path in self.dataset_paths.items():
            print(f"Processing {name}...")
            dataset = load_dataset(path)
            balanced_dataset = self.balance_datasets(dataset)
            output_path = os.path.join(self.output_dir, f"{name.lower()}.json")
            save_dataset(balanced_dataset, output_path)
            print(f"Saved balanced dataset to {output_path}")
        print("Data balancing completed!")


if __name__ == "__main__":
    # dataset_paths = {
    #     "TRAIN": "../../data/MERGED/train.json",
    #     "VAL": "../../data/MERGED/val.json",
    #     "TEST": "../../data/MERGED/test.json"
    # }
    #
    # balancer = DataBalancer(dataset_paths, "../../data/BALANCED/")
    # balancer.process_datasets()
    with open("../../data/MOLWENI/train.json", "r") as f:
        train_data = json.load(f)

    with open("../../data/BALANCED/test.json", "r") as f:
        test_data = json.load(f)

    with open("../../data/MOLWENI/test.json", "r") as f:
        val_data = json.load(f)

    print(f"train data: {len(train_data)} - val data: {len(val_data)} - test data: {len(test_data)}")


from utils import load_dataset, save_dataset


class DataCleaner:
    def __init__(self, dataset_paths): # noqa
        self.dataset_paths = dataset_paths

    @staticmethod
    def clean_dataset(dataset_name, dataset):
        for entry in dataset:
            # Remove 'speechturn' sub-attribute in STAC dataset
            for edu in entry.get("edus", []):
                if dataset_name in ["STAC_test", "STAC_train"]:
                    edu.pop("speechturn", None)

            # Replace relation types
            if "relations" in entry:
                for rel in entry["relations"]:
                    if rel["type"] == "Question_answer_pair" or rel["type"] == "Question-answer_pair":
                        rel["type"] = "QAP"
                    elif rel["type"] == "Q_Elab":
                        rel["type"] = "Q-Elab"

    def process_datasets(self):
        for name, path in self.dataset_paths.items():
            print(f"Processing {name}...")
            dataset = load_dataset(path)
            self.clean_dataset(name, dataset)
            save_dataset(dataset, path)
        print("Data cleaning completed!")


if __name__ == "__main__":
    dataset_paths = {
        "TEST_133": "../../data/MINECRAFT/TEST_133.json",
        "TEST_101_bert": "../../data/MINECRAFT/TEST_101_bert.json",
        "DEV_32_bert": "../../data/MINECRAFT/DEV_32_bert.json",
        "TRAIN_307_bert": "../../data/MINECRAFT/TRAIN_307_bert.json",
        "VAL_100_bert": "../../data/MINECRAFT/VAL_100_bert.json",
        "MOLWENI_dev": "../../data/MOLWENI/dev.json",
        "MOLWENI_test": "../../data/MOLWENI/test.json",
        "MOLWENI_train": "../../data/MOLWENI/train.json",
        "STAC_test": "../../data/STAC/test_subindex.json",
        "STAC_train": "../../data/STAC/train_subindex.json",
    }

    cleaner = DataCleaner(dataset_paths)
    cleaner.process_datasets()

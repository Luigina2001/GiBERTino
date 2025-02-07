import json


class DataCleaner:
    def __init__(self, dataset_paths):
        self.dataset_paths = dataset_paths

    @staticmethod
    def load_dataset(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_dataset(path, dataset):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=4, ensure_ascii=False)

    @staticmethod
    def clean_dataset(dataset_name, dataset):
        for entry in dataset:
            # Remove 'speechturn' sub-attribute in STAC datasets
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
            dataset = self.load_dataset(path)
            self.clean_dataset(name, dataset)
            self.save_dataset(path, dataset)
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

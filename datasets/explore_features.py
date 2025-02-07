import json


def load_dataset(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset


def explore_features(dataset, dataset_name):
    if not dataset:
        print("Dataset is empty or invalid.")

    # --- ALL AVAILABLE FEATURES IN THE DATASET ---
    keys = set()
    for entry in dataset:
        keys.update(entry.keys())
    print(f"\nAvailable features in the dataset {dataset_name}: ")
    print(keys)

    sub_attributes = set()
    for key in keys:
        for entry in dataset:
            if isinstance(entry.get(key), dict):
                sub_attributes.update(entry[key].keys())
            elif isinstance(entry.get(key), list) and entry[key] and isinstance(entry[key][0], dict):
                for item in entry[key]:
                    sub_attributes.update(item.keys())
    print(f"Sub-attributes: {sub_attributes}")

    # --- UNIQUE RELATIONSHIP TYPES ---
    relation_types = set()
    for entry in dataset:
        relation_types.update(rel["type"] for rel in entry["relations"])
    print("\nUnique relationship types in the dataset:")
    print(relation_types)


import json


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset


def explore_features(dataset, dataset_name):
    if not dataset:
        print("Dataset is empty or invalid.")
        return

    # Print all available features in the dataset
    keys = set()
    for entry in dataset:
        keys.update(entry.keys())
    print(f"\nAvailable features in the dataset {dataset_name}: ")
    print(keys)

    sub_attributes = set()
    for key in keys:
        for entry in dataset:
            if isinstance(entry.get(key), dict):
                sub_attributes.update(entry[key].keys())
            elif isinstance(entry.get(key), list) and entry[key] and isinstance(entry[key][0], dict):
                for item in entry[key]:
                    sub_attributes.update(item.keys())
    print("Sub-attributes:", sub_attributes)

    # Print all unique relationship types
    relation_types = set()
    for entry in dataset:
        if "relations" in entry:
            relation_types.update(rel["type"] for rel in entry["relations"])

    print("\nUnique relationship types in the dataset:")
    print(relation_types)


def compare_datasets(datasets):
    feature_sets = {name: set() for name in datasets}
    relation_sets = {name: set() for name in datasets}
    sub_attribute_sets = {name: set() for name in datasets}

    for name, dataset in datasets.items():
        for entry in dataset:
            feature_sets[name].update(entry.keys())
            if "relations" in entry:
                relation_sets[name].update(rel["type"] for rel in entry["relations"])
            for key in entry.keys():
                if isinstance(entry.get(key), dict):
                    sub_attribute_sets[name].update(entry[key].keys())
                elif isinstance(entry.get(key), list) and entry[key] and isinstance(entry[key][0], dict):
                    for item in entry[key]:
                        sub_attribute_sets[name].update(item.keys())

    # Compare feature sets
    print("\nComparison of dataset features:")
    all_features = set.union(*feature_sets.values())
    for name, features in feature_sets.items():
        missing_features = all_features - features
        if bool(missing_features):
            print(f"{name} is missing: {missing_features}")

    # Compare sub-attributes
    print("\nComparison of dataset sub-attributes:")
    all_sub_attributes = set.union(*sub_attribute_sets.values())
    for name, sub_attrs in sub_attribute_sets.items():
        missing_sub_attrs = all_sub_attributes - sub_attrs
        if bool(missing_sub_attrs):
            print(f"{name} is missing sub-attributes: {missing_sub_attrs}")

    # Compare relationship types
    print("\nComparison of relationship types:")
    all_relations = set.union(*relation_sets.values())
    for name, relations in relation_sets.items():
        missing_relations = all_relations - relations
        print(f"{name} is missing: {missing_relations}")


if __name__ == "__main__":

    dataset_paths = {
        "TEST_133": "../data/MINECRAFT/TEST_133.json",
        "TEST_101_bert": "../data/MINECRAFT/TEST_101_bert.json",
        "DEV_32_bert": "../data/MINECRAFT/DEV_32_bert.json",
        "TRAIN_307_bert": "../data/MINECRAFT/TRAIN_307_bert.json",
        "VAL_100_bert": "../data/MINECRAFT/VAL_100_bert.json",
        "MOLWENI_dev": "../data/MOLWENI/dev.json",
        "MOLWENI_test": "../data/MOLWENI/test.json",
        "MOLWENI_train": "../data/MOLWENI/train.json",
        "STAC_test": "../data/STAC/test_subindex.json",
        "STAC_train": "../data/STAC/train_subindex.json",
    }

    datasets = {}
    for name, path in dataset_paths.items():
        datasets[name] = load_dataset(path)
        explore_features(datasets[name], name)

    compare_datasets(datasets)


import os
from pathlib import Path

import numpy as np
from utils import load_dataset
from collections import defaultdict
import matplotlib.pyplot as plt

global final_global_relations


def explore_features(dataset, dataset_name):
    if not dataset:  # noqa
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


def count_relationships(dataset):
    relation_counts = defaultdict(int)

    for entry in dataset:
        if "relations" in entry:
            for rel in entry["relations"]:
                relation_counts[rel["type"]] += 1
    return relation_counts


def compare_datasets(datasets):  # noqa
    feature_sets = {name: set() for name in datasets}  # noqa
    relation_sets = {name: set() for name in datasets}  # noqa
    sub_attribute_sets = {name: set() for name in datasets}  # noqa
    relation_counts = {name: defaultdict(int) for name in datasets}  # noqa
    global_relation_counts = defaultdict(int)

    output_folder = Path("relationship_frequencies_plots")
    output_folder.mkdir(parents=True, exist_ok=True)

    for name, dataset in datasets.items():  # noqa
        for entry in dataset:
            feature_sets[name].update(entry.keys())
            if "relations" in entry:
                relation_sets[name].update(rel["type"] for rel in entry["relations"])
                for rel in entry["relations"]:
                    relation_counts[name][rel["type"]] += 1
                    global_relation_counts[rel["type"]] += 1
            for key in entry.keys():
                if isinstance(entry.get(key), dict):
                    sub_attribute_sets[name].update(entry[key].keys())
                elif isinstance(entry.get(key), list) and entry[key] and isinstance(entry[key][0], dict):
                    for item in entry[key]:
                        sub_attribute_sets[name].update(item.keys())

    # Compare feature sets
    print("\nComparison of dataset features:")
    all_features = set.union(*feature_sets.values())
    for name, features in feature_sets.items():  # noqa
        missing_features = all_features - features
        if bool(missing_features):
            print(f"{name} is missing: {missing_features}")

    # Compare sub-attributes
    print("\nComparison of dataset sub-attributes:")
    all_sub_attributes = set.union(*sub_attribute_sets.values())
    for name, sub_attrs in sub_attribute_sets.items():  # noqa
        missing_sub_attrs = all_sub_attributes - sub_attrs
        if bool(missing_sub_attrs):
            print(f"{name} is missing sub-attributes: {missing_sub_attrs}")

    # Compare relationship types
    print("\nComparison of relationship types:")
    all_relations = set.union(*relation_sets.values())
    final_global_relations.update(all_relations)
    for name, relations in relation_sets.items():  # noqa
        missing_relations = all_relations - relations
        print(f"{name} is missing: {missing_relations}")

    print("\n" + "=" * 30 + " RELATIONSHIP COUNTS PER DATASET " + "=" * 30 + "\n")

    for name, counts in relation_counts.items():  # noqa
        total_counts = []
        tot = 0
        print(f"\n{name} relationship counts:")
        for relation, count in counts.items():
            total_counts.append(count)
            tot += count
            print(f"  {relation}: {count}")
        mean_val = np.mean(total_counts)
        std_val = np.std(total_counts)
        percentile = np.percentile(total_counts, 25)

        print("\n" + "-" * 80)
        print(f"TOTAL RELATIONSHIPS: {tot}")
        print(f"MEAN COUNT: {mean_val:.2f}")
        print(f"STANDARD DEVIATION: {std_val:.2f}")
        print(f"25° PERCENTILE: {percentile:.2f}")

        print(f"\nFor {name} the relationships below the {percentile} threshold are:")
        for relation, count in counts.items():
            if count < percentile:
                print(f" {relation}: {count}")

        print("-" * 80)
        print("\n")

        # Plot relative frequencies and percentages
        relations = list(counts.keys())
        counts_list = list(counts.values())
        relative_frequencies = [count / tot for count in counts_list]
        percentages = [freq * 100 for freq in relative_frequencies]

        plt.figure(figsize=(16, 8))
        bars = plt.bar(relations, relative_frequencies)
        plt.xlabel("Relationship types")
        plt.ylabel("Relative frequency")
        plt.title(f"Relative frequencies of relationship types of {name}")
        plt.xticks(rotation=45, ha="right")

        for bar, percentage in zip(bars, percentages):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{percentage:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        output_file = os.path.join(output_folder, f'{name}_relationship_frequencies.png')
        plt.savefig(output_file)
        plt.close()

    # Plot global relative frequencies and percentages
    print("\n" + "=" * 30 + " GLOBAL RELATIONSHIP FREQUENCIES " + "=" * 30 + "\n")

    total_global_relationships = sum(global_relation_counts.values())
    global_relative_frequencies = {rel: count / total_global_relationships for rel, count in
                                   global_relation_counts.items()}

    sorted_relations = sorted(global_relative_frequencies.keys(), key=lambda x: global_relative_frequencies[x],
                              reverse=True)
    sorted_frequencies = [global_relative_frequencies[rel] for rel in sorted_relations]
    sorted_percentages = [freq * 100 for freq in sorted_frequencies]

    plt.figure(figsize=(16, 8))
    bars = plt.bar(sorted_relations, sorted_frequencies)
    plt.xlabel("Relationship types")
    plt.ylabel("Relative frequency")
    plt.title("Global relative frequencies of relationship types across all datasets")
    plt.xticks(rotation=45, ha="right")

    for bar, percentage in zip(bars, sorted_percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{percentage:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    output_file = os.path.join(output_folder, 'global_relationship_frequencies.png')
    plt.savefig(output_file)
    plt.close()

    print(f"\nGlobal relationship frequencies plot saved to: {output_file}")


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

    merged_datasets = {
        "TRAIN": "../../data/MERGED/train.json",
        "VAL": "../../data/MERGED/val.json",
        "TEST": "../../data/MERGED/test.json"
    }

    '''datasets = {}
    final_global_relations = set()
    for name, path in dataset_paths.items():
        datasets[name] = load_dataset(path)
        explore_features(datasets[name], name)

    compare_datasets(datasets)
    print(f"\n\n\n{final_global_relations}")
    print(len(final_global_relations))'''

    print("\n\n" + "=" * 60 + " MERGED DATASETS " + "=" * 60 + "\n")

    datasets = {}
    final_global_relations = set()
    for name, path in merged_datasets.items():
        datasets[name] = load_dataset(path)
        explore_features(datasets[name], name)

    compare_datasets(datasets)
    print(f"\n\n\n{final_global_relations}")
    print(len(final_global_relations))

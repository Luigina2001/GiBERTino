import os
from pathlib import Path

import numpy as np
from scipy import stats

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


def count_sentence_length(dataset_train, dataset_val, dataset_test, name):
    # Combine all datasets for processing
    all_datasets = [dataset_train, dataset_val, dataset_test]

    # Collect EDU lengths using list comprehension
    edus_len = [
        len(edu["text"])
        for dataset in all_datasets
        for entry in dataset
        for edu in entry["edus"]
    ]

    # Calculate statistics
    median = np.median(edus_len)
    mean = np.mean(edus_len)
    std = np.std(edus_len)
    max_pdf_value = stats.norm.pdf(mean, mean, std)  # Peak of normal curve

    print(f"Median: {median}")
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std}")

    # Create output directory
    output_folder = Path("sentences_length_plots")
    output_folder.mkdir(parents=True, exist_ok=True)

    # Initialize plot
    plt.figure(figsize=(12, 7))

    # Plot histogram
    plt.hist(edus_len, bins=30, density=True, alpha=0.6,
             color='b', edgecolor='black', label='Real distribution')

    # Plot normal distribution curve
    x = np.linspace(min(edus_len), max(edus_len), 1000)
    pdf = stats.norm.pdf(x, mean, std)
    plt.plot(x, pdf, 'r', linewidth=2, label='Normal distribution')

    # Highlight Gaussian curve peak
    plt.scatter(mean, max_pdf_value, color='red', zorder=5, s=80,
                label=f'Gaussian Peak: {max_pdf_value:.2f}')

    # Add vertical lines for statistics
    plt.axvline(float(mean), color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean: {mean:.1f}')
    plt.axvline(float(median), color='purple', linestyle='-.', linewidth=1.5, alpha=0.7, label=f'Median: {median:.1f}')

    # Add standard deviation boundaries
    plt.axvline(mean + std, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'+1σ: {mean + std:.1f}')
    plt.axvline(mean - std, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'-1σ: {mean - std:.1f}')

    # Configure plot appearance
    plt.title(f'Distribution of sentence lengths - {name}')
    plt.xlabel('Sentence lengths')
    plt.ylabel('Density')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.grid(True, alpha=0.3)

    # Save and close plot
    output_file = output_folder / f'{name}_sentence_length.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


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

    DATA_ROOT = Path("../../data")
    dataset_paths = {
        "TEST_133": DATA_ROOT / "MINECRAFT/TEST_133.json",
        "TEST_101_bert": DATA_ROOT / "MINECRAFT/TEST_101_bert.json",
        "DEV_32_bert": DATA_ROOT / "MINECRAFT/DEV_32_bert.json",
        "TRAIN_307_bert": DATA_ROOT / "MINECRAFT/TRAIN_307_bert.json",
        "VAL_100_bert": DATA_ROOT / "MINECRAFT/VAL_100_bert.json",
        "MOLWENI_dev": DATA_ROOT / "MOLWENI/dev.json",
        "MOLWENI_test": DATA_ROOT / "MOLWENI/test.json",
        "MOLWENI_train": DATA_ROOT / "MOLWENI/train.json",
        "STAC_test": DATA_ROOT / "STAC/test_subindex.json",
        "STAC_train": DATA_ROOT / "STAC/train_subindex.json",
    }

    MERGED_ROOT = DATA_ROOT / "MERGED"
    merged_datasets = {
        "TRAIN": MERGED_ROOT / "train.json",
        "VAL": MERGED_ROOT / "val.json",
        "TEST": MERGED_ROOT / "test.json",
    }

    '''
    datasets = {}
    final_global_relations = set()
    for name, path in dataset_paths.items():
        datasets[name] = load_dataset(path)
        explore_features(datasets[name], name)

    compare_datasets(datasets)
    print(f"\n\n\n{final_global_relations}")
    print(len(final_global_relations))'''

    train_data = load_dataset(str(merged_datasets["TRAIN"]))
    val_data = load_dataset(str(merged_datasets["VAL"]))
    test_data = load_dataset(str(merged_datasets["TEST"]))

    count_sentence_length(train_data, val_data, test_data, "sentence_len")

    '''
    print("\n\n" + "=" * 60 + " MERGED DATASETS " + "=" * 60 + "\n")

    datasets = {}
    final_global_relations = set()
    for name, path in merged_datasets.items():
        datasets[name] = load_dataset(path)
        # explore_features(datasets[name], name)

    # compare_datasets(datasets)
    # print(f"\n\n\n{final_global_relations}")
    # print(len(final_global_relations))'''

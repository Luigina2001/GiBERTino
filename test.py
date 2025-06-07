import argparse
import os
import re

import lightning
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from dataset.dialogue_graph_datamodule import SubDialogueDataModule
from model.GiBERTino import GiBERTino
from utils.constants import RELATIONS
from utils.metrics import Metrics
from utils.utils import get_device


def test_model(args):  # noqa
    lightning.seed_everything(args.seed)
    os.makedirs(args.eval_dir, exist_ok=True)

    device = get_device()
    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        model_config = checkpoint.get("hyper_parameters", {})
        in_channels = model_config.get("in_channels", 770)
        hidden_channels = model_config.get("hidden_channels", 10)
        num_layers = model_config.get("num_layers", 3)

        print(f"Model parameters loaded from checkpoint: "
              f"in_channels={in_channels}, hidden_channels={hidden_channels}, num_layers={num_layers}")

        model = GiBERTino.load_from_checkpoint(args.checkpoint_path, map_location=device)
        model.eval()

        print("Loading test data...")
        data_module = SubDialogueDataModule(args.data_path, num_workers=0)
        data_module.setup(stage="test")
        test_loader = data_module.test_dataloader()

        metrics = Metrics(num_classes=len(RELATIONS['UNIFIED']), log_dir=args.eval_dir)
        all_metrics = []

        all_rel_preds = []
        all_rel_labels = []

        print("Running classification...")
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Processing test data"):
                link_logits, rel_probs = model(batch)

                link_labels = batch[('edu', 'to', 'edu')].get('link_labels', None)
                rel_labels = batch[('edu', 'to', 'edu')].get('rel_labels', None)

                link_metrics = metrics.compute_metrics(link_logits, link_labels, 'link', 'test', 0)
                rel_metrics = metrics.compute_metrics(rel_probs, rel_labels, 'rel', 'test', 0)

                batch_metrics = {
                    'link_accuracy': link_metrics['link_accuracy'],
                    'link_precision': link_metrics['link_precision'],
                    'link_recall': link_metrics['link_recall'],
                    'link_f1': link_metrics['link_f1'],
                    'link_roc': link_metrics['link_roc'],
                    'rel_accuracy': rel_metrics['rel_accuracy'],
                    'rel_precision': rel_metrics['rel_precision'],
                    'rel_recall': rel_metrics['rel_recall'],
                    'rel_f1': rel_metrics['rel_f1'],
                    'rel_roc': rel_metrics['rel_roc']
                }
                metrics.log(batch_metrics, 'test', 0)
                all_metrics.append(batch_metrics)

                if rel_labels is not None:
                    preds = rel_probs.argmax(dim=-1).cpu().numpy()
                    labels = rel_labels.cpu().numpy()

                    all_rel_preds.extend(preds.flatten())
                    all_rel_labels.extend(labels.flatten())

        aggregated_metrics = metrics.aggregate_metrics()

        print("Test results:")
        for key, value in aggregated_metrics.items():
            print(f"{key}: {value:.4f}")

        print("\nAggregate metrics:")
        print(f"Link Prediction Accuracy: {aggregated_metrics['link_accuracy']:.4f}")
        print(f"Relation Classification Accuracy: {aggregated_metrics['rel_accuracy']:.4f}")

        eval_path = os.path.join(args.eval_dir, args.eval_file)
        header = "Model;Dataset;" + ";".join(aggregated_metrics.keys()) + "\n"

        if os.path.exists(eval_path):
            with open(eval_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
        else:
            first_line = ""

        with open(eval_path, "a", encoding="utf-8") as f:
            if first_line != header.strip():
                f.write(header)

            f.write(
                f"{args.model_name};{args.dataset_name};" +
                ";".join(f"{c:.5f}" for c in aggregated_metrics.values()) + "\n"
            )

        print(f"Metrics saved to {eval_path}")

        # Confusion matrix for relation prediction
        if all_rel_preds and all_rel_labels:
            cm = confusion_matrix(all_rel_labels, all_rel_preds, labels=range(len(RELATIONS['UNIFIED'])))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=RELATIONS['UNIFIED'])

            fig, ax = plt.subplots(figsize=(12, 10))
            disp.plot(ax=ax, xticks_rotation='vertical', cmap='Blues', values_format='d')

            cm_path = os.path.join(args.eval_dir, f"confusion_matrix_{args.model_name}_{args.dataset_name}.png")
            plt.title(f"Relation Confusion Matrix: {args.model_name} on {args.dataset_name}")
            plt.savefig(cm_path, bbox_inches='tight')
            plt.close()

            print(f"Confusion matrix saved to {cm_path}")

        return aggregated_metrics

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def find_checkpoints(root_dir):
    """
    Recursively finds .ckpt files in a 'checkpoints' subdirectory.
    Returns a list of tuples: (checkpoint_path, model_name, dataset_name)
    Expected filename format: giBERTino-<model_name>-<dataset_name>-epoch=xx-val_loss=xx.ckpt
    """
    pattern = re.compile(r"giBERTino-(.+)-(.+?)-epoch=\d+-val_loss=[\d\.]+\.ckpt$")
    found = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "checkpoints" in dirpath:
            for filename in filenames:
                if filename.endswith(".ckpt"):
                    match = pattern.match(filename)
                    if match:
                        model_name, dataset_name = match.groups()
                        full_path = os.path.join(dirpath, filename)
                        found.append((full_path, model_name + "-" + dataset_name, dataset_name))
    return found


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_dir", type=str, default=os.path.join(os.getcwd(), "eval_results"))
    parser.add_argument("--eval_file", type=str, default="scores.csv")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory to search for checkpoints")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory where datasets are stored")

    args = parser.parse_args()
    checkpoint_infos = find_checkpoints(args.root_dir)

    datasets = ["STAC", "MOLWENI", "MINECRAFT", "BALANCED"]

    if not checkpoint_infos:
        print("No valid checkpoints found.")
    else:
        for ckpt_path, model_name, _ in checkpoint_infos:  # Ignore dataset_name from filename
            for dataset_name in datasets:
                dataset_path = os.path.join(args.data_root, dataset_name, "alibaba-graphs")

                if not os.path.exists(dataset_path):
                    print(f"Dataset path does not exist for {dataset_name}: {dataset_path}")
                    continue

                print(f"\n--- Evaluating {ckpt_path} on {dataset_name} ---")
                test_args = argparse.Namespace(
                    seed=args.seed,
                    eval_dir=args.eval_dir,
                    eval_file=args.eval_file,
                    checkpoint_path=ckpt_path,
                    data_path=dataset_path,
                    model_name=model_name,
                    dataset_name=dataset_name
                )
                test_model(test_args)


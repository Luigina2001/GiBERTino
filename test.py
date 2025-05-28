import argparse
import os

import lightning
import torch
from tqdm import tqdm

from dataset.dialogue_graph_datamodule import SubDialogueDataModule
from model.GiBERTino import GiBERTino
from utils.constants import RELATIONS
from utils.metrics import Metrics
from utils.utils import get_device


def test_model(args):  # noqa
    """
    Load the model from the checkpoint, perform classification on test data, and compute evaluation metrics.
    """
    lightning.seed_everything(args.seed)
    os.makedirs(args.eval_dir, exist_ok=True)

    device = get_device()
    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        # Extract model parameters from the checkpoint
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

        # Aggregate metrics over all batches
        aggregated_metrics = metrics.aggregate_metrics()

        print("Test results:")
        for key, value in aggregated_metrics.items():
            print(f"{key}: {value:.4f}")

        print("\nAggregate metrics:")
        print(f"Link Prediction Accuracy: {aggregated_metrics['link_accuracy']:.4f}")
        print(f"Relation Classification Accuracy: {aggregated_metrics['rel_accuracy']:.4f}")

        eval_path = os.path.join(args.eval_dir, args.eval_file)

        header = "Model;Dataset;" + ";".join(aggregated_metrics.keys()) + "\n"

        # Check if the header exists
        if os.path.exists(eval_path):
            with open(eval_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
        else:
            first_line = ""

        # Open file in append mode and write data
        with open(eval_path, "a", encoding="utf-8") as f:
            if first_line != header.strip():  # Avoid duplicate headers
                f.write(header)

            f.write(
                f"{args.model_name};{args.dataset_name};" +
                ";".join(f"{c:.5f}" for c in aggregated_metrics.values()) + "\n"
            )

        print(f"Metrics saved to {eval_path}")

        return aggregated_metrics

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_dir", type=str, default=os.path.join(os.getcwd(), "eval_results"))
    parser.add_argument("--eval_file", type=str, default="scores.csv")
    parser.add_argument("--checkpoint_path", type=str, help="Path of the model to evaluate", required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)

    test_model(parser.parse_args())

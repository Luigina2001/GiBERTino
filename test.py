import os
import torch
import json
from dataset.dialogue_graph_datamodule import SubDialogueDataModule
from model.GiBERTino import GiBERTino
from utils.metrics import Metrics
from utils.utils import get_device
import lightning

def test_model(checkpoint_path: str, data_path: str, metrics_output_path: str = "test_metrics.json"): # noqa
    """
    Load the model from the checkpoint, perform classification on test data, and compute evaluation metrics.
    """
    device = get_device()
    print(f"Loading model from checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # print(checkpoint.keys())
        # print(checkpoint["hyper_parameters"])
        state_dict = checkpoint["state_dict"]  # Extract only the model weights

        new_state_dict = {}

        # TODO: to remove
        for key, value in state_dict.items():
            if "link_predictor" in key:
                new_key = key.replace("link_predictor", "link_classifier")
            elif "relation_predictor" in key:
                new_key = key.replace("relation_predictor", "rel_classifier")
            else:
                new_key = key
            new_state_dict[new_key] = value

        # Extract model parameters from the checkpoint
        model_config = checkpoint.get("hyper_parameters", {})
        in_channels = model_config.get("in_channels", 2304)
        hidden_channels = model_config.get("hidden_channels", 10)
        num_layers = model_config.get("num_layers", 3)

        print(f"Model parameters loaded from checkpoint: "
              f"in_channels={in_channels}, hidden_channels={hidden_channels}, num_layers={num_layers}")

        model = GiBERTino.load_from_checkpoint(checkpoint_path, map_location=device)
        model.to(device)
        model.eval()

        print("Loading test data...")
        data_module = SubDialogueDataModule(data_path, num_workers=0)
        data_module.setup(stage="test")
        test_loader = data_module.test_dataloader()

        metrics = Metrics()
        all_metrics = []

        print("Running classification...")
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()} if isinstance(batch, dict) else batch.to(device)

                link_logits, rel_probs = model(batch)

                # print(f"Batch_keys: {batch.keys()}")
                # print(f"Batch[('edu', 'to', 'edu')]: {batch[('edu', 'to', 'edu')]}")

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

        # Save metrics to JSON file
        results = {
            "aggregated_metrics": aggregated_metrics,
            "batch_metrics": all_metrics
        }

        os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
        with open(metrics_output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4) # noqa
        print(f"Metrics saved to {metrics_output_path}")

        return aggregated_metrics

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    lightning.seed_everything(42)
    checkpoint_path = "lightning_logs/version_1/checkpoints/alibaba-modernbert-stac-epoch-epoch=18.ckpt"
    data_path = "./data/STAC/graphs/"
    test_model(checkpoint_path, data_path)

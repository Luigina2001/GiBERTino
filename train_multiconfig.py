import os
from typing import Dict


def run_experiment(dataset: str, graph_type: str, params: Dict):
    log_steps = {"STAC": 5, "MOLWENI": 15, "MINECRAFT": 50, "BALANCED": 50}[dataset]

    experiment_name = f"giBERTino-{graph_type}-{params['gnn_model']}-{dataset}"
    data_root = f"data/{dataset}/{graph_type}-graphs"

    checkpoint_callback = (
        f"\"[{{'class_path': 'lightning.pytorch.callbacks.ModelCheckpoint', "
        f"'init_args': {{'filename': '{experiment_name}-epoch-{{{{epoch:02d}}}}-val_loss={{{{val_loss:.2f}}}}'}}}}]\""
    )

    cmd = (
        f"python train.py fit "
        f"--config config.yaml "
        f"--trainer.logger.init_args.name={experiment_name} "
        f"--trainer.callbacks+={checkpoint_callback} "
        f"--model.relations={dataset} "
        f"--data.root={data_root} "
        f"--data.dataset_name={dataset} "
        f"--trainer.log_every_n_steps={log_steps} "
        f"--model.gnn_model={params['gnn_model']} "
        f"--model.hidden_channels={params['hidden_channels']} "
        f"--model.num_layers={params['num_layers']} "
        f"--model.lr={params['lr']} "
        f"--data.batch_size={params['batch_size']} "
        f"--trainer.default_root_dir ./lightning_logs/{experiment_name} "
        f"--trainer.logger.init_args.project=giBERTino "
        f"--trainer.logger.init_args.save_dir=lightning_logs"
    )

    print(f"\n{'#' * 50}")
    print(f"Starting {experiment_name}")
    print("Command:", cmd)

    result = os.system(cmd)
    return result == 0


def main():
    configurations = {
        "GCN": {"hidden_channels": 128, "num_layers": 64, "lr": 1e-5, "batch_size": 32},
        "GraphSage": {
            "hidden_channels": 512,
            "num_layers": 32,
            "lr": 1e-5,
            "batch_size": 32,
        },
    }

    datasets = ["STAC", "MOLWENI", "MINECRAFT", "BALANCED"]
    graph_types = ["alibaba", "dialogpt"]

    for model_name, params in configurations.items():
        params["gnn_model"] = model_name
        for dataset in datasets:
            for graph_type in graph_types:
                success = run_experiment(dataset, graph_type, params)
                status = "SUCCESS" if success else "FAILED"
                print(f"\n{status} - {model_name} - {dataset} - {graph_type}\n")


if __name__ == "__main__":
    main()

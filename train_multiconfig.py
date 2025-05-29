from typing import Dict

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from dataset import SubDialogueDataModule
from model import GiBERTino


def run_experiment(dataset: str, graph_type: str, params: Dict):
    log_steps = {"STAC": 5, "MOLWENI": 15, "MINECRAFT": 50, "BALANCED": 50}[dataset]

    experiment_name = f"giBERTino-{graph_type}-{params['gnn_model']}-{dataset}"
    data_root = f"data/{dataset}/{graph_type}-graphs"
    log_dir = f"./lightning_logs/{experiment_name}"

    print(f"\n{'#' * 50}")
    print(f"Starting {experiment_name}")

    data_module = SubDialogueDataModule(
        root=data_root,
        batch_size=params["batch_size"],
        num_workers=0,
    )

    model = GiBERTino(
        in_channels=770,
        gnn_model=params["gnn_model"],
        hidden_channels=params["hidden_channels"],
        num_layers=params["num_layers"],
        lr=params["lr"],
        dataset_name=dataset, # noqa
    )

    logger = WandbLogger(
        name=experiment_name,
        save_dir="lightning_logs",
        project="giBERTino",
        log_model=False,
        reinit=True,
    )

    # Initialize callbacks
    early_stop = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0,
        patience=10,
        verbose=True,
        mode="min",
        strict=True,
        check_finite=True,
        log_rank_zero_only=False,
    )

    checkpoint_callback = ModelCheckpoint(
        filename=f"{experiment_name}-{{epoch:02d}}-{{val_loss:.2f}}",
        monitor="val_loss",
        save_top_k=1,
        save_weights_only=False,
        mode="min",
        auto_insert_metric_name=True,
        save_on_train_epoch_end=True,
        enable_version_counter=True,
    )

    trainer = Trainer(
        accelerator="auto",
        max_epochs=30,
        logger=logger,
        callbacks=[early_stop, checkpoint_callback],
        log_every_n_steps=log_steps,
        default_root_dir=log_dir,
    )

    try:
        trainer.fit(model, datamodule=data_module)
        return True
    except Exception as e:
        print(f"Training failed: {e}")
        return False


def main():
    configurations = {
        "GCN": {"hidden_channels": 128, "num_layers": 64, "lr": 1e-5, "batch_size": 32},
        "GraphSAGE": {
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

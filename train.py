from lightning.pytorch.cli import LightningCLI

from model import GiBERTino
from dataset.dialogue_graph_datamodule import SubDialogueDataModule


def cli_main():
    LightningCLI(model_class=GiBERTino, datamodule_class=SubDialogueDataModule,
                 parser_kwargs={"fit": {"default_config_files": ["config.yaml"]}})


if __name__ == '__main__':
    cli_main()

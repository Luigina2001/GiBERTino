from os import environ
from lightning.pytorch.cli import LightningCLI

from model import GiBERTino
from dataset.dialogue_graph_datamodule import SubDialogueDataModule


def cli_main():
    environ['TOKENIZERS_PARALLELISM'] = 'false'
    environ['`PYTORCH_ENABLE_MPS_FALLBACK`'] = '1' # fallback for aten::scatter_reduce.two_out from MPS to CPU
    LightningCLI(model_class=GiBERTino, datamodule_class=SubDialogueDataModule,
                 parser_kwargs={"fit": {"default_config_files": ["config.yaml"]}},
                 save_config_kwargs={"overwrite": True})


if __name__ == '__main__':
    cli_main()

# flake8: noqa: E402

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.cli import LightningCLI

from src import data, models, utils

OmegaConf.register_new_resolver("eval", eval)

log = utils.get_pylogger(__name__)


def cli_main():
    cli = LightningCLI(
        model_class=pl.LightningModule,
        subclass_mode_model=True,
        datamodule_class=pl.LightningDataModule,
        subclass_mode_data=True,
        parser_kwargs={"parser_mode": "omegaconf"},  # to enable usage of environment variables in the config files
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    cli_main()

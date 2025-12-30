"""Command-line interface."""

import logging

from lightning.pytorch import callbacks as pytorch_callbacks, cli
from yoyodyne import trainers

from . import callbacks, data, models


class YoyodynePretrainedCLI(cli.LightningCLI):
    """The Yoyodyne Pretrained CLI interface.

    Use with `--help` to see the full list of options.
    """

    def add_arguments_to_parser(
        self, parser: cli.LightningArgumentParser
    ) -> None:
        parser.add_lightning_class_args(
            pytorch_callbacks.ModelCheckpoint,
            "checkpoint",
            required=False,
        )
        parser.add_lightning_class_args(
            callbacks.PredictionWriter,
            "prediction",
            required=False,
        )
        parser.link_arguments("model.init_args.model_name", "data.model_name")
        parser.link_arguments(
            "data.model_dir",
            "trainer.logger.init_args.save_dir",
            apply_on="instantiate",
        )


def main() -> None:
    logging.basicConfig(
        format="%(levelname)s: %(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level="INFO",
    )
    # Select the model.
    YoyodynePretrainedCLI(
        models.BaseModel,
        data.DataModule,
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_callback=None,
        subclass_mode_model=True,
        trainer_class=trainers.Trainer,
    )


def python_interface(args: cli.ArgsType = None) -> None:
    """Interface to use models through Python."""
    YoyodynePretrainedCLI(
        models.BaseModel,
        data.DataModule,
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_callback=None,
        subclass_mode_model=True,
        trainer_class=trainers.Trainer,
        args=args,
    )

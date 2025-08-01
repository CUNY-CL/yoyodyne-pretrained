"""Command-line interface."""

import logging

from lightning.pytorch import callbacks as pytorch_callbacks, cli

from .. import callbacks, data, models, trainers


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


def main() -> None:
    logging.basicConfig(
        format="%(filename)s %(levelname)s: %(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level="INFO",
    )
    # Select the model.
    YoyodynePretrainedCLI(
        model_class=models.BaseModel,
        datamodule_class=data.DataModule,
        subclass_mode_model=True,
        # Prevents predictions from accumulating in memory; see the
        # documentation in `trainers.py` for more context.
        trainer_class=trainers.Trainer,
    )


def python_interface(args: cli.ArgsType = None) -> None:
    """Interface to use models through Python."""
    YoyodynePretrainedCLI(
        models.BaseModel,
        data.DataModule,
        subclass_mode_model=True,
        # Prevents predictions from accumulating in memory; see the
        # documentation in `trainers.py` for more context.
        trainer_class=trainers.Trainer,
        args=args,
    )

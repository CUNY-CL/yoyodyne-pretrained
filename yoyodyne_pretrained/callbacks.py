"""Custom callbacks."""

from typing import Sequence, TextIO

import lightning
from lightning.pytorch import callbacks, trainer
import torch
import transformers

from . import data, models


class PredictionWriter(callbacks.BasePredictionWriter):
    """Writes predictions.

    Args:
        path: Path for the predictions file.
    """

    path: str
    sink: TextIO | None
    tokenizer: transformers.AutoTokenizer | None

    # path is given a default argument to silence a warning if no prediction
    # callback is configured.
    # TODO: remove default if this is addressed:
    #
    #   https://github.com/Lightning-AI/pytorch-lightning/issues/20851
    def __init__(self, path: str = ""):
        super().__init__("batch")
        self.path = path
        self.sink = None
        self.tokenizer = None

    # Required API.

    def on_predict_start(
        self, trainer: trainer.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        # Placing this here prevents the creation of an empty file in the case
        # where a prediction callback was specified but this is not running
        # in predict mode.
        self.sink = open(self.path, "w")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pl_module.hparams.model_name
        )

    def write_on_batch_end(
        self,
        trainer: trainer.Trainer,
        model: models.BaseModel,
        predictions: torch.Tensor,
        batch_indices: Sequence[int] | None,
        batch: data.Batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        for prediction in self.tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True,
        ):
            print(prediction, file=self.sink)
        self.sink.flush()

    def on_predict_end(
        self, trainer: trainer.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        self.sink.close()

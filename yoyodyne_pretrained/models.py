"""Models."""

import lightning
from lightning.pytorch import cli
import tokenizers
import torch
from torch import nn, optim
import transformers

from . import data, defaults, pretrained

# FIXME do I need pretrained?
# FIXME do I need defaults?


class YoyodynePretrained(lightning.LightningModule):
    """Yoyodyne pretrained model."""

    model: transformers.EncoderDecoderModel
    loss_func: nn.CrossEntropyLoss
    # FIXME metrics
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler.LRScheduler

    def __init__(
        self,
        encoder=defaults.ENCODER,
        decoder=defaults.DECODER,
        optimizer: cli.OptimizerCallable = defaults.OPTIMIZER,
        scheduler: cli.LRSchedulerCallable = defaults.SCHEDULER,
    ):
        super().__init__()
        self.model = (
            transformers.EncoderDecoderModel.from_encoder_decoder_pretrained(
                encoder, decoder
            )
        )
        self.loss_func = nn.CrossEntropyLoss()  # FIXME label smoothing etc.
        self.optimizer = optimizer
        self.scheduler = scheduler

    def configure_optimizers(self) -> dict:
        optimizer = self.optimizer(self.model.parameters())
        scheduler = optimizer(scheduler)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(*args, **kwargs):
        return self.model(*args, **kwargs)

    def predict_step(self, batch: data.Batch, batch_idx: int) -> torch.Tensor:
        return self(batch)

    def training_step(self, batch: data.Batch, batch_idx: int) -> torch.Tensor:
        predictions = self(batch)
        return self.loss_func(predictions, batch.target.input_ids)

    def test_step(self, batch: data.Batch, batch_idx: int) -> None:
        pass
        # predictions = torch.argmax(self(batch), dim=1)
        # FIXME metrics.

    def validation_step(self, batch: data.Batch, batch_idx: int) -> None:
        predictions = self(batch)
        loss = self.loss_func(predictions, batch.target.input_ids)
        self.log(
            "val_loss",
            loss,
            batch_size=len(batch),
            logger=True,
            on_epoch=True,
            prog_bar=True,
        )
        # FIXME metrics.

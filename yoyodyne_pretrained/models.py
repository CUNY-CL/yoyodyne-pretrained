"""Models."""

import lightning
from lightning.pytorch import cli
import torch
from torch import nn, optim
import transformers

from . import data, defaults, metrics


class YoyodynePretrained(lightning.LightningModule):
    """Yoyodyne pretrained model.

    This model consists of a pretrained encoder and decoder with randomly
    initialized cross-attention.

    After:
        Rothe, S., Narayan, S., and Severyn, A. 2020. Leveraging pre-trained
        checkpoints for sequence generation tasks. Transactions of the
        Association for Computational Linguistics 8: 264-280.

    * The forward method returns a tensor of shape B x vocab_size x seq_length
      for compatibility with loss and evaluation functions.
    * Cross-entropy loss is the loss function.
    * One or more predictions tensor(s) are returned by predict_step.
    * Loss is returned by training_step.
    * Evaluation metrics are tracked by test_step; nothing is returned.
    * Validation loss and evaluation metrics are tracked by validation_step;
      nothing is returned.

    Args:
        dropout: Dropout probability.
        encoder: Name of the Hugging Face encoder model.
        decoder: Name of the Hugging Face decoder model.
    """

    model: transformers.EncoderDecoderModel
    loss_func: nn.CrossEntropyLoss
    # TODO: update with new metrics as they become available.
    accuracy: metrics.Accuracy | None
    ser: metrics.SER | None
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler.LRScheduler

    def __init__(
        self,
        dropout=defaults.DROPOUT,
        encoder=defaults.ENCODER,
        decoder=defaults.DECODER,
        compute_accuracy=True,
        compute_ser=False,
        optimizer: cli.OptimizerCallable = defaults.OPTIMIZER,
        scheduler: cli.LRSchedulerCallable = defaults.SCHEDULER,
    ):
        super().__init__()
        self.model = (
            transformers.EncoderDecoderModel.from_encoder_decoder_pretrained(
                encoder,
                decoder,
                encoder_hidden_dropout_prob=dropout,
                decoder_hidden_dropout_prob=dropout,
            )
        )
        self.loss_func = nn.CrossEntropyLoss(ignore_index=0)
        self.accuracy = (
            metrics.Accuracy(
                num_classes=self.model.get_output_embeddings().out_features
            )
            if compute_accuracy
            else None
        )
        self.ser = metrics.SER() if compute_ser else None
        self.optimizer = optimizer
        self.scheduler = scheduler

    def configure_optimizers(self) -> dict:
        optimizer = self.optimizer(self.model.parameters())
        scheduler = self.scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # TODO: update with new metrics as they become available.

    @property
    def has_accuracy(self) -> bool:
        return self.accuracy is not None

    @property
    def has_ser(self) -> bool:
        return self.ser is not None

    def forward(self, batch: data.Batch) -> torch.Tensor:
        output = self.model(
            input_ids=batch.source,
            attention_mask=batch.source_mask,
            decoder_input_ids=batch.target,
            decoder_attention_mask=batch.target_mask,
        )
        # -> B x vocab_size x seq_length.
        return output.logits.transpose(1, 2)

    def predict_step(self, batch: data.Batch, batch_idx: int) -> torch.Tensor:
        logits = self(batch)
        return torch.argmax(logits, dim=1)

    def training_step(self, batch: data.Batch, batch_idx: int) -> torch.Tensor:
        logits = self(batch)
        return self.loss_func(logits, batch.target)

    def on_test_epoch_start(self) -> None:
        self._reset_metrics()

    def test_step(self, batch: data.Batch, batch_idx: int) -> None:
        logits = self(batch)
        self._update_metrics(logits, batch.target)

    def on_test_epoch_end(self) -> None:
        self._log_metrics_on_epoch_end("test")

    def on_validation_epoch_start(self) -> None:
        self._reset_metrics()

    def validation_step(self, batch: data.Batch, batch_idx: int) -> None:
        logits = self(batch)
        loss = self.loss_func(logits, batch.target)
        self.log(
            "val_loss",
            loss,
            batch_size=len(batch),
            logger=True,
            on_epoch=True,
            prog_bar=True,
        )
        self._update_metrics(logits, batch.target)

    def on_validation_epoch_end(self) -> None:
        self._log_metrics_on_epoch_end("val")

    def _reset_metrics(self) -> None:
        # TODO: update with new metrics as they become available.
        if self.has_accuracy:
            self.accuracy.reset()
        if self.has_ser:
            self.ser.reset()

    def _update_metrics(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> None:
        # TODO: update with new metrics as they become available.
        if self.has_accuracy:
            self.accuracy.update(logits, target)
        if self.has_ser:
            self.ser.update(logits, target)

    def _log_metrics_on_epoch_end(self, subset: str) -> None:
        # TODO: update with new metrics as they become available.
        if self.has_accuracy:
            self.log(
                f"{subset}_accuracy",
                self.accuracy.compute(),
                logger=True,
                on_epoch=True,
                prog_bar=True,
            )
        if self.has_ser:
            self.log(
                f"{subset}_ser",
                self.ser.compute(),
                logger=True,
                on_epoch=True,
                prog_bar=True,
            )

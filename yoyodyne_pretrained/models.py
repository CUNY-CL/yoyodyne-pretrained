"""Models."""

import lightning
from lightning.pytorch import cli
import torch
from torch import nn, optim
import transformers

from . import data, defaults, metrics


class PretrainedModel(lightning.LightningModule):
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
        label_smoothing: Label smoothing probability.
        encoder: Name of the Hugging Face encoder model.
        decoder: Name of the Hugging Face decoder model.
        tie_encoder_decoder: Should we tie the two?
    """

    model: transformers.EncoderDecoderModel
    loss_func: nn.CrossEntropyLoss
    # TODO: update with new metrics as they become available.
    accuracy: metrics.Accuracy | None
    ser: metrics.SER | None
    generation_config: transformers.GenerationConfig
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler.LRScheduler

    def __init__(
        self,
        dropout=defaults.DROPOUT,
        label_smoothing=defaults.LABEL_SMOOTHING,
        encoder=defaults.ENCODER,
        decoder=defaults.DECODER,
        tie_encoder_decoder=defaults.TIE_ENCODER_DECODER,
        compute_accuracy=True,
        compute_ser=False,
        optimizer: cli.OptimizerCallable = defaults.OPTIMIZER,
        scheduler: cli.LRSchedulerCallable = defaults.SCHEDULER,
    ):
        super().__init__()
        # Needed for various attributes.
        self.model = (
            transformers.EncoderDecoderModel.from_encoder_decoder_pretrained(
                encoder,
                decoder,
                tie_encoder_decoder=tie_encoder_decoder,
                #encoder_hidden_dropout_prob=dropout,
                #decoder_hidden_dropout_prob=dropout,
            )
        )
        # Necessary patching for decoding.
        decoder_tokenizer = transformers.AutoTokenizer.from_pretrained(decoder)
        self.bos = decoder_tokenizer.cls_token_id
        self.eos = decoder_tokenizer.sep_token_id
        self.pad = decoder_tokenizer.pad_token_id
        self.model.config.decoder_start_token_id = self.bos
        self.model.config.bos_token_id = self.bos
        self.model.config.eos_token_id = self.eos
        self.model.config.pad_token_id = self.pad
        target_vocab_size = (
            self.model.get_decoder().get_output_embeddings().out_features
        )
        self.accuracy = (
            metrics.Accuracy(
                ignore_index=self.pad, num_classes=target_vocab_size
            )
            if compute_accuracy
            else None
        )
        self.ser = metrics.SER(self.eos) if compute_ser else None
        # Actually initialized by configure_optimizers.
        self.optimizer = optimizer
        self.scheduler = scheduler

    def configure_optimizers(self) -> dict[str, ...]:
        optimizer = self.optimizer(self.model.parameters())
        scheduler = self.scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                **scheduler.config_dict(),
            },
        }

    # TODO: update with new metrics as they become available.

    @property
    def has_accuracy(self) -> bool:
        return self.accuracy is not None

    @property
    def has_ser(self) -> bool:
        return self.ser is not None

    def forward(self, batch: data.Batch) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model(
            input_ids=batch.source,
            attention_mask=batch.source_mask,
            labels=batch.target,
        )
        # -> B x vocab_size x seq_length.
        logits = output.logits.transpose(1, 2)
        return logits, output.loss

    def predict_step(self, batch: data.Batch, batch_idx: int) -> torch.Tensor:
        # FIXME
        return self._decode(batch.source, batch.source_mask)

    def training_step(self, batch: data.Batch, batch_idx: int) -> torch.Tensor:
        _, loss = self(batch)
        return loss

    def on_test_epoch_start(self) -> None:
        self._reset_metrics()

    def test_step(self, batch: data.Batch, batch_idx: int) -> None:
        pass
        # FIXME
        # self._update_metrics(predictions, batch.target)

    def on_test_epoch_end(self) -> None:
        self._log_metrics_on_epoch_end("test")

    def on_validation_epoch_start(self) -> None:
        self._reset_metrics()

    def validation_step(self, batch: data.Batch, batch_idx: int) -> None:
        _, loss = self(batch)
        self.log(
            "val_loss",
            loss,
            batch_size=len(batch),
            logger=True,
            on_epoch=True,
            prog_bar=True,
        )
        predictions = self.model.generate(
            batch.target,
            generation_config=transformers.GenerationConfig(
                bos_token_id=self.bos, 
                eos_token_id=self.eos,
                pad_token_id=self.pad,
                early_stopping=True,
                no_repeat_ngram_size=2,
                num_beams=5
            ),
        )
        decoder_tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model.get_decoder().name_or_path
        )
        print(
            decoder_tokenizer.batch_decode(
                predictions, skip_special_tokens=True
            )
        )
        print()
        # self._update_metrics(predictions, batch.target)

    def on_validation_epoch_end(self) -> None:
        self._log_metrics_on_epoch_end("val")

    def _reset_metrics(self) -> None:
        # TODO: update with new metrics as they become available.
        if self.has_accuracy:
            self.accuracy.reset()
        if self.has_ser:
            self.ser.reset()

    def _update_metrics(
        self, predictions: torch.Tensor, target: torch.Tensor
    ) -> None:
        # TODO: update with new metrics as they become available.
        # TODO: this design assumes that at least one metric is enabled and
        # does an unnecessary decode if all are disabled. Should we do
        # anything to avoid this?
        if self.has_accuracy:
            self.accuracy.update(predictions, target)
        if self.has_ser:
            self.ser.update(predictions, target)

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

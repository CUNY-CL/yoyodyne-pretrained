"""Models."""

import lightning
from lightning.pytorch import cli
import torch
from torch import optim
import transformers

from . import data, defaults, metrics


class BaseModel(lightning.LightningModule):
    """Model base class.

    * The forward method returns a loss tensor.
    * Cross-entropy loss is the loss function.
    * One or more predictions tensor(s) are returned by predict_step.
    * Loss is returned by training_step.
    * Evaluation metrics are tracked by test_step; nothing is returned.
    * Validation loss and evaluation metrics are tracked by validation_step;
      nothing is returned.
    """

    # TODO: update with new metrics as they become available.
    accuracy: metrics.Accuracy | None
    generation_config: transformers.GenerationConfig
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler.LRScheduler

    def configure_optimizers(
        self,
    ) -> tuple[list[optim.Optimizer], list[optim.lr_scheduler.LRScheduler]]:
        optimizer = self.optimizer(self.model.parameters())
        scheduler = self.scheduler(optimizer)
        return [optimizer], [scheduler]

    # TODO: update with new metrics as they become available.

    @property
    def has_accuracy(self) -> bool:
        return self.accuracy is not None

    def forward(self, batch: data.Batch) -> torch.Tensor:
        return self.model(
            batch.source, attention_mask=batch.source_mask, labels=batch.target
        ).loss

    def predict_step(self, batch: data.Batch, batch_idx: int) -> torch.Tensor:
        return self._decode(batch.source, batch.source_mask)

    def training_step(self, batch: data.Batch, batch_idx: int) -> torch.Tensor:
        loss = self(batch)
        self.log(
            "train_loss",
            loss,
            batch_size=len(batch),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        return loss

    def on_test_epoch_start(self) -> None:
        self._reset_metrics()

    def test_step(self, batch: data.Batch, batch_idx: int) -> None:
        self._update_metrics(batch)

    def on_test_epoch_end(self) -> None:
        self._log_metrics_on_epoch_end("test")

    def on_validation_epoch_start(self) -> None:
        self._reset_metrics()

    def validation_step(self, batch: data.Batch, batch_idx: int) -> None:
        loss = self(batch)
        self.log(
            "val_loss",
            loss,
            batch_size=len(batch),
            logger=True,
            on_epoch=True,
            prog_bar=True,
        )
        self._update_metrics(batch)

    def on_validation_epoch_end(self) -> None:
        self._log_metrics_on_epoch_end("val")

    def _reset_metrics(self) -> None:
        # TODO: update with new metrics as they become available.
        if self.has_accuracy:
            self.accuracy.reset()

    def _update_metrics(self, batch: data.Batch) -> None:
        # TODO: update with new metrics as they become available.
        if self.has_accuracy:
            predictions = self._decode(batch.source, batch.source_mask)
            self.accuracy.update(predictions, batch.target)

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

    def _decode(
        self, source: torch.Tensor, source_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.model.generate(
            source,
            attention_mask=source_mask,
            generation_config=self.generation_config,
        )


class EncoderDecoderModel(BaseModel):
    """Pretrained encoder/decoder.

    This model consists of a pretrained encoder and decoder with randomly
    initialized cross-attention.

    After:
        Rothe, S., Narayan, S., and Severyn, A. 2020. Leveraging pre-trained
        checkpoints for sequence generation tasks. Transactions of the
        Association for Computational Linguistics 8: 264-280.

    Args:
        encoder: Name of the Hugging Face encoder model.
        decoder: Name of the Hugging Face decoder model.
        dropout: Dropout probability.
        label_smoothing: Label smoothing probability.
        num_beams: Width of the beam to use during decoding.
    """

    model: transformers.EncoderDecoderModel

    def __init__(
        self,
        *,
        model_name="google-bert/bert-base-multilingual-cased",
        dropout=defaults.DROPOUT,
        label_smoothing=defaults.LABEL_SMOOTHING,
        tie_encoder_decoder=True,
        num_beams=defaults.NUM_BEAMS,
        compute_accuracy=True,
        optimizer: cli.OptimizerCallable = defaults.OPTIMIZER,
        scheduler: cli.LRSchedulerCallable = defaults.SCHEDULER,
    ):
        super().__init__()
        self.model = (
            transformers.EncoderDecoderModel.from_encoder_decoder_pretrained(
                model_name,
                model_name,
                tie_encoder_decoder=tie_encoder_decoder,
                encoder_hidden_dropout_prob=dropout,
                decoder_hidden_dropout_prob=dropout,
            )
        )
        # Necessary patching for decoding.
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        bos = tokenizer.cls_token_id
        eos = tokenizer.sep_token_id
        pad = tokenizer.pad_token_id
        self.model.config.decoder_start_token_id = bos
        self.model.config.bos_token_id = bos
        self.model.config.eos_token_id = eos
        self.model.config.pad_token_id = pad
        target_vocab_size = (
            self.model.get_decoder().get_output_embeddings().out_features
        )
        self.accuracy = (
            metrics.Accuracy(ignore_index=pad, num_classes=target_vocab_size)
            if compute_accuracy
            else None
        )
        self.generation_config = transformers.GenerationConfig(
            decoder_start_token_id=bos,
            bos_token_id=bos,
            eos_token_id=eos,
            pad_token_id=pad,
            early_stopping=True,
            num_beams=num_beams,
        )
        # Actually initialized by configure_optimizers.
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_hyperparameters()


class T5Model(BaseModel):
    """T5 model (usually mT5 or ByT5).

    After:
        Raffel, C., Shazeer, N. M., Roberts, A., Lee, K., Narang, S., Matena,
        M., ..., and Liu, P. J. 2020. Exploring the limits of transfer
        learning with a unified text-to-text transformer. Journal of Machine
        Learning Research 21: 1-67.

    Args:
        dropout: Dropout probability.
        label_smoothing: Label smoothing probability.
        model_name: Name of the Hugging Face T5 model.
        num_beams: Width of the beam to use during decoding.
    """

    model: transformers.T5ForConditionalGeneration

    def __init__(
        self,
        *,
        model_name="google/byt5-base",
        dropout=defaults.DROPOUT,
        label_smoothing=defaults.LABEL_SMOOTHING,
        num_beams=defaults.NUM_BEAMS,
        compute_accuracy=True,
        optimizer: cli.OptimizerCallable = defaults.OPTIMIZER,
        scheduler: cli.LRSchedulerCallable = defaults.SCHEDULER,
    ):
        super().__init__()
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            model_name, dropout_rate=dropout
        )
        # BOS is apparently not used.
        eos = self.model.config.eos_token_id
        pad = self.model.config.pad_token_id
        self.model.config.decoder_start_token_id = pad
        target_vocab_size = self.model.get_decoder().embed_tokens.embedding_dim
        self.accuracy = (
            metrics.Accuracy(ignore_index=pad, num_classes=target_vocab_size)
            if compute_accuracy
            else None
        )
        self.generation_config = transformers.GenerationConfig(
            decoder_start_token_id=pad,
            eos_token_id=eos,
            pad_token_id=pad,
            early_stopping=True,
            num_beams=num_beams,
        )
        # Actually initialized by configure_optimizers.
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_hyperparameters()

    # Overrides to discard the leading BOS in the POS. This is harmless for
    # prediction.

    def _decode(
        self, source: torch.Tensor, source_mask: torch.Tensor
    ) -> torch.Tensor:
        return super()._decode(source, source_mask)[:, 1:]

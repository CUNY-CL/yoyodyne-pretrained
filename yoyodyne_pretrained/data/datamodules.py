"""Data modules."""

import lightning
import transformers
from torch.utils import data

from .. import defaults
from . import collators, datasets, tsv


class DataModule(lightning.LightningDataModule):
    """String pair data module.

    Args:
        predict: Path to a TSV file for prediction.
        test: Path to a TSV file for testing.
        train: Path to a TSV file for training.
        val: Path to a TSV file for validation.
        source_col: 1-indexed column in TSV containing source strings.
        features_col: 1-indexed column in TSV containing features strings.
        target_col: 1-indexed column in TSV containing target strings.
        encoder: Full name of a Hugging Face encoder.
        decoder: Full name of a Hugging Face decoder.
        batch_size: Batch size.
    """

    predict: str | None
    test: str | None
    train: str | None
    val: str | None

    encoder_tokenizer: transformers.AutoTokenizer
    decoder_tokenizer: transformers.AutoTokenizer

    def __init__(
        self,
        # Paths.
        *,
        train=None,
        val=None,
        predict=None,
        test=None,
        # TSV parsing arguments.
        source_col: int = defaults.SOURCE_COL,
        features_col: int = defaults.FEATURES_COL,
        target_col: int = defaults.TARGET_COL,
        # Other.
        encoder: str = defaults.ENCODER,
        decoder: str = defaults.DECODER,
        batch_size: int = defaults.BATCH_SIZE,
    ):
        super().__init__()
        self.train = train
        self.val = val
        self.predict = predict
        self.test = test
        self.parser = tsv.TsvParser(
            source_col=source_col,
            features_col=features_col,
            target_col=target_col,
        )
        self.encoder_tokenizer = transformers.AutoTokenizer.from_pretrained(
            encoder
        )
        self.decoder_tokenizer = transformers.AutoTokenizer.from_pretrained(
            decoder
        )
        self.batch_size = batch_size

    # Required API.

    def train_dataloader(self) -> data.DataLoader:
        assert self.train is not None, "no train path"
        return data.DataLoader(
            self._dataset(self.train),
            collate_fn=self._collate_fn,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
            persistent_workers=True,
        )

    def val_dataloader(self) -> data.DataLoader:
        assert self.val is not None, "no val path"
        return data.DataLoader(
            self._dataset(self.val),
            collate_fn=self._collate_fn,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
        )

    def predict_dataloader(self) -> data.DataLoader:
        assert self.predict is not None, "no predict path"
        return data.DataLoader(
            self._dataset(self.predict),
            collate_fn=self._collate_fn,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
        )

    def test_dataloader(self) -> data.DataLoader:
        assert self.test is not None, "no test path"
        return data.DataLoader(
            self._dataset(self.test),
            collate_fn=self._collate_fn,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
        )

    def _dataset(self, path: str) -> datasets.Dataset:
        return datasets.Dataset(
            list(self.parser.samples(path)),
        )

    @property
    def _collate_fn(self) -> collators.Collator:
        return collators.Collator(
            self.encoder_tokenizer,
            self.decoder_tokenizer,
        )

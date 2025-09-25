"""Data modules."""

import lightning
import transformers
from torch.utils import data

from .. import defaults
from . import collators, datasets, tsv


class DataModule(lightning.LightningDataModule):
    """String pair data module.

    Args:
        model_dir: Path for checkpoints and logs.
        model_name: Full name of a Hugging Face model; filled in by linking.
        predict: Path to a TSV file for prediction.
        test: Path to a TSV file for testing.
        train: Path to a TSV file for training.
        val: Path to a TSV file for validation.
        source_col: 1-indexed column in TSV containing source strings.
        features_col: 1-indexed column in TSV containing features strings.
        target_col: 1-indexed column in TSV containing target strings.
        batch_size: Batch size.
    """

    predict: str | None
    test: str | None
    train: str | None
    val: str | None
    batch_size: int
    tokenizer: transformers.AutoTokenizer

    def __init__(
        self,
        # Paths.
        *,
        model_dir: str,
        model_name: str,
        train=None,
        val=None,
        predict=None,
        test=None,
        # TSV parsing arguments.
        source_col: int = defaults.SOURCE_COL,
        features_col: int = defaults.FEATURES_COL,
        target_col: int = defaults.TARGET_COL,
        # Other.
        batch_size: int = defaults.BATCH_SIZE,
    ):
        super().__init__()
        self.model_dir = model_dir
        self.train = train
        self.val = val
        self.predict = predict
        self.test = test
        self.parser = tsv.TsvParser(
            source_col=source_col,
            features_col=features_col,
            target_col=target_col,
        )
        self.batch_size = batch_size
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

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
        return collators.Collator(self.tokenizer)

"""TSV parsing.

Adapted from Yoyodyne:

    https://github.com/CUNY-CL/yoyodyne/blob/master/yoyodyne/data/tsv.py

Most of the features of the TsvParser can be quickly simplified however.
"""

import csv
import dataclasses
from typing import Iterator

from .. import defaults


class Error(Exception):
    pass


SampleType = str | tuple[str, str]


@dataclasses.dataclass
class TsvParser:
    """Streams data from a TSV file.

    Args:
        source_col: 1-indexed column in TSV containing source strings.
        features_col: 1-indexed column in TSV containing features strings.
        target_col: 1-indexed column in TSV containing target strings.
    """

    source_col: int = defaults.SOURCE_COL
    features_col: int = defaults.FEATURES_COL
    target_col: int = defaults.TARGET_COL

    def __post_init__(self) -> None:
        # This is automatically called after initialization.
        if self.source_col < 1:
            raise Error(f"Out of range source column: {self.source_col}")
        if self.features_col < 0:
            raise Error(f"Out of range features column: {self.features_col}")
        if self.target_col < 0:
            raise Error(f"Out of range target column: {self.target_col}")

    @staticmethod
    def _tsv_reader(path: str) -> Iterator[str]:
        with open(path, "r", encoding=defaults.ENCODING) as tsv:
            yield from csv.reader(tsv, delimiter="\t")

    @staticmethod
    def _get_string(row: list[str], col: int) -> str:
        """Returns a string from a row by index.

        Args:
           row: the split row.
           col: the column index.

        Returns:
           str: symbol from that string.
        """
        return row[col - 1]  # -1 because we're using one-based indexing.

    @property
    def has_features(self) -> bool:
        return self.features_col != 0

    @property
    def has_target(self) -> bool:
        return self.target_col != 0

    def samples(self, path: str) -> Iterator[SampleType]:
        """Yields source, and features and/or target if available."""
        for row in self._tsv_reader(path):
            source = self._get_string(row, self.source_col)
            if self.has_features:
                features = self._get_string(row, self.features_col)
                # Concatenates features after the source.
                source = f"{source} {features}"
            if self.has_target:
                target = self._get_string(row, self.target_col)
                yield source, target
            else:
                yield source

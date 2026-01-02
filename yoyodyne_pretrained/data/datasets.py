"""Datasets and related utilities."""

import abc
import dataclasses
import mmap
from typing import BinaryIO, Iterator

from torch.utils import data

from .. import defaults
from . import tsv


@dataclasses.dataclass
class AbstractDataset(abc.ABC):
    """Base class for datasets.

    Args:
        path (str).
        parser (tsv.TsvParser).
    """

    path: str
    parser: tsv.TsvParser


@dataclasses.dataclass
class IterableDataset(AbstractDataset, data.IterableDataset):
    """Iterable (non-random access) data set."""

    def __iter__(self) -> Iterator[tsv.SampleType]:
        yield from self.parser.samples(self.path)


@dataclasses.dataclass
class MappableDataset(data.Dataset):
    """Mappable (random access) data set.

    This is implemented with a memory map after making a single pass through
    the file to compute offsets."""

    _offsets: list[int] = dataclasses.field(default_factory=list, init=False)
    _mmap: mmap.mmap | None = dataclasses.field(default=None, init=False)
    _fobj: BinaryIO | None = dataclasses.field(default=None, init=False)

    def __post_init__(self):
        # Computes offsets.
        self._offsets = []
        with open(self.path, "rb") as source:
            offset = 0
            for line in source:
                self._offsets.append(offset)
                offset += len(line)

    def _get_mmap(self) -> mmap.mmap:
        # Makes this safe for use with multiple workers.
        if self._mmap is None:
            self._fobj = open(self.path, "rb")
            self._mmap = mmap.mmap(
                self._fobj.fileno(), 0, access=mmap.ACCESS_READ
            )
        return self._mmap

    # Required API.

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, idx: int) -> tsv.SampleType:
        mm = self._get_mmap()
        start = self._offsets[idx]
        if idx + 1 < len(self._offsets):
            end = self._offsets[idx + 1]
        else:
            end = mm.size()
        line = mm[start:end].decode(defaults.ENCODING).rstrip()
        return self.parser.parse_line(line)

    def __del__(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
        if self._fobj is not None:
            self._fobj.close()

    # Properties.

    @property
    def has_target(self) -> bool:
        return self.parser.has_target

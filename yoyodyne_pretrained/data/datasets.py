"""Dataset objects."""

import dataclasses

from torch.utils import data

from . import tsv


@dataclasses.dataclass
class Dataset(data.Dataset):

    samples: list[tsv.SampleType]

    # Required API.

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tsv.SampleType:
        return self.samples[idx]

"""Collator objects."""

import dataclasses

import transformers

from . import batches, tsv


@dataclasses.dataclass
class Collator:
    """Collator for text data."""

    tokenizer: transformers.AutoTokenizer

    def __call__(self, itemlist: list[tsv.SampleType]) -> batches.Batch:
        source, target = zip(*itemlist)
        encoding = self.tokenizer(
            source, padding="longest", return_tensors="pt"
        )
        if target:
            decoding = self.tokenizer(
                target, padding="longest", return_tensors="pt"
            )
            return batches.Batch(
                encoding.input_ids,
                encoding.attention_mask,
                decoding.input_ids,
                decoding.attention_mask,
            )
        else:
            return batches.Batch(encoding.input_ids, encoding.attention_mask)

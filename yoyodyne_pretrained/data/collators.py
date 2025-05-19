"""Collator objects."""

import dataclasses
import logging  # FIXME

import transformers
import torch

from . import batches, tsv


@dataclasses.dataclass
class Collator:
    """Collator for text data."""

    source_tokenizer: transformers.AutoTokenizer
    target_tokenizer: transformers.AutoTokenizer | None

    def __call__(self, itemlist: list[tsv.SampleType]) -> batches.Batch:
        source, target = zip(*itemlist)
        source_ids, source_mask = self._tokenize(source, self.source_tokenizer)
        if target:
            target_ids, target_mask = self._tokenize(
                target, self.target_tokenizer
            )
            return batches.Batch(
                source_ids, source_mask, target_ids, target_mask
            )
        else:
            return batches.Batch(source_ids, source_mask)

    def _tokenize(
        self,
        itemlist: list[tuple[str, str]],
        tokenizer: transformers.AutoTokenizer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = tokenizer(
            itemlist,
            padding="longest",
            truncation=False,
            return_tensors="pt",
            add_special_tokens=False,
        )
        return output.input_ids, output.attention_mask
        # FIXME: detect and handle overlong strings.

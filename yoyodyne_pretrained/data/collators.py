"""Collator objects."""

import dataclasses
import logging  # FIXME

import transformers

from . import batches, tsv


@dataclasses.dataclass
class Collator:
    """Collator for text data."""

    source_tokenizer: transformers.AutoTokenizer
    target_tokenizer: transformers.AutoTokenizer | None

    def __call__(self, itemlist: list[tsv.SampleType]) -> batches.Batch:
        source, target = zip(*itemlist)
        source_tokenized = self._tokenize(source, self.source_tokenizer)
        target_tokenized = (
            self._tokenize(target, self.target_tokenizer) if target else None
        )
        return batches.Batch(source=source_tokenized, target=target_tokenized)

    def _tokenize(
        self,
        itemlist: list[tuple[str, str]],
        tokenizer: transformers.AutoTokenizer,
    ) -> batches.Tokenized:
        output = tokenizer(
            itemlist,
            padding="longest",
            truncation=False,
            return_tensors="pt",
            add_special_tokens=False,
        )
        return batches.Tokenized(output.input_ids, output.attention_mask)

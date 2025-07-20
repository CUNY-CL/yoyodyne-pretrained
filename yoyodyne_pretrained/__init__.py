"""Yoyodyne Pretrained: pre-trained sequence-to-sequence transduction.

This module just silences some uninformative warnings.
"""

import os
import warnings

# Silences tokenizers warning about forking.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings(
    "ignore",
    ".*To copy construct from a tensor.*",
)
warnings.filterwarnings(
    "ignore", ".*to train encoder-decoder models by computing the loss.*"
)
warnings.filterwarnings(
    "ignore", ".*does not have many workers which may be a bottleneck.*"
)
warnings.filterwarnings(
    "ignore", ".*Couldn't infer the batch indices fetched from.*"
)

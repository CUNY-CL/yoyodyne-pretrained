"""Defaults."""

from torch import optim

from . import schedulers

# Data configuration arguments.
SOURCE_COL = 1
TARGET_COL = 2
FEATURES_COL = 0

# Training arguments.
BATCH_SIZE = 32
DROPOUT = 0.5

# Architecture arguments.
DECODER = "google-bert/bert-base-multilingual-cased"
ENCODER = "google-bert/bert-base-multilingual-cased"

# Optimizer options.
OPTIMIZER = optim.Adam
SCHEDULER = schedulers.Dummy

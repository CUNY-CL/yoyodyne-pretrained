"""Defaults."""

from torch import optim
from yoyodyne import schedulers

# Default text encoding.
ENCODING = "utf-8"

# Data configuration arguments.
SOURCE_COL = 1
TARGET_COL = 2
FEATURES_COL = 0

# Training arguments.
BATCH_SIZE = 32
DROPOUT = 0.5
LABEL_SMOOTHING = 0.0

# Decoding arguments.
NUM_BEAMS = 5

# Optimizer options.
OPTIMIZER = optim.Adam
SCHEDULER = schedulers.Dummy

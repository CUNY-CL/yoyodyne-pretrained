"""Custom schedulers.

Adapted from UDTube:

    https://github.com/CUNY-CL/udtube/blob/master/udtube/schedulers.py
"""

from torch import optim


# TODO: This will eventually be available in vanilla Yoyodyne; use that and
# delete this once it is.


class Dummy(optim.lr_scheduler.LRScheduler):
    """A dummy scheduler that holds learning rate constant.

    Args:
        optimizer: optimizer.
    """

    optimizer: optim.Optimizer

    def __init__(self, optimizer):
        super().__init__(optimizer)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.optimizer})"

    def get_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

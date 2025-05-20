"""Custom schedulers.

Adapted from UDTube:

    https://github.com/CUNY-CL/udtube/blob/master/udtube/schedulers.py
"""

import numpy
from torch import optim


# TODO: This will eventually be available from vanilla Yoyodyne; use that and
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


class WarmupInverseSquareRoot(optim.lr_scheduler.LambdaLR):
    """Linear warmup and then inverse square root decay.

    Linearly increases learning rate from 0 to the learning rate over the
    warmup steps, then decreases learning rate according to an inverse root
    square schedule.

    After:
        Wu, S., Cotterell, R., and Hulden, M. 2021. Applying the transformer to
        character-level transductions. In Proceedings of the 16th Conference of
        the European Chapter of the Association for Computational Linguistics:
        Main Volume, pages 1901-1907.

    Args:
        optimizer (optim.Optimizer): optimizer.
        warmup_steps (int): number of warmup steps.
        *args: ignored.
        **kwargs: ignored.
    """

    warmup_steps: int
    decay_factor: float

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps,
        *args,
        **kwargs,
    ):
        self.warmup_steps = warmup_steps
        self.decay_factor = numpy.sqrt(warmup_steps)
        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step: int) -> float:
        """Computes the learning rate lambda at a given steps.

        Args:
            step (int): current step.

        Returns:
            float: lr_lambda.
        """
        if step < self.warmup_steps:
            # +1 in numerator avoids a zero-LR first step.
            return (step + 1) / self.warmup_steps
        # +1 in base of exponent avoids an undefined operation (0 to a negative
        # exponent) in the unlikely case one is using this without warmup.
        return self.decay_factor * (step + 1) ** -0.5

    def config_dict(self) -> dict[str, ...]:
        return {"interval": "step", "frequency": 1}

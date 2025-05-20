"""Subclasses the trainer to prevent a memory leak in prediction.

This is adapted from UDTube:

    https://github.com/CUNY-CL/udtube/blob/master/udtube/trainers.py
"""

# TODO: This will eventually be available from vanilla Yoyodyne; use that and
# delete this once it is.


from lightning.pytorch import trainer


class Trainer(trainer.Trainer):

    def predict(self, *args, **kwargs):
        return super().predict(*args, return_predictions=False, **kwargs)

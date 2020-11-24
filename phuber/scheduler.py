import torch


class ExponentialDecayLR(torch.optim.lr_scheduler.LambdaLR):
    """Implements ExponentialDecay scheduler from Keras.
    To match Keras behaviour, call scheduler at each optimizer step (usually at each iteration).
    When last_epoch=-1, sets initial lr as lr.

    Sets the learning rate to:
    lr = initial_lr * (decay_rate ^ (step / decay_step))

    If staircase is True, step / decay_steps is an integer division.

    Args:
        optimizer (Optimizer): Wrapped optimizer
        decay_steps (int): Must be positive, see the decay computation above.
        decay_rate (float): The decay rate
        staircase (bool): If True, the learning rate is decayed at discrete intervals.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        decay_steps: int,
        decay_rate: float,
        staircase: bool = False,
        last_epoch: int = -1,
    ) -> None:

        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

        if staircase:
            decay = lambda step: decay_rate ** (int(step / decay_steps))
        else:
            decay = lambda step: decay_rate ** (step / decay_steps)

        super().__init__(optimizer, lr_lambda=decay, last_epoch=last_epoch)


class ConstantLR(torch.optim.lr_scheduler.LambdaLR):
    """Keeps the learning rate constant (does nothing).

    Can be used when no scheduler is needed, avoids modifying the training loop.

    Args:
        optimizer (Optimizer): Wrapped optimizer
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, last_epoch: int = -1,) -> None:

        decay = lambda epoch: 1
        super().__init__(optimizer, lr_lambda=decay, last_epoch=last_epoch)

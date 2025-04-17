from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer

def get_scheduler(optimizer: Optimizer, name: str="const", **kwargs):
    """
    Get the scheduler.

    Parameters:
        optimizer(torch.optim.Optimizer): optimizer.
        name(str, default="const"): name of the scheduler.
        kwargs: other arguments for the scheduler.

    Returns:
        scheduler(torch.optim.lr_scheduler._LRScheduler): scheduler.
    """
    if name == "const":
        return None
    elif name == "TransformerLR":
        return TransformerLR(optimizer, **kwargs)
    else:
        raise ValueError(f"Scheduler {name} is not supported.")

class TransformerLR(_LRScheduler):
    """TransformerLR class for adjustment of learning rate.

    The scheduling is based on the method proposed in 'Attention is All You Need'.
    """

    def __init__(self, optimizer, warmup_epochs=1000, last_epoch=-1, verbose=False):
        """Initialize class."""
        self.warmup_epochs = warmup_epochs
        self.normalize = self.warmup_epochs**0.5
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """Return adjusted learning rate."""
        step = self.last_epoch + 1
        scale = self.normalize * min(step**-0.5, step * self.warmup_epochs**-1.5)
        return [base_lr * scale for base_lr in self.base_lrs]
from torch.optim.lr_scheduler import _LRScheduler

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
import torch
import warnings

def get_optimizer(variables: list[torch.Tensor], lr: float, name: str="SGD", **kwargs):
    """
    Get the optimizer.

    Parameters:
        variables(list[torch.Tensor]): list of variables to optimize.
        lr(float): learning rate.
        name(str): name of the optimizer. Default is SGD.
        kwargs: other arguments for the optimizer.d

    Returns:
        optimizer(torch.optim.Optimizer): optimizer.
    """
    warnings.warn("ph_opt.optimizer.optimizer.get_optimizer is deprecated.", DeprecationWarning)

    if name == "SGD":
        return torch.optim.SGD(variables, lr=lr)
    elif name == "Adam":
        return torch.optim.Adam(variables, lr=lr)
    elif name == "RMSprop":
        return torch.optim.RMSprop(variables, lr=lr)
    else:
        raise ValueError(f"Optimizer {name} is not supported.")
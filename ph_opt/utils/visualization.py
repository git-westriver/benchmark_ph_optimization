import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from gudhi import plot_persistence_diagram
from typing import Optional
from collections.abc import Sized, Iterable
from numbers import Number

from ph_opt import RipsPH

def is_persistence_diagram(obj):
    """
    Check if the input is a persistence diagram (Sized[Sized[int, Sized[float, float]]]) or not.
    Sized is an iterable type with method __len__, such as list, tuple, etc.

    Args:
        obj (Any): Input object to check.

    Returns:
        bool: True if the input is a persistence diagram, False otherwise.
    """
    if not isinstance(obj, Sized):
        return False
    for item in obj:
        if not (isinstance(item, Sized) and len(item) == 2):
            return False
        if not isinstance(item[0], int):
            return False
        if not (isinstance(item[1], Sized) and len(item[1]) == 2):
            return False
        if not all(isinstance(coord, Number) for coord in item[1]):
            return False
    return True

def get_max_death_of_pds(pds):
    """
    Get the maximum death value of the persistence diagrams.

    Args:
        pds (Array[list[tuple[int, tuple[float, float]]]]): Array of persistence diagrams. Each PD is a list of tuples (dimension, (birth, death)).\
            The array can be nested, i.e., `pds` can be a persistence diagrams, a list of persistence diagrams, or a list of lists of persistence diagrams (and so on).

    Returns:
        float: Maximum death value.
    """
    if is_persistence_diagram(pds):
        return max([death for dim, (birth, death) in pds if not math.isinf(death)])
    elif not isinstance(pds, Iterable):
        raise TypeError("The argument `pds` have to be Iterable.")
    else:
        return max([get_max_death_of_pds(obj) for obj in pds])

def plot_pd_with_specified_lim(pds, axes, high=None, 
                               titles=None, x_labels=None, y_labels=None,
                               legend=True):
    """
    Plot the persistence diagram with specified limits.

    Args:
        pds (list[list[tuple[int, tuple[float, float]]]]): List of persistence diagrams. Each PD is a list of tuples (dimension, (birth, death)).
        axes (list[matplotlib.axes.Axes]): List of axes to plot the persistence diagram.
        high (Optional[float]): Upper limit of the x-axis and y-axis. If None, the maximum death value is used.
        titles (Optional[list[str]]): Title of the plot. If None, the default of GUDHI is used.
        x_labels (Optional[list[str]]): Label of the x-axis. If None, the default of GUDHI is used.
        y_labels (Optional[list[str]]): Label of the y-axis. If None, the default of GUDHI is used.
        legend (bool, default=True): Whether to show the legend or not.
    """
    # get maximum death value if high is None
    if high is None:
        high = get_max_death_of_pds(pds)

    # get maximum dimension
    max_dim = max([max([bar[0] for bar in pd]) for pd in pds])

    # get colormap
    pd_colormap = list(plt.cm.Set1.colors)
    for dim in range(len(pd_colormap)):
        if dim == max_dim+1:
            pd_colormap[max_dim+1] = (1, 1, 1, 0) # transparent
        else:
            pd_colormap[dim] = pd_colormap[dim] + (1,) # add alpha value (non-transparent)
    pd_colormap = tuple(pd_colormap)

    # plot persistence diagrams
    for i, (pd, ax) in enumerate(zip(pds, axes)):
        # add null point to the PD
        pd.append((max_dim+1, (0, high*1.01)))

        # plot the PD
        plot_persistence_diagram(pd, axes=ax, colormap=pd_colormap, legend=False)

        # set title and labels
        if titles is not None:
            ax.set_title(titles[i])
        if x_labels is not None:
            ax.set_xlabel(x_labels[i])
        if y_labels is not None:
            ax.set_ylabel(y_labels[i])

        # add legend
        if legend:
            for dim in range(max_dim+1):
                ax.scatter([], [], color=plt.cm.Set1.colors[dim], label=str(dim))
            ax.legend(loc="lower right")

def get_animation(
        X_history: list[list[torch.Tensor]], 
        loss_history: list[list[list[float]]],
        dim_list: list[int], 
        title_list: list[str],
        color_list: Optional[list[str]] = None,
        figsize: Optional[tuple[int, int]] = None,
        vertical: Optional[bool] = True,
    ) -> animation.FuncAnimation:
    """
    Create an animation of optimization process.
    Note that you can specify `X_history` for only one trial in each experiment setting.

    Args:
        X_history (list[list[torch.Tensor]]): List of optimization variables. `X_history[i][j]` denotes the variable at `j`-th epoch in `i`-th experiment setting. 
        loss_history (list[list[list[float]]]): List of loss values. `loss_history[i][j][k]` denotes the loss value at `k`-th epoch of `j`-th trial in `i`-th experiment setting.
        dim_list (list[int]): List of dimensions to plot the persistence diagram.
        title_list (list[str]): List of titles for each experiment setting.
        color_list (Optional[list[str]], default=None): List of colors for each experiment setting. If None, red, green, and blue are repeatedly used.
        figsize (tuple[int, int]): Figure size. If None, the size is set to (5 * len(title_list), 15).
        vertical (Optional[bool], default=True): If True, the animation is drawn vertically for each experiment setting.
    """
    # num_setting, num_epoch
    print("uoaaa3")
    num_setting = len(X_history)
    num_epoch = len(loss_history[0][0])
    for i in range(len(loss_history)):
        for j in range(len(loss_history[i])):
            num_epoch = min(num_epoch, len(loss_history[i][j]))

    # convert None to default values
    print("uoaaa4")
    if color_list is None:
        default_colors = ["red", "green", "blue", "orange"]
        color_list = [default_colors[i%len(default_colors)] for i in range(num_setting)]
    if figsize is None:
        if vertical:
            figsize = (15, 5 * len(title_list))
        else:
            figsize = (5 * len(title_list), 15)

    # loss_mean, loss_std
    print("uoaaa5")
    loss_mean, loss_std = [], []
    for i in range(num_setting):
        loss_mean.append(np.mean([loss_history[i][j][:num_epoch] for j in range(len(loss_history[i]))], axis=0))
        loss_std.append(np.std([loss_history[i][j][:num_epoch] for j in range(len(loss_history[i]))], axis=0))

    # xmin, xmax, ymin, ymax
    print("uoaaa6")
    xmin = min([torch.min(X_history[i][j][:, 0]).item() for i in range(num_setting) for j in range(len(X_history[i]))])
    xmax = max([torch.max(X_history[i][j][:, 0]).item() for i in range(num_setting) for j in range(len(X_history[i]))])
    ymin = min([torch.min(X_history[i][j][:, 1]).item() for i in range(num_setting) for j in range(len(X_history[i]))])
    ymax = max([torch.max(X_history[i][j][:, 1]).item() for i in range(num_setting) for j in range(len(X_history[i]))])

    # min_loss, max_loss
    print("uoaaa7")
    min_loss = min([np.min(loss_mean[i]) for i in range(num_setting)])
    max_loss = max([loss_mean[i][0] for i in range(num_setting)])
    loss_range = max_loss - min_loss
    min_loss, max_loss = min_loss - 0.005 * loss_range, max_loss + 0.005 * loss_range
    if vertical:
        fig, axes = plt.subplots(3, num_setting, figsize=figsize, squeeze=False)
    else:
        fig, axes = plt.subplots(num_setting, 3, figsize=figsize, squeeze=False)
        axes = axes.T

    # compute PDs beforehand
    print("uoaaa8")
    PD_history = [[] for i in range(num_setting)]
    max_dim = max(dim_list)
    max_death = 0
    for i in range(num_setting):
        for j in range(len(X_history[i])):
            print(i, j)
            rph = RipsPH(X_history[i][j].detach().numpy(), maxdim=max_dim)
            barcode = []
            for dim in dim_list:
                barcode += [(dim, (birth, death)) for birth, death in rph.get_barcode(dim)]
            # rips = RipsComplex(points=X_history[i][j].detach().numpy())
            # simplex_tree = rips.create_simplex_tree(max_dimension=max_dim+1)
            # barcode = simplex_tree.persistence()
            # barcode = [(dim, (birth, death)) for dim, (birth, death) in barcode if dim in dim_list]
            PD_history[i].append(barcode)

    # get maximum death value
    print("uoaaa9")
    max_death = get_max_death_of_pds(PD_history)

    # update function for animation
    def update(idx):
        for i in range(num_setting):
            # get data at idx-th epoch
            X = X_history[i][idx].detach().numpy()
            pd = PD_history[i][idx]
            loss = loss_mean[i][idx]

            # get axes
            ax_X, ax_pd, ax_loss = axes[:, i]

            # plot the optimization variable
            ax_X.clear()
            ax_X.set_title(title_list[i])
            ax_X.set_xlim(xmin, xmax)
            ax_X.set_ylim(ymin, ymax)
            ax_X.scatter(X[:, 0], X[:, 1], c="black")
            ax_X.set_aspect("equal")

            # draw the PD
            ax_pd.clear()
            plot_pd_with_specified_lim([pd], [ax_pd], high=max_death, 
                                       titles=[""], x_labels=[""], y_labels=[""])

            # draw the loss curve
            ax_loss.clear()
            ax_loss.set_xlim(-1, num_epoch)
            ax_loss.set_ylim(min_loss, max_loss)
            ax_loss.plot(loss_mean[i][:idx+1], color=color_list[i])
            ax_loss.fill_between(range(idx+1), loss_mean[i][:idx+1]-loss_std[i][:idx+1], loss_mean[i][:idx+1]+loss_std[i][:idx+1], color=color_list[i], alpha=0.3)
            ax_loss.scatter(idx, loss, c=color_list[i])
            ax_loss.plot([idx, idx], [min_loss, max_loss], color=color_list[i], linestyle="--")

    # create animation
    print("uoaaa10")
    ani = animation.FuncAnimation(fig, update, frames=num_epoch, interval=100)

    return ani

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from gudhi.rips_complex import RipsComplex
from gudhi import plot_persistence_diagram
from typing import Optional

def get_animation(
        X_history: list[list[torch.Tensor]], 
        loss_history: list[list[list[float]]],
        dim_list: list[int], 
        title_list: list[str],
        color_list: Optional[list[str]] = None,
        figsize: Optional[tuple[int, int]] = None,
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
    """
    # num_setting, num_epoch
    num_setting = len(X_history)
    num_epoch = len(loss_history[0][0])
    for i in range(len(loss_history)):
        for j in range(len(loss_history[i])):
            num_epoch = min(num_epoch, len(loss_history[i][j]))
    # convert None to default values
    if color_list is None:
        default_colors = ["red", "green", "blue", "orange"]
        color_list = [default_colors[i%len(default_colors)] for i in range(num_setting)]
    if figsize is None:
        figsize = (5 * len(title_list), 15)
    # loss_mean, loss_std
    loss_mean = []; loss_std = []
    for i in range(num_setting):
        loss_mean.append(np.mean([loss_history[i][j][:num_epoch] for j in range(len(loss_history[i]))], axis=0))
        loss_std.append(np.std([loss_history[i][j][:num_epoch] for j in range(len(loss_history[i]))], axis=0))
    # xmin, xmax, ymin, ymax
    xmin = min([torch.min(X_history[i][j][:, 0]).item() for i in range(num_setting) for j in range(len(X_history[i]))])
    xmax = max([torch.max(X_history[i][j][:, 0]).item() for i in range(num_setting) for j in range(len(X_history[i]))])
    ymin = min([torch.min(X_history[i][j][:, 1]).item() for i in range(num_setting) for j in range(len(X_history[i]))])
    ymax = max([torch.max(X_history[i][j][:, 1]).item() for i in range(num_setting) for j in range(len(X_history[i]))])
    # min_loss, max_loss
    min_loss = min([np.min(loss_mean[i]) for i in range(num_setting)])
    max_loss = max([loss_mean[i][0] for i in range(num_setting)])
    loss_range = max_loss - min_loss
    min_loss, max_loss = min_loss - 0.005 * loss_range, max_loss + 0.005 * loss_range
    fig, axes = plt.subplots(3, num_setting, figsize=(5*num_setting, 15))
    # compute PDs beforehands
    PD_history = [[] for i in range(num_setting)]
    max_dim = max(dim_list)
    max_death = 0
    for i in range(num_setting):
        for j in range(len(X_history[i])):
            rips = RipsComplex(points=X_history[i][j].detach().numpy())
            simplex_tree = rips.create_simplex_tree(max_dimension=max_dim+1)
            barcode = simplex_tree.persistence()
            barcode = [(dim, (birth, death)) for dim, (birth, death) in barcode if dim in dim_list]
            PD_history[i].append(barcode)
            max_death = max(max_death, max([bar[1][1] for bar in barcode]))
    pd_colormap = list(plt.cm.Set1.colors)
    for dim in range(len(pd_colormap)):
        if dim == max_dim+1:
            pd_colormap[max_dim+1] = (1, 1, 1, 0)
        else:
            pd_colormap[dim] = pd_colormap[dim] + (1,)
    pd_colormap = tuple(pd_colormap)
    # update function for animation
    def update(idx):
        for i in range(num_setting):
            X = X_history[i][idx].detach().numpy(); pd = PD_history[i][idx] + [(max_dim+1, (0, max_death*1.01))]; loss = loss_mean[i][idx]
            ax_X = axes[0, i]; ax_pd = axes[1, i]; ax_loss = axes[2, i]
            # plot the optimization variable
            ax_X.clear(); ax_pd.clear(); ax_loss.clear()
            ax_X.set_title(title_list[i])
            ax_X.set_xlim(xmin, xmax); ax_X.set_ylim(ymin, ymax)
            ax_X.scatter(X[:, 0], X[:, 1], c="black")
            ax_X.set_aspect("equal")
            # draw the PD
            plot_persistence_diagram(pd, axes=ax_pd, colormap=pd_colormap, legend=False)
            ax_pd.set_title(""); ax_pd.set_xlabel(""); ax_pd.set_ylabel("")
            for dim in dim_list:
                ax_pd.scatter([], [], color=plt.cm.Set1.colors[dim], label=str(dim))
            ax_pd.legend(loc="lower right")
            # draw the loss curve
            ax_loss.set_xlim(-1, num_epoch)
            ax_loss.set_ylim(min_loss, max_loss)
            ax_loss.plot(loss_mean[i][:idx+1], color=color_list[i])
            ax_loss.fill_between(range(idx+1), loss_mean[i][:idx+1]-loss_std[i][:idx+1], loss_mean[i][:idx+1]+loss_std[i][:idx+1], color=color_list[i], alpha=0.3)
            ax_loss.scatter(idx, loss, c=color_list[i])
            ax_loss.plot([idx, idx], [min_loss, max_loss], color=color_list[i], linestyle="--")
    # create animation
    ani = animation.FuncAnimation(fig, update, frames=num_epoch, interval=100)

    return ani

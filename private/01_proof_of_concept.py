from ph_opt import PHTrainerConfig, PHTrainer, RipsPH, powered_wasserstein_distance_one_sided, topk_persistence_loss
import torch
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR
from functools import partial
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys

import warnings
warnings.filterwarnings("ignore")

def loss_for_large_hole_with_regtangle_regularization(X: torch.Tensor, grad_type,
                                                      sigma: float, 
                                                      lamb: float, x_min: float, y_min: float, x_max: float, y_max: float):
    ref_pd = [torch.zeros(size=(0, 2), dtype=X.dtype)] # [torch.tensor([[0., 4.]], dtype=X.dtype)]
    loss = (-1) * powered_wasserstein_distance_one_sided(
        X = X, 
        ref_pd = ref_pd, 
        dims = [1],
        # order = 1, 
        grad_type = grad_type, 
        clip_grad = "none",
        eps = 1.,
        n_strata=50, 
    )
    # loss = (-1) * topk_persistence_loss(
    #     X = X, 
    #     dims = [1],
    #     # order = 1, 
    #     topk=1,
    #     grad_type = grad_type, 
    #     clip_grad = "none", # "linf",
    #     eps = 1.,
    #     n_strata=50, 
    # )

    # Rectangle regularization
    penalty_x = torch.relu(X[:, 0] - x_max) + torch.relu(x_min - X[:, 0])
    penalty_y = torch.relu(X[:, 1] - y_max) + torch.relu(y_min - X[:, 1])
    loss += lamb * (torch.sum(penalty_x ** 2) + torch.sum(penalty_y ** 2))

    return loss

if __name__ == "__main__":
    loss_obj = partial(loss_for_large_hole_with_regtangle_regularization, 
                       sigma=0.5, 
                       lamb=1., x_min=-2., y_min=-2., x_max=2., y_max=2.)
    
    save_dirpath = Path("logs/poc/")
    method_list = [] # ["bigstep"] # ["gd", "stratified", "continuation", "diffeo"] # ["gd", "stratified", "continuation", "diffeo"] # 
    lr_list = [0.128]  #[0.256, 0.128, 0.064]
    gamma_list = [1.0, 0.9, 0.8, 0.7]
    num_epoch = 20
    do_all_plot = True

    for method in method_list:
        for lr in lr_list:
            for gamma in gamma_list:
                config = PHTrainerConfig(
                    loss_obj=loss_obj, 
                    exp_name=f"{method}_lr={lr:.3f}_gamma={gamma:.1f}", 
                    save_dirpath=save_dirpath, 
                    method=method, lr=lr, 
                    num_epoch=num_epoch, log_interval=1, 
                    # scheduler_class=LambdaLR, 
                    # scheduler_config={"lr_lambda": lambda step: max(0.0, 1.0 - step / num_epoch)}, 
                    scheduler_class=ExponentialLR, 
                    scheduler_config={"gamma": gamma}, 
                )
                pht = PHTrainer(config, viz_dims=[1])
                pht.train()
                print(f"↑ method = {method}, lr = {lr}, gamma = {gamma}")
    
    snapshot_steps = [0, 1, 5, 10, 20]
    max_arrow_len = 0.5
    for method in method_list:
        for lr in lr_list:
            for gamma in gamma_list:
                load_path = save_dirpath / f"{method}_lr={lr:.3f}_gamma={gamma:.1f}" / "X_history.pkl"
                with open(load_path, mode="rb") as f:
                    X_histories = pickle.load(f)

                for step in snapshot_steps:
                    X = X_histories[step]
                    X.requires_grad_()
                    points = X.detach().cpu().numpy()

                    # 損失と各点の勾配
                    X.grad = None
                    loss = loss_obj(X, grad_type=method if method != "gd" else "standard")
                    loss.backward()
                    grads  = X.grad.detach().cpu().numpy()

                    # 最大の矢印の長さが max_arrow_len になるようにスケール
                    norms = np.linalg.norm(grads, axis=1)  # shape: (num_pts,)
                    max_norm = norms.max()
                    if max_norm == 0:
                        max_norm = 1.0
                    grads_vis = grads * (max_arrow_len / max_norm)
                    
                    # 勾配降下方向を描きたいのでマイナス
                    dx = -grads_vis[:, 0]
                    dy = -grads_vis[:, 1]

                    # 勾配が 0 でない点のみを残す
                    eps = 0.0  
                    mask = (np.abs(dx) > eps) | (np.abs(dy) > eps)
                    points_masked = points[mask]
                    dx_masked = dx[mask]
                    dy_masked = dy[mask]

                    # プロット
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.scatter(points[:, 0], points[:, 1], s=20)
                    ax.quiver(
                        points_masked[:, 0], points_masked[:, 1],
                        dx_masked, dy_masked,
                        angles='xy',
                        scale_units='xy',
                        scale=1.0
                    )
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.set_aspect('equal')

                    # 画像の保存
                    save_path = save_dirpath / f"{method}_lr={lr:.3f}" / f"snapshot_step={step}.png"
                    fig.savefig(save_path)
                    print(f"Saved figure to: {save_path}")
    
    # ----- 複数画像をまとめてプロット -----
    if not do_all_plot:
        sys.exit()

    plot_targets = [
        ("gd", 0.256, 0.9), 
        ("stratified", 0.256, 1.0), 
        ("continuation", 0.256, 1.0), 
        ("diffeo", 0.256, 0.9), 
        ("bigstep", 0.064, 0.7)
    ]
    xylims = (-1.4, 1.7, -1.2, 1.9)

    # -- 損失関数の描画 --
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Steps", fontsize=16)
    ax.set_ylabel("Loss", fontsize=16)
    ax.set_xticks(list(range(0, 21, 5)))
    ax.tick_params(axis="both", labelsize=13)

    for method, lr, gamma in plot_targets:
        load_path = save_dirpath / f"{method}_lr={lr:.3f}_gamma={gamma:.1f}" / "result_dict.pkl"
        with open(load_path, mode="rb") as f:
            result = pickle.load(f)
        ax.plot(
            list(range(21)), result["loss_history"][0],
            label=method, 
            linewidth=2.5
        )

    ax.legend(fontsize=14)

    # 画像の保存
    save_path = save_dirpath / "loss.png"
    fig.tight_layout()  
    fig.savefig(save_path)
    print(f"Saved figure to: {save_path}")

    # -- snapshot の描画 -- 
    ncols, nrows = len(snapshot_steps), len(plot_targets)
    fig, axes = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=(5*ncols, 4*nrows), 
        sharex=True, 
        sharey=True
    )
    for row, (method, lr, gamma) in enumerate(plot_targets):
        load_path = save_dirpath / f"{method}_lr={lr:.3f}_gamma={gamma:.1f}" / "X_history.pkl"
        with open(load_path, mode="rb") as f:
            X_histories = pickle.load(f)

        for col, step in enumerate(snapshot_steps):
            ax = axes[row, col]
            # 各行の一番左に手法名をかく
            if col == 0:
                ax.set_ylabel(
                    method, 
                    rotation=0,     # 横書き
                    ha="right",     # 右寄せ
                    va="center",
                    labelpad=30,    # 軸から少し離す
                    fontsize=20,
                )

            # x軸ラベル & ticks を一番下の行だけにする
            if row == nrows - 1:
                ax.tick_params(labelbottom=True, labelsize=12)    # 目盛ラベルを表示（sharex=True なら自動で揃う）
            elif row == 0:
                ax.set_title(f"Step = {step}", fontsize=16)
            else:
                ax.tick_params(labelbottom=False, labelsize=12)   # 上の行では x の目盛ラベルを消す

            # y軸ラベル & ticks を一番左の列だけにする
            if col == 0:
                ax.tick_params(labelleft=True, labelsize=12)
            else:
                ax.tick_params(labelleft=False, labelsize=12)

            # 表示範囲を指定
            if xylims is not None:
                ax.set_xlim(xylims[0], xylims[1])
                ax.set_ylim(xylims[2], xylims[3])

            X = X_histories[step]
            X.requires_grad_()
            points = X.detach().cpu().numpy()

            # 損失と各点の勾配
            X.grad = None
            loss = loss_obj(X, grad_type=method if method != "gd" else "standard")
            loss.backward()
            grads  = X.grad.detach().cpu().numpy()

            # 最大の矢印の長さが max_arrow_len になるようにスケール
            norms = np.linalg.norm(grads, axis=1)  # shape: (num_pts,)
            max_norm = norms.max()
            if max_norm == 0:
                max_norm = 1.0
            grads_vis = grads * (max_arrow_len / max_norm)
            
            # 勾配降下方向を描きたいのでマイナス
            dx = -grads_vis[:, 0]
            dy = -grads_vis[:, 1]

            # 勾配が 0 でない点のみを残す
            eps = 0.0  
            mask = (np.abs(dx) > eps) | (np.abs(dy) > eps)
            points_masked = points[mask]
            dx_masked = dx[mask]
            dy_masked = dy[mask]

            # プロット
            ax.scatter(points[:, 0], points[:, 1], s=20)
            ax.quiver(
                points_masked[:, 0], points_masked[:, 1],
                dx_masked, dy_masked,
                angles='xy',
                scale_units='xy',
                scale=1.0
            )
            ax.set_box_aspect(1)   
            ax.set_aspect('equal', adjustable="box")

    # 画像の保存
    save_path = save_dirpath / "all.png"
    fig.tight_layout()  
    fig.savefig(save_path)
    print(f"Saved figure to: {save_path}")
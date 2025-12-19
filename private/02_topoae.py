import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (3D 描画のために必要)
from ph_opt import plot_pd_with_specified_lim, RipsPH, powered_wasserstein_distance_one_sided
import random
import numpy as np
from time import time

# ============================================================
# シード固定用ユーティリティ
# ============================================================

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 再現性重視の設定（速度は多少落ちる可能性あり）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # 古い PyTorch などで未対応な場合は無視
        pass

# ============================================================
# 1. 点群生成
# ============================================================

def generate_point_cloud(
    dataset_name: str,
    N=100, 
    noise_scale=0.1, 
    device="cpu"
) -> torch.Tensor:
    """
    N 個の 3 次元点群を生成する
      ・半分は (x-0.7)^2 + y^2 = 1, z = 0 上
      ・残りは (x+0.7)^2 + z^2 = 1, y = 0 上
      ・各座標に [-noise_scale, noise_scale] の一様ノイズ
    """
    if dataset_name == "rings_interlock":
        N1 = N // 2
        N2 = N - N1

        # 1 つ目の円: (x-0.7)^2 + y^2 = 1, z = 0
        theta1 = torch.rand(N1, device=device) * 2.0 * math.pi
        x1 = 0.7 + torch.cos(theta1)
        y1 = torch.sin(theta1)
        z1 = torch.zeros_like(x1)

        # 2 つ目の円: (x+0.7)^2 + z^2 = 1, y = 0
        theta2 = torch.rand(N2, device=device) * 2.0 * math.pi
        x2 = -0.7 + torch.cos(theta2)
        y2 = torch.zeros_like(x2)
        z2 = torch.sin(theta2)

        # 結合
        x = torch.cat([x1, x2], dim=0)
        y = torch.cat([y1, y2], dim=0)
        z = torch.cat([z1, z2], dim=0)

        pts = torch.stack([x, y, z], dim=1)  # (N, 3)

        # 一様ノイズ [-noise_scale, noise_scale]
        noise = (torch.rand_like(pts) - 0.5) * (2.0 * noise_scale)
        pts_noisy = pts + noise

        return pts_noisy  # (N, 3)

    elif dataset_name == "one_ring":
        # 1 つ目の円: x^2 + y^2 = 1, z = 0
        theta = torch.rand(N, device=device) * 2.0 * math.pi
        x = torch.cos(theta)
        y = torch.sin(theta)
        z = torch.zeros_like(x)

        pts = torch.stack([x, y, z], dim=1)  # (N, 3)

        # 一様ノイズ [-noise_scale, noise_scale]
        noise = (torch.rand_like(pts) - 0.5) * (2.0 * noise_scale)
        pts_noisy = pts + noise

        return pts_noisy  # (N, 3)

    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

# ============================================================
# 2. 3x3 歪対称行列を作って行列指数 -> 9 次元埋め込み
# ============================================================

def skew_matrix_exp_embedding(points):
    """
    points: (N, 3) tensor (x, y, z)
    各点 (x, y, z) から
      [[ 0,  x,  y],
       [-x,  0,  z],
       [-y, -z,  0]]
    を作り，その matrix exponential を計算して 9 次元に flatten する
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    N = points.size(0)
    device = points.device

    M = torch.zeros(N, 3, 3, device=device)

    # 上三角
    M[:, 0, 1] = x
    M[:, 0, 2] = y
    M[:, 1, 2] = z
    # 下三角 (歪対称)
    M[:, 1, 0] = -x
    M[:, 2, 0] = -y
    M[:, 2, 1] = -z

    # 行列指数 (バッチ対応)
    ExpM = torch.matrix_exp(M)  # (N, 3, 3)
    embedding = ExpM.reshape(N, 9)  # 9 次元に flatten

    return embedding


# ============================================================
# 3. Autoencoder の定義 (9 -> 2 -> 9)
# ============================================================

class AE9to2(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(9, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Linear(width, 2),  
            nn.BatchNorm1d(2), # 潜在表現 2 次元
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Linear(width, 9),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z
    
# ============================================================
# 損失関数: データ生成/読み込み & 学習 & 可視化
# ============================================================
def recon_loss_with_topo_reg(
    inputs: torch.Tensor,
    recon: torch.Tensor,
    z: torch.Tensor,
    pd: list,
    lamb: float = 1.0,
    grad_type: str = "standard",
    sigma: float = 0.1,
) -> torch.Tensor:
    """
    入力点群 inputs と再構成点群 recon の MSE 損失に加えて，
    潜在表現 z のトポロジカル正則化項を加えた損失を計算する

    inputs: (N, 9) 入力点群の 9 次元埋め込み
    recon: (N, 9) 再構成点群の 9 次元埋め込み
    z: (N, 2) 潜在表現
    pd: 1 次のパーシステンス図 
    lamb: トポロジカル正則化の重み
    """
    # --- 再構成誤差 (MSE) ---
    mse_loss = nn.MSELoss()(recon, inputs)

    if lamb == 0.0:
        return mse_loss
    
    # --- トポロジカル正則化 ---
    topo_reg = powered_wasserstein_distance_one_sided(
        X=z, 
        ref_pd=[pd],
        dims=[1],
        order=2, 
        grad_type=grad_type,
        sigma=sigma
    )
    assert topo_reg.requires_grad, "topo_reg should require grad"

    total_loss = mse_loss + lamb * topo_reg
    return total_loss

# ============================================================
# メイン: データ生成/読み込み & 学習 & 可視化
# ============================================================

def main(
    N=100, 
    width=32, 
    seed=0, 
    lamb=1.0,
    num_epochs=1000,
    lamb_nonzero_steps=None,
    dataset_name="rings_interlock",
    grad_type="standard",
    sigma=0.1,
) -> None:
    # デバイス設定
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ディレクトリ作成
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs/topoae", exist_ok=True)

    if lamb > 0 and ((lamb_nonzero_steps is None) or (lamb_nonzero_steps > 0)):
        experiment_name = grad_type 
    else:
        experiment_name = "no_topo_reg"

    pointcloud_path = f"data/{dataset_name}.pt"
    rings_img_path = f"data/{dataset_name}.png"
    pd_img_path = "data/rings_interlock_pd.png"
    result_img_path = f"logs/topoae/{experiment_name}.png"
    result_pd_path = f"logs/topoae/{experiment_name}_pd.png"

    # ---------- (1) 点群生成または読み込み ----------
    if os.path.exists(pointcloud_path):
        # 既存ファイルがあればそれを読み込む
        points = torch.load(pointcloud_path, map_location=device)
        # 念のため tensor であること & 型を合わせる
        if not isinstance(points, torch.Tensor):
            raise ValueError(f"{pointcloud_path} のフォーマットが想定外です。")
        points = points.to(device)
        print(f"Loaded 3D point cloud from: {pointcloud_path}")
    else:
        # なければ生成して保存
        points = generate_point_cloud(
            dataset_name=dataset_name,
            N=N, 
            noise_scale=0.1, 
            device=device
        )  # (N, 3)
        torch.save(points.cpu(), pointcloud_path)
        print(f"Saved 3D point cloud to: {pointcloud_path}")

    # 3D 点群を可視化して保存
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    pts_cpu = points.detach().cpu()
    ax.scatter(
        pts_cpu[:, 0].numpy(),
        pts_cpu[:, 1].numpy(),
        pts_cpu[:, 2].numpy(),
        s=20,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Noisy Interlocking Rings (3D)")
    plt.tight_layout()
    plt.savefig(rings_img_path, dpi=200)
    plt.close(fig)
    print(f"Saved 3D point cloud visualization to: {rings_img_path}")

    # compute persistence diagram
    rph = RipsPH(pts_cpu.numpy(), maxdim=1)
    barcode = []
    for dim in [0, 1]:
        barcode += [(dim, (birth, death)) for birth, death in rph.get_barcode(dim)]
    target_pd = rph.get_barcode(dim=1, out_format="torch")

    # パーシステンス図も保存
    fig_pd, ax_pd = plt.subplots(figsize=(6, 6))
    plot_pd_with_specified_lim(
        [barcode],
        [ax_pd],
        high=None,
        titles=["Persistence Diagram of Noisy Rings"],
        x_labels=["Birth"],
        y_labels=["Death"],
    )
    pd_img_path = "data/rings_interlock_pd.png"
    fig_pd.savefig(pd_img_path, dpi=200)
    plt.close(fig_pd)
    print(f"Saved persistence diagram to: {pd_img_path}")

    # ---------- (2) 歪対称行列 -> 行列指数 -> 9 次元埋め込み ----------
    embeddings = skew_matrix_exp_embedding(points)  # (N, 9)

    # --- 訓練シード固定 ---
    set_seed(seed)

    # ---------- DataLoader 準備 ----------
    dataset = TensorDataset(embeddings)
    batch_size = N
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ---------- (3) Autoencoder 準備 ----------
    model = AE9to2(width=width).to(device)

    # optimizer: SGD
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

    # scheduler: 例として StepLR を使用
    # 50 epoch ごとに lr を 0.5 倍
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    # ---------- 学習ループ ----------
    elapsed_time = 0.0
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for (batch_emb,) in dataloader:
            start_time = time()

            # 埋め込み点群・再構成点群
            batch_emb = batch_emb.to(device)
            optimizer.zero_grad()
            recon, z = model(batch_emb)

            # lamb の決定
            if lamb_nonzero_steps is None:
                current_lamb = lamb
            else:
                current_lamb = 0.0 if epoch <= num_epochs - lamb_nonzero_steps else lamb

            # 損失計算と逆伝播
            loss = recon_loss_with_topo_reg(
                inputs=batch_emb,
                recon=recon,
                z=z,
                pd=target_pd,
                lamb=current_lamb,
                grad_type=grad_type, 
                sigma=sigma
            )
            loss.backward()
            optimizer.step()

            if current_lamb > 0.0:
                elapsed_time += time() - start_time

            total_loss += loss.item() * batch_emb.size(0)

        scheduler.step()

        avg_loss = total_loss / len(dataset)
        if epoch % 10 == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch:3d}/{num_epochs}]  "
                f"Loss: {avg_loss:.6f}  LR: {current_lr:.5f}"
            )

    # topological loss の更新にかかった時間を表示
    print(f"Elapsed time for topological loss optimization: {elapsed_time:.2f} seconds")

    # ---------- 学習後の埋め込み取得 ----------
    model.eval()
    with torch.no_grad():
        # z: (N, 2) が 2 次元埋め込み
        _, z = model(embeddings)

    print("Latent embedding shape:", z.shape)  # -> (N, 2)

    # ---------- 埋め込みの可視化 ----------
    if experiment_name != "no_topo_reg":
        embd_plot_title = grad_type
    else:
        embd_plot_title = "No Topological Regularization"
    z_cpu = z.cpu()
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(
        z_cpu[:, 0].numpy(),
        z_cpu[:, 1].numpy(),
        s=15,
    )
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title(embd_plot_title)
    plt.axis("equal")
    plt.tight_layout()

    plt.savefig(result_img_path, dpi=200)
    plt.close(fig)
    print(f"Saved latent embedding visualization to: {result_img_path}")

    # パーシステンス図も保存
    rph = RipsPH(z_cpu.numpy(), maxdim=1)
    barcode = []
    for dim in [0, 1]:
        barcode += [(dim, (birth, death)) for birth, death in rph.get_barcode(dim)]
    fig_pd, ax_pd = plt.subplots(figsize=(6, 6))
    plot_pd_with_specified_lim(
        [barcode],
        [ax_pd],
        high=None,
        titles=["Persistence Diagram of Noisy Rings"],
        x_labels=["Birth"],
        y_labels=["Death"],
    )
    fig_pd.savefig(result_pd_path, dpi=200)
    plt.close(fig_pd)
    print(f"Saved persistence diagram to: {result_pd_path}")


if __name__ == "__main__":
    main(
        N=128, 
        width=32, 
        seed=2,
        lamb=0.1, 
        num_epochs=1000,
        lamb_nonzero_steps=100,
        dataset_name="rings_interlock",
        grad_type="diffeo",
        sigma=0.1
    )

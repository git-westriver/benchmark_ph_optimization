import dataclasses
from typing import Optional
import sys
import os
from typing import Callable
import time
from itertools import product, accumulate
import pickle

from matplotlib import pyplot as plt
import matplotlib.animation as animation

from persistence_based_loss import *
from regularization import *
from data_loader import *
from lib.ph_optimization_library import *

@dataclasses.dataclass
class Configuration:
    exp_name: str = ""
    ### 共通設定 ###
    # データ
    data_func: Callable = circle_with_one_outlier
    # 繰り返し回数
    num_trial: int = 20
    # エポック数
    num_epoch: int = 500
    # ログ表示の間隔
    log_interval: int = 50
    # 時間制限（エポック数を設定しない場合のみ）
    time_limit: Optional[float] = None
    ### 損失関数 ###
    # 損失関数の種類
    loss_obj: PersistenceBasedLoss = ExpandLoss(1, 1)
    regularization_obj: Optional[Regularization] = RectangleRegularization(-2., -2., 2., 2., 1., 2)
    ### 手法関連 ###
    method: str = "gd" # "gd", "bigstep", "continuation"
    lr: float = 1e-1
    reg_proj: bool = False # 正則化項 = 0 となる領域に射影するかどうか
    # for gd and bigstep
    optimizer_conf: dict = dataclasses.field(default_factory=dict)
    scheduler_conf: dict = dataclasses.field(default_factory=dict)
    # for continuation
    num_in_iter: int = 1
    
    def __post_init__(self):
        # 条件による設定の変更
        if (self.num_epoch is not None) and (self.time_limit is not None):
            print("`num_epoch` and time_limit are both set. `time_limit` is ignored.")
            self.time_limit = None
        elif (self.num_epoch is None) and (self.time_limit is None):
            raise ValueError("Either `num_epoch` or `time_limit` must be set.")
    
    def print(self):
        print("===== Configuration =====")
        for k, v in self.__dict__.items():
            print(f"{k}: {v}")
        sys.stdout.flush()

def main(conf: Optional[Configuration] = None):
    if conf is None:
        conf = Configuration()
    conf.print()
    if len(sys.argv) >= 2:
        savedirpath = sys.argv[1]
    else:
        savedirpath = "results/ph_optimization/sample"
    if conf.exp_name != "":
        savedirpath += f"/{conf.exp_name}"
        if not os.path.exists(savedirpath):
            os.makedirs(savedirpath)
    ### データの読み取り ###
    dataset = get_data(conf.data_func, 100)
    ### num_trial 個の初期データに対して最適化を行う ###
    loss_history_list: list[list[float]] = []; time_history_list: list[list[float]] = []
    for trial in range(conf.num_trial):
        trial_start = time.time()
        X = torch.tensor(dataset[trial], dtype=torch.float32, requires_grad=True)
        if conf.method == "gd":
            poh = GradientDescent(X, conf.loss_obj, conf.regularization_obj, reg_proj=conf.reg_proj, 
                                  lr=conf.lr, optimizer_conf=conf.optimizer_conf, scheduler_conf=conf.scheduler_conf)
        elif conf.method == "bigstep":
            poh = BigStep(X, conf.loss_obj, conf.regularization_obj, reg_proj=conf.reg_proj, 
                          lr=conf.lr, optimizer_conf=conf.optimizer_conf, scheduler_conf=conf.scheduler_conf)
        elif conf.method == "continuation":
            poh = Continuation(X, conf.loss_obj, conf.regularization_obj, 
                               lr=conf.lr, in_iter_num=conf.num_in_iter)
        else:
            raise NotImplementedError
        X_history = []; loss_history = []; time_history = [0]
        epoch = -1; ellapsed_time = 0
        while ((conf.num_epoch is not None and epoch < conf.num_epoch - 1) 
                or (conf.time_limit is not None and ellapsed_time < conf.time_limit)):
            epoch += 1; ellapsed_time += time_history[-1]
            # 損失の計算 + 記録 + 出力
            loss = poh.get_loss()
            loss_history.append(loss.item())
            X_history.append(poh.X.detach().clone())
            if epoch % conf.log_interval == 0:
                print(f"epoch: {epoch}, loss: {loss.item()}", flush=True)
            # 変数の更新 + 時間計測
            start = time.time()
            poh.update()
            time_history.append(time.time() - start)
        loss = poh.get_loss()
        loss_history.append(loss.item())
        X_history.append(poh.X.detach().clone())
        print(f"Final loss: {loss.item()}")
        ## time_history を累積和に変換 ##
        time_history = list(accumulate(time_history))
        ## loss_history, time_history を loss_history_list, time_history_list に追加 ##
        loss_history_list.append(loss_history)
        time_history_list.append(time_history)
        ## trial = 0 の場合は X_history を ndarray と gif で保存 ##
        if trial == 0:
            with open(f"{savedirpath}/X_history.pkl", "wb") as f:
                pickle.dump(X_history, f)
            X_history = torch.stack(X_history, axis=0).numpy()
            xmin, xmax = np.min(X_history[:, :, 0]), np.max(X_history[:, :, 0])
            ymin, ymax = np.min(X_history[:, :, 1]), np.max(X_history[:, :, 1])
            fig = plt.figure(); ax = fig.add_subplot(111)
            sc = ax.scatter([], [], color='#377eb8') # 空の散布図
            ax.set_aspect("equal"); ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
            def pc_update(i):
                sc.set_offsets(X_history[i, :, :])
                return sc, 
            anim = animation.FuncAnimation(fig, pc_update, frames=X_history.shape[0], interval=100)
            anim.save(f"{savedirpath}/X_history.gif", writer='pillow')
        print(f"Trial {trial} finished. ellapsed time: {time.time() - trial_start}", flush=True)

    ### 結果の可視化・保存 ###
    ## epoch - loss の可視化 ##
    if conf.num_epoch is None: # エポックとロスの散布図
        fig = plt.figure(); ax = fig.add_subplot(111)
        for trial in range(conf.num_trial):
            ax.scatter(np.arange(len(loss_history_list[trial])), loss_history_list[trial], color="blue", alpha=0.3)
        ax.set_xlabel("epoch"); ax.set_ylabel("loss")
        fig.savefig(f"{savedirpath}/epoch-loss.png")
    else:
        loss_history_mat = np.stack(loss_history_list, axis=0) # (num_trial, num_epoch+1)
        loss_mean = np.mean(loss_history_mat, axis=0)
        loss_std = np.std(loss_history_mat, axis=0)
        fig = plt.figure(); ax = fig.add_subplot(111)
        ax.plot(loss_mean, color="blue")
        ax.fill_between(np.arange(loss_mean.shape[0]), loss_mean - loss_std, loss_mean + loss_std, color="blue", alpha=0.3)
        ax.set_xlabel("epoch"); ax.set_ylabel("loss")
        fig.savefig(f"{savedirpath}/epoch-loss.png")
    ## time - loss の可視化 ##
    if conf.time_limit is None: # 時間とロスの散布図
        fig = plt.figure(); ax = fig.add_subplot(111)
        for trial in range(conf.num_trial):
            ax.scatter(time_history_list[trial], loss_history_list[trial], color="blue", alpha=0.3)
        ax.set_xlabel("time"); ax.set_ylabel("loss")
        fig.savefig(f"{savedirpath}/time-loss.png")
    else:
        time_linspace = np.linspace(0, conf.time_limit, 101)
        time_loss_list = [[loss_history_list[trial][0]] for trial in range(conf.num_trial)]
        for trial in range(conf.num_trial):
            # 時刻 t のときの loss は，t より前の最後の loss とする
            # 注： time_history_list[trial][cur_step] は cur_epoch 終了後，loss_history_list[trial][cur_step] は開始前
            cur_epoch = 0 
            for t in time_linspace[1:]:
                while cur_epoch < conf.num_epoch - 1 and time_history_list[trial][cur_epoch] < t:
                    cur_epoch += 1
                time_loss_list[trial].append(loss_history_list[trial][cur_epoch])
        time_loss_mat = np.stack(time_loss_list, axis=0) # (num_trial + 1, 101)
        time_loss_mean = np.mean(time_loss_mat, axis=0)
        time_loss_std = np.std(time_loss_mat, axis=0)
        fig = plt.figure(); ax = fig.add_subplot(111)
        ax.plot(time_linspace, time_loss_mean, color="blue")
        ax.fill_between(time_linspace, time_loss_mean - time_loss_std, time_loss_mean + time_loss_std, color="blue", alpha=0.3)
        ax.set_xlabel("time"); ax.set_ylabel("loss")
        fig.savefig(f"{savedirpath}/time-loss.png")
    # 結果を pickle で保存
    result_dict = {
        "loss_history": loss_history_list,
        "time_history": time_history_list,
    }
    with open(f"{savedirpath}/result_dict.pkl", "wb") as f:
        pickle.dump(result_dict, f)
    
if __name__ == "__main__":
    data_type_list = ["circle-w-one-outlier", "voronoi"]
    method_list = ["bigstep"]
    lr_list = [(4**i) * 1e-3 for i in range(5)]
    scheduler_list = ["const"]
    idx = -1
    for data_type, method, lr, scheduler in product(data_type_list, method_list, lr_list, scheduler_list):
        idx += 1
        main(Configuration({
            "data_type": data_type, 
            "num_epoch": 100, 
            "log_interval": 50,
            "loss_order": 1, 
            "reg_proj": True,
            "method": method,
            "lr": lr, 
            "scheduler_conf": {"name": scheduler}, 
            "exp_name": f"exp-{str(idx).zfill(3)}"}))
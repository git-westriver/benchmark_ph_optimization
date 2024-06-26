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
class OptConfig:
    """
    Configuration for persistence homology optimization.

    Parameters:
        - COMMON SETTINGS
            - exp_name(str, default=""): Experiment name. This will be used as a directory name to save the results.
            - save_dirpath(str, default="results/sample"): Directory path to save the results.
            - data_func(Callable, default=circle_with_one_outlier): Data generating function. You can define your own function in `data_loader.py`.
            - num_trial(int, default=1): If you want to perform the optimization multiple times with different initial values and see the average results, set this parameter.
            - num_epoch(int, default=None): Number of epochs. If `None`, the optimization is performed until `time_limit`.
            - time_limit(Optional[float], default=None): Time limit. If `None`, the optimization is performed until `num_epoch`.
            - log_interval(int, default=10): The logs (for example, loss value) are printed every `log_interval` epochs.
        - LOSS FUNCTION
            - loss_obj(PersistenceBasedLoss, default=ExpandLoss([1], 1), topk=1): 
                Object that determines the loss function. You can define your own function in `persistence_based_loss.py`.
            - regularization_obj(Optional[Regularization], default=RectangleRegularization(-2., -2., 2., 2., 1., 2)): 
                Regularization. You can define your own function in `regularization.py`.
        - METHOD
            - method(str, default="gd"): Optimization method. "gd", "bigstep", "continuation" and "diffeo" are available.
            - lr(float, default=1e-1): Learning rate.
            - reg_proj(bool, default=False): 
                If `True`, the algorithm projects the variables to the region where the regularization term is zero at the end of each epoch.
            - optimizer_conf(dict, default={}): Configuration for the optimizer used in "gd" and "bigstep". You can specify the following keys:
                - "name"(str, default="SGD"): Name of the optimizer. You can choose from "SGD" and "Adam".
            - scheduler_conf(dict, default={}): Configuration for the scheduler. You can specify the following keys:
                - "name"(str, default="const"): Name of the scheduler. You can choose from "const" and "TransformerLR".
            - num_in_iter(int, default=1): Number of iterations in the continuation method.
    """
    ### COMMON SETTINGS ###
    exp_name: str = ""
    save_dirpath: str = f"results/sample"
    data_func: Callable = circle_with_one_outlier
    num_trial: int = 1
    num_epoch: Optional[int] = None
    time_limit: Optional[float] = None
    log_interval: int = 10
    ### LOSS FUNCTION ###
    loss_obj: PersistenceBasedLoss = ExpandLoss([1], 1, topk=1)
    regularization_obj: Optional[Regularization] = RectangleRegularization(-2., -2., 2., 2., 1., 2)
    ### METHOD ###
    method: str = "gd" # "gd", "bigstep", "continuation", "diffeo"
    lr: float = 1e-1
    reg_proj: bool = False
    optimizer_conf: dict = dataclasses.field(default_factory=dict) # for gd and bigstep
    scheduler_conf: dict = dataclasses.field(default_factory=dict) # for gd and bigstep
    num_in_iter: int = 1 # for continuation
    
    def __post_init__(self):
        # Check if `num_epoch` and `time_limit` are set correctly.
        if (self.num_epoch is not None) and (self.time_limit is not None):
            print("`num_epoch` and time_limit are both set. `time_limit` is ignored.")
            self.time_limit = None
        elif (self.num_epoch is None) and (self.time_limit is None):
            raise ValueError("Either `num_epoch` or `time_limit` must be set.")
    
    def print(self):
        print("===== Configuration =====")
        for k, v in self.__dict__.items():
            if "__name__" in dir(v):
                print(f"{k}: {v.__name__}")
            else:
                print(f"{k}: {v}")
        sys.stdout.flush()

def ph_opt_main(conf: Optional[OptConfig] = None):
    if conf is None:
        conf = OptConfig()
    conf.print()
    ### Create a directory if it does not exist ###
    savedirname = f"{conf.save_dirpath}/{conf.exp_name}" if conf.exp_name != "" else conf.save_dirpath
    if not os.path.exists(savedirname):
        os.makedirs(savedirname)
    ### Read data ###
    dataset = get_data(conf.data_func, 100)
    ### Optimization for `num_trial` different initial values ###
    loss_history_list: list[list[float]] = []; time_history_list: list[list[float]] = []
    for trial in range(conf.num_trial):
        print(f"--- Trial {trial} ---")
        trial_start = time.time()
        ## Initialize the variables and the optimization method ##
        X = torch.tensor(dataset[trial], dtype=torch.float32, requires_grad=True)
        if conf.method == "gd":
            poh = GradientDescent(X, conf.loss_obj, conf.regularization_obj, reg_proj=conf.reg_proj, 
                                  lr=conf.lr, optimizer_conf=conf.optimizer_conf, scheduler_conf=conf.scheduler_conf)
        elif conf.method == "bigstep":
            poh = BigStep(X, conf.loss_obj, conf.regularization_obj, reg_proj=conf.reg_proj, 
                          lr=conf.lr, optimizer_conf=conf.optimizer_conf, scheduler_conf=conf.scheduler_conf)
        elif conf.method == "diffeo":
            poh = Diffeo(X, conf.loss_obj, conf.regularization_obj, reg_proj=conf.reg_proj, 
                        lr=conf.lr, optimizer_conf=conf.optimizer_conf, scheduler_conf=conf.scheduler_conf)
        elif conf.method == "continuation":
            poh = Continuation(X, conf.loss_obj, conf.regularization_obj, 
                               lr=conf.lr, in_iter_num=conf.num_in_iter)
        else:
            raise NotImplementedError
        ## Optimization ##
        X_history = []; loss_history = []; time_history = [0]
        epoch = -1; ellapsed_time = 0
        while ((conf.num_epoch is not None and epoch < conf.num_epoch - 1) 
                or (conf.time_limit is not None and ellapsed_time < conf.time_limit)):
            epoch += 1; ellapsed_time += time_history[-1]
            # Compute loss, record it, and output it to the console
            loss = poh.get_loss()
            loss_history.append(loss.item())
            X_history.append(poh.X.detach().clone())
            if epoch % conf.log_interval == 0:
                print(f"epoch: {epoch}, loss: {loss.item()}", flush=True)
            # update the variables with the specified method and measure the time of epoch
            start = time.time()
            poh.update()
            time_history.append(time.time() - start)
        loss = poh.get_loss()
        loss_history.append(loss.item())
        X_history.append(poh.X.detach().clone())
        print(f"Final loss: {loss.item()}")
        ## Finish the optimization and record the results ##
        # convert time_history to cumulative time
        time_history = list(accumulate(time_history))
        # add loss_history, time_history to the list
        loss_history_list.append(loss_history)
        time_history_list.append(time_history)
        ## For the first trial, save X_history as ndarray and gif ##
        if trial == 0:
            with open(f"{savedirname}/X_history.pkl", "wb") as f:
                pickle.dump(X_history, f)
            X_history = torch.stack(X_history, axis=0).numpy()
            xmin, xmax = np.min(X_history[:, :, 0]), np.max(X_history[:, :, 0])
            ymin, ymax = np.min(X_history[:, :, 1]), np.max(X_history[:, :, 1])
            fig = plt.figure(); ax = fig.add_subplot(111)
            sc = ax.scatter([], [], color='#377eb8')
            ax.set_aspect("equal"); ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
            def pc_update(i):
                sc.set_offsets(X_history[i, :, :])
                return sc, 
            anim = animation.FuncAnimation(fig, pc_update, frames=X_history.shape[0], interval=100)
            anim.save(f"{savedirname}/X_history.gif", writer='pillow')
        ## Finish the trial ##
        print(f"Trial {trial} finished. ellapsed time: {time.time() - trial_start}", flush=True)

    ### Visualization and saving the results ###
    ## visualization of the transition of the loss over epochs ##
    if conf.num_epoch is None: 
        # Then, scatter between epoch and loss
        fig = plt.figure(); ax = fig.add_subplot(111)
        for trial in range(conf.num_trial):
            ax.scatter(np.arange(len(loss_history_list[trial])), loss_history_list[trial], color="blue", alpha=0.3)
        ax.set_xlabel("epoch"); ax.set_ylabel("loss")
        fig.savefig(f"{savedirname}/epoch-loss.png")
    else: 
        # Then, mean and std of the loss over trials
        loss_history_mat = np.stack(loss_history_list, axis=0) # (num_trial, num_epoch+1)
        loss_mean = np.mean(loss_history_mat, axis=0)
        loss_std = np.std(loss_history_mat, axis=0)
        fig = plt.figure(); ax = fig.add_subplot(111)
        ax.plot(loss_mean, color="blue")
        ax.fill_between(np.arange(loss_mean.shape[0]), loss_mean - loss_std, loss_mean + loss_std, color="blue", alpha=0.3)
        ax.set_xlabel("epoch"); ax.set_ylabel("loss")
        fig.savefig(f"{savedirname}/epoch-loss.png")
    ## visualization of the transition of the loss over time ##
    if conf.time_limit is None: 
        # Then, scatter between time and loss
        fig = plt.figure(); ax = fig.add_subplot(111)
        for trial in range(conf.num_trial):
            ax.scatter(time_history_list[trial], loss_history_list[trial], color="blue", alpha=0.3)
        ax.set_xlabel("time"); ax.set_ylabel("loss")
        fig.savefig(f"{savedirname}/time-loss.png")
    else: 
        # Then, mean and std of the loss over trials
        time_linspace = np.linspace(0, conf.time_limit, 101)
        time_loss_list = [[loss_history_list[trial][0]] for trial in range(conf.num_trial)]
        for trial in range(conf.num_trial):
            # loss value at time t is the last loss value before t
            # Note: loss_history_list[trial][cur_epoch] is after cur_epoch, loss_history_list[trial][cur_epoch-1] is before cur_epoch
            cur_epoch = 0 
            for t in time_linspace[1:]:
                while cur_epoch < len(loss_history_list[trial]) - 1 and time_history_list[trial][cur_epoch] < t:
                    cur_epoch += 1
                time_loss_list[trial].append(loss_history_list[trial][cur_epoch])
        time_loss_mat = np.stack(time_loss_list, axis=0) # (num_trial + 1, 101)
        time_loss_mean = np.mean(time_loss_mat, axis=0)
        time_loss_std = np.std(time_loss_mat, axis=0)
        fig = plt.figure(); ax = fig.add_subplot(111)
        ax.plot(time_linspace, time_loss_mean, color="blue")
        ax.fill_between(time_linspace, time_loss_mean - time_loss_std, time_loss_mean + time_loss_std, color="blue", alpha=0.3)
        ax.set_xlabel("time"); ax.set_ylabel("loss")
        fig.savefig(f"{savedirname}/time-loss.png")
    ## save loss_history_list and time_history_list ##
    result_dict = {
        "loss_history": loss_history_list,
        "time_history": time_history_list,
    }
    with open(f"{savedirname}/result_dict.pkl", "wb") as f:
        pickle.dump(result_dict, f)
    
if __name__ == "__main__":
    method_list = ["gd", "continuation", "bigstep", "diffeo"]
    lr_list = [(4**i) * 1e-3 for i in range(6)]
    for method in method_list:
        for lr in lr_list:
            config = OptConfig(exp_name=f"{method}_lr={lr:.3f}", method=method, lr=lr, num_epoch=100)
            ph_opt_main(config)
import dataclasses
from typing import Optional, Type
import sys
import os
from typing import Callable
import time
from itertools import accumulate
import pickle
from matplotlib import pyplot as plt
from pathlib import Path
import torch
import numpy as np
from functools import partial

from ph_opt import (
    PersistenceBasedLoss, Regularization,
    PHOptimization, GradientDescent, BigStep, Continuation, Diffeo, 
    get_animation
)
from ph_opt.data import circle_with_one_outlier, get_data

@dataclasses.dataclass
class PHTrainerConfig:
    """
    Configuration class for PHTrainer.

    This class encapsulates all the necessary parameters for training using
    persistent homology-based optimization methods. It includes loss function
    definitions, data loading strategy, optimization hyperparameters, and logging behavior.

    Attributes:
        loss_obj (PersistenceBasedLoss | Callable): Loss function or callable object used for optimization.
        regularization_obj (Optional[Regularization]): Optional regularization term to be added to the loss.
        exp_name (str): Name of the experiment, used for naming output directories and files.
        save_dirpath (str | Path): Path to directory where results and logs will be saved.
        data_source (Callable | str | Path): Function or path that provides the training data.
        init_strategy (Optional[Callable]): Initialization function for input variables.
        num_trial (int): Number of trials to run with different initializations.
        num_epoch (Optional[int]): Number of training epochs. Required if time_limit is not set.
        time_limit (Optional[float]): Time limit for training. Required if num_epoch is not set.
        log_interval (int): Interval of epochs at which logs are printed.
        method (str): Optimization method. One of "gd", "bigstep", "continuation", "diffeo".
        lr (float): Learning rate.
        reg_proj (bool): Whether to apply projection in the presence of regularization.
        optimizer_class (Type[torch.optim.Optimizer]): Optimizer class to use (e.g., SGD).
        optimizer_config (dict): Additional configuration for the optimizer.
        scheduler_class (Optional[Type[torch.optim.lr_scheduler._LRScheduler]]): Optional learning rate scheduler class.
        scheduler_config (dict): Additional configuration for the scheduler.
        num_in_iter (int): Number of inner iterations used in continuation methods.

    Raises:
        ValueError: If neither `num_epoch` nor `time_limit` is set.
    """
    ### LOSS FUNCTION ###
    loss_obj: PersistenceBasedLoss | Callable
    regularization_obj: Optional[Regularization] = None

    ### COMMON SETTINGS ###
    exp_name: str = ""
    save_dirpath: str | Path = f"tmp/"
    data_source: Callable | str | Path = circle_with_one_outlier
    init_strategy: Optional[Callable] = None
    num_trial: int = 1
    num_epoch: Optional[int] = None
    time_limit: Optional[float] = None
    log_interval: int = 10

    ### METHOD ###
    method: str = "gd" # "gd", "bigstep", "continuation", "diffeo"
    lr: float = 1e-1
    reg_proj: bool = False
    optimizer_class: Type[torch.optim.Optimizer] = torch.optim.SGD
    optimizer_config: dict = dataclasses.field(default_factory=dict)
    scheduler_class: Optional[Type[torch.optim.lr_scheduler._LRScheduler]] = None
    scheduler_config: dict = dataclasses.field(default_factory=dict)
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
            if isinstance(v, PersistenceBasedLoss) or isinstance(v, Regularization):
                print(f"{k}: {v.__class__.__name__}")
            elif isinstance(v, partial):
                print(f"{k}: {v.func.__name__}({', '.join([f'{k}={v}' for k, v in v.keywords.items()])})")
            elif callable(v):
                print(f"{k}: {v.__name__}")
            else:
                print(f"{k}: {v}")
        sys.stdout.flush()

class PHTrainer:
    """
    Persistent Homology Trainer for point cloud optimization.

    This class orchestrates training with persistent homology-based optimization methods
    using the configuration provided in PHTrainerConfig. It handles dataset preparation,
    initialization, optimization, logging, and visualization.

    Executes the training process based on the given configuration by calling`train` method.
    Supports multiple trials and saves visualizations, model histories, and logs to disk.

    Args:
        config (PHTrainerConfig): Configuration object specifying training parameters.
        scatter_config (Optional[dict]): Configuration for scatter plot visualization.
        viz_dims (Optional[list[int]]): List of dimensions to use for visualization. This have to be set if config.loss_obj is not a PersistenceBasedLoss.
        save_dirpath (Path | str): Directory to save logs, models, and visualizations.
    """

    def __init__(self, config: Optional[PHTrainerConfig] = None, scatter_config: Optional[dict] = None, viz_dims: Optional[list[int]] = None):
        if config is None:
            config = PHTrainerConfig()
        config.print()

        self.config = config
        self.scatter_config = scatter_config

        if viz_dims is None:
            self.viz_dims = config.loss_obj.dim_list
        else:
            self.viz_dims = viz_dims

        self.save_dirpath =Path(config.save_dirpath) / self.config.exp_name if self.config.exp_name != "" else Path(self.save_dirpath)

    def train(self):
        # Create a directory if it does not exist
        self._maybe_create_dir()

        # Read data
        dataset = self.read_data()

        # Optimization for `num_trial` different initial values
        loss_history_list: list[list[float]] = []
        time_history_list: list[list[float]] = []
        for trial in range(self.config.num_trial):
            print(f"--- Trial {trial} ---")
            trial_start = time.time()

            # Initialize the variable
            X = self.initialize_variable(dataset, trial)

            if isinstance(self.config.loss_obj, PersistenceBasedLoss):
                # initialize the optimization method
                poh = self.initialize_method(X)
            
                # Optimization
                X_history, loss_history, time_history = self.optimize_persistence_based_loss(poh)

            elif isinstance(self.config.loss_obj, Callable):
                # Initialize the optimization method
                _method = self.config.method if self.config.method != "gd" else "standard"
                loss_func = partial(self.config.loss_obj, grad_type=_method)

                # Optimization
                X_history, loss_history, time_history = self.optimize_with_gradient_based_interface(loss_func, X)

            else:
                raise ValueError("loss_obj must be a PersistenceBasedLoss or Callable.")

            # add loss_history, time_history to the list
            loss_history_list.append(loss_history)
            time_history_list.append(time_history)
            
            # Finish the trial
            print(f"Trial {trial} finished. elapsed time: {time.time() - trial_start}", flush=True)

            # For the first trial, save X_history and create a gif
            if trial == 0:
                print("Saving results...", flush=True, end=" ")
                # save X_history
                with open(self.save_dirpath / "X_history.pkl", "wb") as f:
                    pickle.dump(X_history, f)

                # create a gif
                anim = get_animation([X_history], [[loss_history]], 
                                     dim_list=self.viz_dims,
                                     title_list=[self.config.method], 
                                     vertical=False, 
                                     scatter_config=self.scatter_config)
                anim.save(self.save_dirpath / "trajectory.gif", writer='pillow')

                print("Done", flush=True)

        # save loss_history_list and time_history_list
        result_dict = {
            "loss_history": loss_history_list,
            "time_history": time_history_list,
        }
        with open(self.save_dirpath / "result_dict.pkl", "wb") as f:
            pickle.dump(result_dict, f)
    
    def _maybe_create_dir(self):
        if not os.path.exists(self.save_dirpath):
            os.makedirs(self.save_dirpath)

    def read_data(self):
        if callable(self.config.data_source):
            dataset = get_data(self.config.data_source, 100)
        elif isinstance(self.config.data_source, str) or isinstance(self.config.data_source, Path):
            data = np.load(self.config.data_source)
            dataset = [data for _ in range(self.config.num_trial)]
        else:
            raise ValueError("data_source must be a function, str, or Path.")
        return dataset
    
    def initialize_variable(self, dataset, trial):
        if self.config.init_strategy is not None:
            X = self.config.init_strategy(dataset[trial])
        else:
            X = torch.from_numpy(dataset[trial])
        X = X.float().requires_grad_()
        return X
    
    def initialize_method(self, X) -> PHOptimization:
        if self.config.method == "gd":
            poh = GradientDescent(X, self.config.loss_obj, self.config.regularization_obj, reg_proj=self.config.reg_proj, 
                                  lr=self.config.lr, optimizer_conf=self.config.optimizer_conf, scheduler_conf=self.config.scheduler_conf)
        elif self.config.method == "bigstep":
            poh = BigStep(X, self.config.loss_obj, self.config.regularization_obj, reg_proj=self.config.reg_proj, 
                          lr=self.config.lr, optimizer_conf=self.config.optimizer_conf, scheduler_conf=self.config.scheduler_conf)
        elif self.config.method == "diffeo":
            poh = Diffeo(X, self.config.loss_obj, self.config.regularization_obj, reg_proj=self.config.reg_proj, 
                        lr=self.config.lr, optimizer_conf=self.config.optimizer_conf, scheduler_conf=self.config.scheduler_conf)
        elif self.config.method == "continuation":
            poh = Continuation(X, self.config.loss_obj, self.config.regularization_obj, 
                               lr=self.config.lr, in_iter_num=self.config.num_in_iter)
        else:
            raise NotImplementedError
        return poh
    
    def optimize_persistence_based_loss(self, poh: PHOptimization) -> tuple[list[torch.Tensor], list[float], list[float]]:
        X_history = []
        loss_history = []
        time_history = [0]
        epoch = -1
        elapsed_time = 0
        while ((self.config.num_epoch is not None and epoch < self.config.num_epoch - 1)
                or (self.config.time_limit is not None and elapsed_time < self.config.time_limit)):
            epoch += 1
            elapsed_time += time_history[-1]

            # Compute loss, record it, and output it to the console
            loss = poh.get_loss()
            loss_history.append(loss.item())
            X_history.append(poh.X.detach().clone())
            if epoch % self.config.log_interval == 0:
                print(f"epoch: {epoch}, loss: {loss.item()}", flush=True)

            # update the variables with the specified method and measure the time of epoch
            start = time.time()
            poh.update()
            time_history.append(time.time() - start)

        loss = poh.get_loss()
        loss_history.append(loss.item())
        X_history.append(poh.X.detach().clone())
        print(f"Final loss: {loss.item()}", flush=True)

        # convert time_history to cumulative time
        time_history = list(accumulate(time_history))

        return X_history, loss_history, time_history
    
    def optimize_with_gradient_based_interface(self, loss_func: Callable, X: torch.Tensor) -> tuple[list[torch.Tensor], list[float], list[float]]:
        # Initialize optimizer and scheduler
        optimizer = self.config.optimizer_class([X], lr=self.config.lr, **self.config.optimizer_config)
        if self.config.scheduler_class is not None:
            scheduler = self.config.scheduler_class(optimizer, **self.config.scheduler_config) if self.config.scheduler_class is not None else None
        else:
            scheduler = None

        # Initialize some variables
        X_history = []
        loss_history = []
        time_history = [0]
        epoch = -1
        elapsed_time = 0

        # Optimization
        while ((self.config.num_epoch is not None and epoch < self.config.num_epoch - 1)
                or (self.config.time_limit is not None and elapsed_time < self.config.time_limit)):
            epoch += 1
            elapsed_time += time_history[-1]

            # record the current X
            X_history.append(X.detach().clone())

            # reset the gradient
            optimizer.zero_grad()

            # compute the loss and update the variables
            start = time.time()
            loss = loss_func(X)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            time_history.append(time.time() - start)

            # record loss and output it to the console
            loss_history.append(loss.item())
            if epoch % self.config.log_interval == 0:
                print(f"epoch: {epoch}, loss: {loss.item()}", flush=True)
            
        loss = loss_func(X)
        loss_history.append(loss.item())
        X_history.append(X.detach().clone())
        print(f"Final loss: {loss.item()}", flush=True)

        # convert time_history to cumulative time
        time_history = list(accumulate(time_history))

        return X_history, loss_history, time_history
    
    def visualize_loss(self, loss_history_list: list[list[float]], time_history_list: list[list[float]]):
        ## visualization of the transition of the loss over epochs ##
        if self.config.num_epoch is None: 
            # Then, scatter between epoch and loss
            fig = plt.figure(dpi=300)
            ax = fig.add_subplot(111)
            for trial in range(self.config.num_trial):
                ax.scatter(np.arange(len(loss_history_list[trial])), loss_history_list[trial], color="blue", alpha=0.3)
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            fig.savefig(self.save_dirpath / "epoch-loss.png")
        else:
            # Then, mean and std of the loss over trials
            loss_history_mat = np.stack(loss_history_list, axis=0) # (num_trial, num_epoch+1)
            loss_mean = np.mean(loss_history_mat, axis=0)
            loss_std = np.std(loss_history_mat, axis=0)
            fig = plt.figure(dpi=300)
            ax = fig.add_subplot(111)
            ax.plot(loss_mean, color="blue")
            ax.fill_between(np.arange(loss_mean.shape[0]), loss_mean - loss_std, loss_mean + loss_std, color="blue", alpha=0.3)
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            fig.savefig(self.save_dirpath /"epoch-loss.png")

        ## visualization of the transition of the loss over time ##
        if self.config.time_limit is None: 
            # Then, scatter between time and loss
            fig = plt.figure(dpi=300)
            ax = fig.add_subplot(111)
            for trial in range(self.config.num_trial):
                ax.scatter(time_history_list[trial], loss_history_list[trial], color="blue", alpha=0.3)
            ax.set_xlabel("time")
            ax.set_ylabel("loss")
            fig.savefig(f"{self.save_dirpath}/time-loss.png")
        else: 
            # Then, mean and std of the loss over trials
            time_linspace = np.linspace(0, self.config.time_limit, 101)
            time_loss_list = [[loss_history_list[trial][0]] for trial in range(self.config.num_trial)]
            for trial in range(self.config.num_trial):
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
            fig = plt.figure(dpi=300)
            ax = fig.add_subplot(111)
            ax.plot(time_linspace, time_loss_mean, color="blue")
            ax.fill_between(time_linspace, time_loss_mean - time_loss_std, time_loss_mean + time_loss_std, color="blue", alpha=0.3)
            ax.set_xlabel("time")
            ax.set_ylabel("loss")
            fig.savefig(self.save_dirpath / "time-loss.png")
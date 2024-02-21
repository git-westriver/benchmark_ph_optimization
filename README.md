# Benchmarks for optimization of pointclouds with persistent based loss

By Naoki Nishikawa ([GitHub](https://github.com/git-westriver), [Homepage](https://sites.google.com/view/n-nishikawa))
and Yuichi Ike ([Homepage](https://sites.google.com/view/yuichi-ike))

This repository presents the benchmarks for the optimization of point clouds with persistent based loss.
We aim to compare the exisiting methods to optimize persistence based loss when we use Vietoris Rips filtration of point clouds.

Our implementation has following nice features:
- ***Multiple methods***: standard gradient descent, Conginuation [1], and Big Step [2].
- ***High flexibility***: you can specify any point clouds, persistence based losses or regularizations, **without changing the implementation of the algorithms**.
- ***Fast computation***: basically, persistence homology will be computed by `giotto-ph` [3]. 
If we need matrix decomposition, we use our fast implementation inspired by `ripser` [4].

[TODO] Add some figures to demonstrate how the benchmark works.

## Usage of `scripts/ph_optimization.py`

You can simply run the command scripts/ph_optimization.py to try it out.
Then, given a point cloud data with a circle and one outlier, the algorithms try to make the hole larger.
Three algorithms will be run with multiple learning rates, and the results will be saved in the directory `results/sample`.

You can also specify your favorite settings with the attributes in `Configuration` whose parameters are described in the following.
For more details on how to use `Configuration`, please refer to the part below `if __name__ == "__main__":` in scripts/ph_optimization.py.
Regarding data, loss functions and regularizations, the subsequent sections will clarify how to design them.

Parameters of `Configuration`:
- COMMON SETTINGS
    - exp_name(str, default=""): Experiment name. This will be used as a directory name to save the results.
    - data_func(Callable, default=circle_with_one_outlier): Data generating function. You can define your own function in `data_loader.py`.
    - num_trial(int, default=1): If you want to perform the optimization multiple times with different initial values and see the average results, set this parameter.
    - num_epoch(int, default=100): Number of epochs. If `None`, the optimization is performed until `time_limit`.
    - time_limit(Optional[float], default=None): Time limit. If `None`, the optimization is performed until `num_epoch`.
    - log_interval(int, default=10): The logs (for example, loss value) are printed every `log_interval` epochs.
- LOSS FUNCTION
    - loss_obj(PersistenceBasedLoss, default=ExpandLoss(1, 1), topk=1): 
        Object that determines the loss function. You can define your own function in `persistence_based_loss.py`.
    - regularization_obj(Optional[Regularization], default=RectangleRegularization(-2., -2., 2., 2., 1., 2)): 
        Regularization. You can define your own function in `regularization.py`.
- METHOD
    - method(str, default="gd"): Optimization method. "gd", "bigstep", or "continuation".
    - lr(float, default=1e-1): Learning rate.
    - reg_proj(bool, default=False): 
        If `True`, the algorithm projects the variables to the region where the regularization term is zero at the end of each epoch.
    - optimizer_conf(dict, default={}): Configuration for the optimizer used in "gd" and "bigstep". You can specify the following keys:
        - "name"(str, default="SGD"): Name of the optimizer. You can choose from "SGD" and "Adam".
    - scheduler_conf(dict, default={}): Configuration for the scheduler. You can specify the following keys:
        - "name"(str, default="const"): Name of the scheduler. You can choose from "const" and "TransformerLR".
    - num_in_iter(int, default=1): Number of iterations in the continuation method.

## Data

## Persistence-based loss functions

## Regularizations

## References

[1] Marcio Gameiro, Yasuaki Hiraoka, and Ippei Obayashi. Continuation of point clouds via persistence diagrams. Physica D: Nonlinear Phenomena, 334:118–132, 2016.

[2] Arnur Nigmetov and Dmitriy Morozov. Topological optimization with big steps. arXiv:2203.16748, 2022.

[3] 

[4] 
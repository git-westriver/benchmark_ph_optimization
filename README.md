# Library for optimization of pointclouds with persistence-based loss

By Naoki Nishikawa ([GitHub](https://github.com/git-westriver), [Homepage](https://sites.google.com/view/n-nishikawa))
and Yuichi Ike ([Homepage](https://sites.google.com/view/yuichi-ike))

This repository presents a library for the optimization of point clouds with persistence-based loss.
We aim to provide the implementation of the existing optimization methods and the interface to compare them.

Our implementation has following nice features:
- ***Multiple methods***: standard gradient descent, Conginuation [1], Big Step [2] and Diffeo [3].
- ***High flexibility***: you can specify any point clouds, persistence based losses or regularizations, **without changing the implementation of the algorithms**.
- ***Fast computation***: basically, persistence homology will be computed by `giotto-ph` [4]. 
If we need matrix decomposition, we use our fast implementation inspired by [5].

## Demo

![animation](https://github.com/git-westriver/benchmark_ph_optimization/assets/64912615/3544b12f-b9f9-4d85-90c4-eae94d77e481)

## How to start

1. Install the required packages:  <!-- We mainly require the following:-->
    - python(-3.9.13)
    - giotto-ph(-0.2.2)
    - numpy(-1.24.3)
    - scikit-learn(1.4.1.post1)
    - scipy(-1.10.1)
    - pybind11(-2.6.1)
    - pytorch(-2.2.2)
    - matplotlib(-3.8.0)
    - gudhi(-3.9.0)
    - pot(-0.9.3)
    - ipykernel(-6.25.0)
    - eagerpy(-0.29.0)
    
    <!-- Please refer to `ph_opt_public.yaml` for the detailed package requirements. -->

2. Run `cd scripts/lib; g++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) ph_cpp_library_pybind.cpp -o rips_cpp$(python3-config --extension-suffix); cd ../..` to compile the C++ library for fast computation of persistence homology.

3. You can simply run the command `python scripts/ph_optimization.py` to try it out.
Then, given a point cloud data with a circle and one outlier, the three algorithms try to make the hole larger with multiple learning rates.
The results will be saved in the directory `results/sample` in default.

If you want to start using optimization methods right away, please refer to `notebooks/basic_usage.ipynb` for usage examples. 
If you want to learn about the interface for comparing implemented optimization methods, please refer to `notebooks/interface_for_comparison.ipynb`. 
For more details about the implementation, please see the following explanation.

## Usage of `scripts/ph_optimization.py`

You can also specify your favorite settings by changing the attributes in `OptConfig` whose parameters are described in the following.
For more details on how to use `OptConfig`, please refer to the part below `if __name__ == "__main__":` in `scripts/ph_optimization.py`.
Regarding data, loss functions and regularizations, the subsequent sections will clarify how to design them.

Parameters of `OptConfig`:
- COMMON SETTINGS
    - exp_name(str, default=""): Experiment name. This will be used as a directory name to save the results.
    - save_dirpath(str, default="results/sample"): Directory name to save the results.
    - data_func(Callable, default=circle_with_one_outlier): Data generating function. You can define your own function in `data_loader.py`.
    - num_trial(int, default=1): If you want to perform the optimization multiple times with different initial values and see the average results, set this parameter.
    - num_epoch(int, default=100): Number of epochs. If `None`, the optimization is performed until `time_limit`.
    - time_limit(Optional[float], default=None): Time limit. If `None`, the optimization is performed until `num_epoch`.
    - log_interval(int, default=10): The logs (for example, loss value) are printed every `log_interval` epochs.
- LOSS FUNCTION
    - loss_obj(PersistenceBasedLoss, default=ExpandLoss([1], 1, topk=1)): 
        Object that determines the loss function. You can define your own function in `persistence_based_loss.py`.
    - regularization_obj(Optional[Regularization], default=RectangleRegularization(-2., -2., 2., 2., 1., 2)): 
        Regularization. You can define your own function in `regularization.py`.
- METHOD
    - method(str, default="gd"): Optimization method. "gd", "bigstep", "continuation" and "diffeo" are available.
    - lr(float, default=1e-1): Learning rate.
    - reg_proj(bool, default=False): 
        If `True`, the algorithm projects the variables to the region where the regularization term is zero at the end of each epoch.
    - optimizer_conf(dict, default={}): OptConfig for the optimizer used in "gd" and "bigstep". You can specify the following keys:
        - "name"(str, default="SGD"): Name of the optimizer. You can choose from "SGD" and "Adam".
    - scheduler_conf(dict, default={}): OptConfig for the scheduler. You can specify the following keys:
        - "name"(str, default="const"): Name of the scheduler. You can choose from "const" and "TransformerLR".
    - num_in_iter(int, default=1): Number of iterations in the continuation method.

## Data

The function to provide point cloud data is defined in `scripts/data_loader.py`.
In the default setting of `OptConfig`, the function `circle_with_one_outlier` is used.
This function generates a point cloud data with a circle with radius 1 (+ uniform noise) and one outlier near the origin.

You can define your own data generating function.
When implementing this function yourself, please pay attention to the following points.
- The function should take one argument `num_pts` (int), which represents the number of points in the point cloud.
- The function should return a numpy array of shape `(1000, num_pts, dim)`.
Note that you have to create 1000 point clouds.

## Persistence-based loss functions

The class which determines the loss function is defined in `scripts/persistence_based_loss.py`.
In the default setting of `OptConfig`, the class `ExpandLoss` is used.
This class defines the loss function that tries to expand the hole in the persistence diagram.

You can define your own class to use your favorite loss function.
When implementing this class yourself, please pay attention to the following points.
- The class should inherit `PersistenceBasedLoss`.
- You have to implement the method `__call__` and `get_direction`. 
We describe the role of these methods in the following.
See the comments in the `PersistenceBasedLoss` for more details on how to implement these methods.
    - `__call__`: the method to compute the loss value.
    - `get_direction`: the method to get the desireble direction to move for each point in the persistence diagram. 

## Regularizations

The class which determines the regularization is defined in `scripts/regularization.py`.
In the default setting of `OptConfig`, the class `RectangleRegularization` is used.
This class defines the regularization that restricts the point cloud to a rectangle.

You can define your own class to use your favorite regularization.
When implementing this class yourself, please pay attention to the following points.
- The class should inherit `Regularization`.
- You have to implement the method `__call__` and `projection`. 
We describe the role of these methods in the following.
See the comments in the `Regularization` for more details on how to implement these methods.
    - `__call__`: the method to compute the value of regularization term.
    - `projection`: the method to project the variables to the region where the regularization term is zero. 

## References

[1] Marcio Gameiro, Yasuaki Hiraoka, and Ippei Obayashi. Continuation of point clouds via persistence diagrams. Physica D: Nonlinear Phenomena, 334:118–132, 2016.

[2] Arnur Nigmetov and Dmitriy Morozov. Topological optimization with big steps. Discrete & Computational Geometry, 1-35, 2024.

[3] Mathieu Carrière, Marc Theveneau, Théo Lacombe. Diffeomorphic interpolation for efficient persistence-based topological optimization. arXiv:2405.18820, 2024.

[4] Julián Burella Pérez, Sydney Hauke, Umberto Lupo, Matteo Caorsi, Alberto Dassatti. Giotto-ph: A Python Library for High-Performance Computation of Persistent Homology of Vietoris–Rips Filtrations.arXiv:2107.05412, 2021.

[5] Ulrich Bauer. Ripser: efficient computation of Vietoris-Rips persistence barcodes. Journal of Applied and Computational Topology, 5(3):391-423, 2021.

<!-- 
## Remaining Tasks
- [ ] Implement Gradient Sampling ...?
- [ ] Improve continuation: reimplement the update as a gradient step of auxiliary loss so that the regularization term is treated properly. 
- [ ] Add Diffeo to notebooks/interface_for_comparison.ipynb
-->
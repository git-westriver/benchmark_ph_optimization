# Library for optimization of pointclouds with persistence-based loss

By Naoki Nishikawa ([GitHub](https://github.com/git-westriver), [Homepage](https://sites.google.com/view/n-nishikawa))
and Yuichi Ike ([Homepage](https://sites.google.com/view/yuichi-ike))

This repository presents a library for the optimization of point clouds with persistence-based loss.
We provide the implementation of the existing optimization methods and the interface to compare them.

Our implementation has following nice features:
- ***Multiple methods***: standard gradient descent, Conginuation [1], Big Step [2] and Diffeo [3].
- ***High flexibility***: you can specify any point clouds, persistence based losses or regularizations, **without changing the implementation of the algorithms**.
- ***Fast computation***: basically, persistence homology will be computed by `giotto-ph` [4]. 
If we need matrix decomposition, we use our fast implementation inspired by [5].

![animation](https://github.com/git-westriver/benchmark_ph_optimization/assets/64912615/3544b12f-b9f9-4d85-90c4-eae94d77e481)

## Version table

| Version | Date | Description |
|:---:|:---:|:---|
| [0.1.0](https://github.com/git-westriver/benchmark_ph_optimization/tree/ver-0.1.0) |  | Initial release. |
| [0.2.0](https://github.com/git-westriver/benchmark_ph_optimization/tree/ver-0.2.0) | 2025.03.06 | Refactor the visualization feature. Fix some bugs. |
| [1.0.0](https://github.com/git-westriver/benchmark_ph_optimization/tree/ver-1.0.0) | 2025.05.02 | Package the library for `pip install`. Add new interface `ph-grad`. |

## How to start

You can install the library by using `pip` command:

```bash
pip install git+https://github.com/git-westriver/benchmark_ph_optimization.git@ver-1.0.0
```

## Usage

We offer two interfaces for point cloud optimization of persistence-based loss.
We also provide a convenient nterface that enables optimization, result saving, and visualization with simple code.

### 1. `ph-loss` interface

Through this interface, you can optimize **one point cloud** $X$ with the loss defined as

$\mathcal{L}(X) = L_{\text{topo}}(X) + \lambda L_{\text{reg}}(X),$

where $L_{\text{topo}}$ is the topological loss, $L_{\text{reg}}$ is the regularization term and $\lambda$ is the hyperparameter.
You can customize $L_{\text{topo}}$ and $L_{\text{reg}}$ in the way described in `notebooks/01_usage_of_algorithms`.

### 2. `ph-grad` interface

You can enjoy more flexible optimization with the `ph-grad` interface.
This interface provides a implementation of functions that compute persistence-based loss
with improved gradients.
The loss can be optimized through standard pytorch implementation like the following:
```python
import torch
from torch.optim import SGD
from ph_opt import powered_wasserstein_distance_one_sided

X = torch.randn(100, 2, requires_grad=True) # point cloud
optimizer = SGD([X], lr=0.01) # optimizer
ref_pd = [torch.tensor([1., 2.])] # target persistence diagram
for epoch in range(100): # standard pytorch training loop
    optimizer.zero_grad() 
    loss = powered_wasserstein_distance_one_sided(X, ref_pd, dims=[1], grad_type='bigstep')
    loss.backward()
    optimizer.step()
```
For more details, please refer to `notebooks/02_usage_of_gradient_interface.ipynb`.

### PHTrainer

PHTrainer is an interface that handles both point cloud optimization and trajectory visualization/saving in a unified manner.
By simply specifying the loss function (and regularization), you can avoid writing boilerplate training code, helping to keep your codebase concise.
This is particularly useful when comparing different optimization methods.
The scripts `main_ph_loss.py` and `main_ph_grad.py` use this interface to compare the results of various methods.
For usage details, please refer to these files as well as the docstrings of PHTrainerConfig and PHTrainer.

## References

[1] Marcio Gameiro, Yasuaki Hiraoka, and Ippei Obayashi. Continuation of point clouds via persistence diagrams. Physica D: Nonlinear Phenomena, 334:118–132, 2016.

[2] Arnur Nigmetov and Dmitriy Morozov. Topological optimization with big steps. Discrete & Computational Geometry, 1-35, 2024.

[3] Mathieu Carrière, Marc Theveneau, Théo Lacombe. Diffeomorphic interpolation for efficient persistence-based topological optimization. NeurIPS 2024.

[4] Julián Burella Pérez, Sydney Hauke, Umberto Lupo, Matteo Caorsi, Alberto Dassatti. Giotto-ph: A Python Library for High-Performance Computation of Persistent Homology of Vietoris–Rips Filtrations.arXiv:2107.05412, 2021.

[5] Ulrich Bauer. Ripser: efficient computation of Vietoris-Rips persistence barcodes. Journal of Applied and Computational Topology, 5(3):391-423, 2021.
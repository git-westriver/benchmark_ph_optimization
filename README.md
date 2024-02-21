# Benchmarks for optimization of pointclouds with persistent based loss

By Naoki Nishikawa ([GitHub](https://github.com/git-westriver), [Homepage](https://sites.google.com/view/n-nishikawa))
and Yuichi Ike ([Homepage](https://sites.google.com/view/yuichi-ike))

This repository presents the benchmarks for the optimization of point clouds with persistent based loss.
We concentrate on optimizing Vietoris Rips filtration of point clouds.

Our implementation has following nice features:
- ***Multiple methods***: standard gradient descent, Conginuation [1], and Big Step [2].
- ***High flexibility***: you can specify any point clouds, persistence based losses or regularizations, **without changing the implementation of the algorithms**.
- ***Fast computation***: basically, persistence homology will be computed by `giotto-ph` [3]. 
If we need matrix decomposition, we use our fast implementation inspired by `ripser` [4].

## Usage of `scripts/ph_optimization.py`

## Data

## Persistence-based loss functions

## Regularizations

## References

[1] Marcio Gameiro, Yasuaki Hiraoka, and Ippei Obayashi. Continuation of point clouds via persistence diagrams. Physica D: Nonlinear Phenomena, 334:118–132, 2016.

[2] Arnur Nigmetov and Dmitriy Morozov. Topological optimization with big steps. arXiv:2203.16748, 2022.

[3] 

[4] 
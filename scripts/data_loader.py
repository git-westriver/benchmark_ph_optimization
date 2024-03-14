import os
import random
from typing import Callable
import numpy as np
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt

def get_data(func_name: Callable, num_pts: int) -> np.ndarray:
    filename = f"data/{func_name.__name__}_num-pts={num_pts}"
    if os.path.isfile(filename + ".npy"):
        return np.load(filename + ".npy")
    X = func_name(num_pts)
    if os.path.isdir("data"):
        np.save(filename + ".npy", X)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(X[0, :, 0], X[0, :, 1])
        ax.set_aspect("equal")
        fig.savefig(filename + ".png")
    return X

def circle_with_one_outlier(num_pts: int) -> np.ndarray:
    X_list = []
    for k in range(1000):
        theta = np.linspace(0, 2 * np.pi, num_pts-1)
        X = np.stack([np.cos(theta), np.sin(theta)], axis=1) + np.random.randn(num_pts-1, 2) * 0.1
        outlier = np.random.randn(1, 2) * 0.1
        X = np.concatenate([X, outlier], axis=0)
        X_list.append(X)
    return np.stack(X_list, axis=0)

def voronoi(num_pts: int) -> np.ndarray:
    X_list = []
    for k in range(1000):
        voronoi_site_idx_list = random.sample(range(25), k=15)
        voronoi_site = np.stack([np.array([(idx % 5 - 2) / 2, (idx // 5 - 2) / 2]) for idx in voronoi_site_idx_list], axis=0)
        vor = Voronoi(voronoi_site)
        edge_set = set()
        for _edge in vor.ridge_vertices:
            if -1 in _edge: continue
            else:
                edge = tuple(sorted(_edge))
                if edge not in edge_set:
                    edge_set.add(edge)
        edge_list = list(edge_set)
        edge_length_list = [np.linalg.norm(vor.vertices[edge[0]] - vor.vertices[edge[1]]) for edge in edge_list]
        edge_choices = random.choices(edge_list, weights=edge_length_list, k=num_pts)
        pts_list = []
        for i in range(num_pts):
            edge = edge_choices[i]
            t = random.random()
            pts_list.append(vor.vertices[edge[0]] + t * (vor.vertices[edge[1]] - vor.vertices[edge[0]]))
        X = np.stack(pts_list, axis=0)
        X += np.random.randn(num_pts, 2) * 0.01
        X_list.append(X)
    return np.stack(X_list, axis=0)
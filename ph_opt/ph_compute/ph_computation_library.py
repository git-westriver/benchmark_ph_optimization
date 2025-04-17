from typing import Union, Optional, Any
from scipy.spatial.distance import cdist
import torch
import numpy as np
from gph import ripser_parallel
from numbers import Number

from .bin.rips_cpp import RipsPersistentHomology

class Bar:
    """
    Bar object to represent a bar in the barcode.

    Parameters:
        - birth_time(float): birth time. Note that this is `float` type, not `torch.Tensor`.
        - death_time(float): death time. Note that this is `float` type, not `torch.Tensor`.
        - birth_v1(int): vertex index of the birth simplex.
        - birth_v2(Optional[int]): vertex index of the birth simplex. If the birth simplex is a vertex, `None`.
        - death_v1(int): vertex index of the death simplex.
        - death_v2(int): vertex index of the death simplex.
        - birth_simp(Optional[int]): index of the birth simplex. If giotto-ph is used to compute PH, `None`.
        - death_simp(Optional[int]): index of the death simplex. If giotto-ph is used to compute PH, `None`.
    """
    def __init__(self, birth_time: float, death_time: float, 
                 birth_v1: int, birth_v2: Optional[int], death_v1: int, death_v2: int,
                 birth_simp: Optional[int]=None, death_simp: Optional[int]=None):
        assert isinstance(birth_time, Number), "birth_time should be a number."
        assert isinstance(death_time, Number), "death_time should be a number."
        
        self.birth_v1: int = birth_v1
        self.birth_v2: int = birth_v2
        self.death_v1: int = death_v1
        self.death_v2: int = death_v2
        self.birth_time: float = birth_time
        self.death_time: float = death_time
        self.birth_simp: Optional[int] = birth_simp
        self.death_simp: Optional[int] = death_simp

class RipsPH(RipsPersistentHomology):
    """
    Object to compute persistent homology of Rips filtration.
    Parameters:
        X (Union[torch.Tensor, np.ndarray]) : point cloud. shape=(# of points, dim)
        maxdim (int) : the maximum dimension of the persistent homology.
        distance_matrix (bool, default=False) : if `True`, `X` is treated as a distance matrix.
        num_threads (int, default=1) : the number of threads to use.
    """
    def __init__(self, X: Union[torch.Tensor, np.ndarray], maxdim, distance_matrix=False, num_threads=1):
        if distance_matrix and type(X).__module__ == "numpy":
            self.dist_mat = X.copy()
        elif distance_matrix and type(X).__module__ == "torch":
            self.dist_mat = X.clone()
        elif type(X).__module__ == "torch":
            self.dist_mat = torch.cdist(X, X)
        else:
            self.dist_mat = cdist(X, X)
        if type(self.dist_mat).__module__ in ["numpy", "torch"]:
            _dist_mat = self.dist_mat.tolist()
        else:
            _dist_mat = self.dist_mat
        super().__init__(_dist_mat, maxdim, num_threads)
        self.maxdim: int = maxdim
        self.get_ph_left: bool = False
        self.get_ph_right: bool = False
        self.giotto_dgm: Optional[dict[str, Union[list[np.ndarray], tuple]]] = None
    
    def compute_ph(self, enclosing_opt=True, emgergent_opt=True, clearing_opt=True, get_inv=False):
        self.get_ph_left = True
        return super().compute_ph(enclosing_opt, emgergent_opt, clearing_opt, get_inv)
    
    def compute_ph_right(self, enclosing_opt=True, emgergent_opt=True, get_inv=False):
        self.get_ph_right = True
        return super().compute_ph_right(enclosing_opt, emgergent_opt, get_inv)
    
    def _call_giotto_ph(self):
        if type(self.dist_mat).__module__ == "torch":
            _dist_mat = self.dist_mat.detach().clone().numpy()
        else:
            _dist_mat = self.dist_mat
        self.giotto_dgm = ripser_parallel(_dist_mat, maxdim=self.maxdim, metric="precomputed", return_generators=True)
    
    def get_barcode(self, dim, out_format="list"):
        """
        Get (non-differentiable) barcode of dimension `dim`.
        If compute_ph or compute_ph_right have not been called, PH will be computed with giotto-ph.

        Parameters:
            - dim: dimension of the barcode.
            - out_format: the format of the output. "list", "numpy", or "torch".

        Returns:
            - list of tuples of birth and death time.
        """
        barcode = []
        if (not self.get_ph_left) and (not self.get_ph_right):
            if self.giotto_dgm is None:
                self._call_giotto_ph()
            if out_format == "list":
                return [(b_time, d_time) for b_time, d_time in self.giotto_dgm["dgms"][dim]]
            elif out_format == "numpy":
                return self.giotto_dgm["dgms"][dim]
            elif out_format == "torch":
                return torch.tensor(self.giotto_dgm["dgms"][dim])
            raise NotImplementedError
        else: # PH is already computed
            if self.get_ph_left:
                for death, birth in self.death_to_birth[dim].items():
                    b_time, d_time = self.get_diameter(dim, birth), self.get_diameter(dim+1, death)
                    if b_time < d_time:
                        barcode.append((b_time, d_time))
            else:
                for birth, death in self.birth_to_death[dim].items():
                    b_time, d_time = self.get_diameter(dim, birth), self.get_diameter(dim+1, death)
                    if b_time < d_time:
                        barcode.append((b_time, d_time))
            if out_format == "list":
                return barcode
            elif out_format == "numpy":
                if barcode:
                    return np.array(barcode)
                else:
                    return np.empty([0, 2])
            elif out_format == "torch":
                if barcode:
                    return torch.tensor(barcode)
                else:
                    return torch.empty([0, 2])
            raise NotImplementedError
    
    def get_differentiable_diameter(self, dim: int, idx: int, dist_mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get differentiable diameter of the simplex with index `idx` in dimension `dim`.
        Args:
            dim (int): dimension of the simplex.
            idx (int): index of the simplex.
            dist_mat (Optional[torch.Tensor]): distance matrix. If `None`, use the distance matrix of the object.
        Returns:
            diameter (torch.Tensor): diameter of the simplex.
        """
        if dist_mat is None:
            dist_mat = self.dist_mat

        if type(dist_mat).__module__ != "torch":
            raise ValueError("Differentiable barcode is only available for torch.Tensor")
        v1, v2 = self.get_max_edge(dim, idx)
        return dist_mat[v1, v2]
    
    def get_differentiable_barcode(self, dim: int, dist_mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get differentiable barcode of dimension `dim`.
        If compute_ph or compute_ph_right have not been called, PH will be computed with giotto-ph.

        Args:
            dim (int):  dimension of the barcode.
            dist_mat (Optional[torch.Tensor]): distance matrix. If `None`, use the distance matrix of the object.

        Returns:
            barcode (torch.Tensor): shape=(# of bars, 2).
        """
        if dist_mat is None:
            dist_mat = self.dist_mat

        if type(dist_mat).__module__ != "torch":
            raise ValueError("Differentiable barcode is only available for torch.Tensor")
        b_time_list = []; d_time_list = []
        if self.get_ph_left:
            for d_simp, b_simp in self.death_to_birth[dim].items():
                b_time_list.append(self.get_differentiable_diameter(dim, b_simp))
                d_time_list.append(self.get_differentiable_diameter(dim+1, d_simp))
        elif self.get_ph_right:
            for b_simp, d_simp in self.birth_to_death[dim].items():
                b_time_list.append(self.get_differentiable_diameter(dim, b_simp))
                d_time_list.append(self.get_differentiable_diameter(dim+1, d_simp))
        else:
            if self.giotto_dgm is None:
                self._call_giotto_ph()
            if dim == 0:
                for _, dv1, dv2 in self.giotto_dgm["gens"][0]:
                    b_time_list.append(torch.tensor(0.))
                    d_time_list.append(dist_mat[dv1, dv2])
            else:
                for bv1, bv2, dv1, dv2 in self.giotto_dgm["gens"][1][dim-1]:
                    b_time_list.append(dist_mat[bv1, bv2])
                    d_time_list.append(dist_mat[dv1, dv2])
            
        if b_time_list:
            b_tensor = torch.stack(b_time_list, dim=0)
            d_tensor = torch.stack(d_time_list, dim=0)
            return torch.stack([b_tensor, d_tensor], dim=1)
        else:
            return torch.empty([0, 2], dtype=torch.float32)
    
    def get_bar_object_list(self, dim: int) -> list[Bar]:
        """
        Get list of Bar objects. 
        Remark:
            - If compute_ph or compute_ph_right have not been called, PH will be computed with giotto-ph.
            As a result, the Bar objects will not contain the indices of simplices.
        """
        ret: list[Bar] = []
        if self.get_ph_left:
            for d_simp, b_simp in self.death_to_birth[dim].items():
                bv1, bv2 = self.get_max_edge(dim, b_simp)
                dv1, dv2 = self.get_max_edge(dim+1, d_simp)
                b_time, d_time = self.dist_mat[bv1, bv2], self.dist_mat[dv1, dv2]
                b_time, d_time = float(b_time), float(d_time)
                ret.append(Bar(b_time, d_time, bv1, bv2, dv1, dv2, b_simp, d_simp))
        elif self.get_ph_right:
            for b_simp, d_simp in self.birth_to_death[dim].items():
                bv1, bv2 = self.get_max_edge(dim, b_simp)
                dv1, dv2 = self.get_max_edge(dim+1, d_simp)
                b_time, d_time = self.dist_mat[bv1, bv2], self.dist_mat[dv1, dv2]
                b_time, d_time = float(b_time), float(d_time)
                ret.append(Bar(b_time, d_time, bv1, bv2, dv1, dv2, b_simp, d_simp))
        else:
            if self.giotto_dgm is None:
                self._call_giotto_ph()
            if dim == 0:
                for bv, dv1, dv2 in self.giotto_dgm["gens"][0]:
                    d_time = float(self.dist_mat[dv1, dv2])
                    ret.append(Bar(0, d_time, bv, None, dv1, dv2))
            else:
                for bv1, bv2, dv1, dv2 in self.giotto_dgm["gens"][1][dim-1]:
                    b_time, d_time = self.dist_mat[bv1, bv2], self.dist_mat[dv1, dv2]
                    b_time, d_time = float(b_time), float(d_time)
                    ret.append(Bar(b_time, d_time, bv1, bv2, dv1, dv2))

        return ret
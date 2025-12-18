from ..ph_compute.ph_computation_library import RipsPH, Bar
from .singleton import singleton_loss_from_bar_to_target
from .lib.stratified_gd import dijkstra_over_swaps

import torch
from torch.autograd import Function
from typing import Optional, Literal

def _get_rph_loss_targets_for_topk_persistence(
    X: torch.Tensor, 
    dims: list[int], 
    order: int = 2, 
    topk: Optional[int] = None,
    need_V_and_W: bool = False
) -> list[tuple[int, Bar, torch.Tensor]]:

    target_info = []

    # compute RipsPH
    maxdim = max(dims)
    rph = RipsPH(X, maxdim=maxdim)

    # if need_V_and_W is True, compute PH with access to V and W
    if need_V_and_W:
        rph.compute_ph(clearing_opt=False, get_inv=True)
        rph.compute_ph_right(get_inv=True)

    loss = torch.tensor(0.)

    for dim in dims:
        # get bars and sort them by the decreasing order of persistence
        bars = rph.get_bar_objects(dim)
        bars = sorted(bars, key=lambda bar: bar.death_time - bar.birth_time, reverse=True)

        cnt = 0 # number of bars added to target_info
        for bar in bars:
            loss += ( (bar.death_time - bar.birth_time) / (2 ** 0.5) ) ** order
            coord = (bar.birth_time + bar.death_time) / 2
            target = torch.tensor([coord, coord])
            target_info.append((dim, bar, target))
            cnt += 1

            if cnt >= topk:
                break

    return rph, loss, target_info

# ----- standard wasserstein loss & its gradient -----

def _topk_persistence_loss_with_standard_grad(
    rph: RipsPH, 
    target_info: list[tuple[int, Bar, torch.Tensor]], 
    order: int = 2, 
    X: Optional[torch.Tensor] = None
) -> torch.Tensor:

    # get the (differentiable) distance matrix
    if X is not None:
        dist_mat = torch.cdist(X, X)
    else:
        dist_mat = rph.dist_mat
    assert dist_mat.requires_grad

    # compute the loss
    loss = torch.tensor(0.)
    for dim, bar_to_move, target in target_info:
        # get the birth time and death time
        if bar_to_move.birth_v2 is None:
            birth_time = torch.tensor(0.)
        else:
            v1, v2 = bar_to_move.birth_v1, bar_to_move.birth_v2
            birth_time = dist_mat[v1, v2]
        
        v1, v2 = bar_to_move.death_v1, bar_to_move.death_v2
        death_time = dist_mat[v1, v2]

        # compute the (differentiable) loss
        loss += ( (death_time - birth_time) / (2 ** 0.5) ) ** order
        
    return loss

def _get_standard_gradient_for_topk_persistence(
    X: torch.Tensor, 
    rph: RipsPH, 
    target_info: list[tuple[int, Bar, torch.Tensor]], 
    order: int = 2
) -> torch.Tensor:

    with torch.enable_grad():
        _X = X.detach().clone().requires_grad_()
        _loss = _topk_persistence_loss_with_standard_grad(
            rph, target_info, order, 
            X=_X    # X を与えて barcode に勾配を復活させる
        )
    
    if _loss.requires_grad:
        standard_df_dX, = torch.autograd.grad(
            outputs=_loss, inputs=(_X,), retain_graph=False, create_graph=False
        )
        standard_df_dX = torch.nan_to_num(standard_df_dX, nan=0.0)
    else:
        standard_df_dX = torch.zeros_like(_X)

    return standard_df_dX

# ----- improved gradient -----

def aux_loss_for_stratified_gradient_for_topk_persistence(
    X: torch.Tensor, 
    rph: RipsPH, 
    target_info: list[tuple[int, Bar, torch.Tensor]], 
    order: int, 
    eps: float,
    n_strata: int,
) -> torch.Tensor:
    """
    Stratified gradient 用の補助損失を計算する。

    Parameters:
        X : (num_pts, dim) の点群データ (torch.Tensor)
        rph : X の RipsPH
        target_info : (dim, bar, target) のリスト．各 bar の向かうべき場所を示す．
        order : ワッサースタイン距離の次数
    """
    dist_mat = torch.cdist(X, X)
    dims = list(set([dim for dim, bar, target in target_info]))
    maxdim = max(dims)
    nearby_dist_mats = dijkstra_over_swaps(dist_mat, n_strata, eps)

    loss = torch.tensor(0., device=X.device, dtype=X.dtype)
    for _dist_mat in nearby_dist_mats:
        rph = RipsPH(_dist_mat, maxdim=maxdim, distance_matrix=True)
        rph._call_giotto_ph()
        assert rph.giotto_dgm is not None

        _loss = _topk_persistence_loss_with_standard_grad(
            rph=rph, target_info=target_info, order=order,
            X=X     # X を与えると，ペアはそのままで，X の距離行列で PH を計算
        )
        assert _loss.requires_grad

        loss += _loss

    loss /= len(nearby_dist_mats)
    
    return loss

def _get_improved_gradient_for_topk_persistence(
    X: torch.Tensor, 
    rph: RipsPH, 
    target_info: list[tuple[int, Bar, torch.Tensor]], 
    grad_type: str, 
    order: int = 2, 
    sigma: float = 0.1, 
    lmbd: float = 1e-5, 
    eps: float = 1., 
    n_strata: int = 5,  
    all_X: Optional[torch.Tensor] = None
) -> torch.Tensor:

    if grad_type == "stratified":
        with torch.enable_grad():
            _X = X.detach().clone().requires_grad_()
            _loss = aux_loss_for_stratified_gradient_for_topk_persistence( # TODO
                _X, rph, target_info, order, 
                eps, n_strata # parameters specific for "stratified"
            )

    else: # same as wasserstein
        with torch.enable_grad():
            _X = X.detach().clone().requires_grad_()
            _loss = torch.tensor(0., device=_X.device, dtype=_X.dtype)

            for dim, bar_to_move, target in target_info:
                _loss += singleton_loss_from_bar_to_target(
                    _X, bar_to_move, target, 
                    grad_type=grad_type, 
                    order=order, 
                    normalize_grad=False, # should be False (normalization will be done in this code)
                    dim=dim, rph=rph, sigma=sigma, lmbd=lmbd, all_X=all_X
                )

    if _loss.requires_grad:
        improved_df_dX, = torch.autograd.grad(outputs=_loss, inputs=(_X,), retain_graph=False, create_graph=False)
    else:
        improved_df_dX = torch.zeros_like(_X)

    return improved_df_dX

# ----- interfaces -----

class _topk_persistence_loss_with_improved_grad(Function):
    @staticmethod
    def forward(
        ctx, 
        X: torch.Tensor, 
        dims: list[int], 
        order: int = 2, 
        topk: Optional[int] = None,
        grad_type: Optional[str] = None, 
        clip_grad: Literal["l2", "linf", "none"] = "l2", 
        sigma: float = 0.1, 
        lmbd: float = 1e-5, 
        eps: float = 1., 
        n_strata: int = 5,
        all_X: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # save the variables in the context
        if all_X is not None:
            ctx.save_for_backward(X, all_X)
        else:
            ctx.save_for_backward(X)

        ctx.dims        = dims
        ctx.order       = order
        ctx.grad_type   = grad_type
        ctx.clip_grad   = clip_grad
        ctx.sigma       = sigma
        ctx.lmbd        = lmbd
        ctx.eps         = eps
        ctx.n_strata    = n_strata

        # get RipsPH and loss
        rph, loss, target_info = _get_rph_loss_targets_for_topk_persistence(
            X, dims, order, topk, need_V_and_W=(grad_type == 'bigstep')
        )

        # operation to make grad_fn works
        loss = loss + X.sum() * 0.

        # save rph and target_info in the context
        ctx.rph, ctx.target_info = rph, target_info

        return loss
    
    @staticmethod
    def backward(ctx, dF_df):
        # get the variables from the context
        all_X_provided = (len(ctx.saved_tensors) == 2)
        if all_X_provided:
            X, all_X = ctx.saved_tensors
        else:
            X = ctx.saved_tensors[0]
            all_X = None
        
        dims = ctx.dims                                 # dimensions of PH
        order, grad_type = ctx.order, ctx.grad_type     # basic information of the loss
        clip_grad = ctx.clip_grad                       # whether normalize the gradient or not
        sigma, lmbd = ctx.sigma, ctx.lmbd               # parameters for Diffeo
        eps, n_strata = ctx.eps, ctx.n_strata           # parameters for Stratified GD
        rph, target_info = ctx.rph, ctx.target_info     # information of current PD and directions to move

        # compute the un-normalized gradient
        improved_df_dX = _get_improved_gradient_for_topk_persistence(
            X, rph, target_info, grad_type, order, sigma, lmbd, eps, n_strata, all_X
        )

        # if improved_df_dX is not zero, clip the gradient using the information of standard_dF_df
        if (improved_df_dX.norm() > 0) and (grad_type != 'standard') and (clip_grad != "none"):
            standard_df_dX = _get_standard_gradient_for_topk_persistence(X, rph, target_info, order)

            if clip_grad == "l2":
                improved_df_dX = (improved_df_dX / improved_df_dX.norm()) * standard_df_dX.norm()
            elif clip_grad == "linf":
                max_grad = standard_df_dX.abs().max()
                improved_df_dX = improved_df_dX.clamp(min=-max_grad, max=max_grad)
            else:
                raise NotImplementedError(f"clip_grad = {clip_grad} is not implemented.")

        # compute the gradient
        dF_dX = dF_df * improved_df_dX

        if all_X_provided and (grad_type == 'diffeo'):
            return None, None, None, None, None, None, None, None, None, None, dF_dX
        else:
            return dF_dX, None, None, None, None, None, None, None, None, None, None

def topk_persistence_loss(
    X: torch.Tensor, 
    dims: list[int], 
    order: int = 2, 
    topk: Optional[int] = None,
    grad_type: str = "standard", 
    clip_grad: Literal["l2", "linf", "none"] = "l2",
    sigma: float = 0.1, 
    lmbd: float = 1e-5, 
    eps: float = 1., 
    n_strata: int = 5,
    all_X: Optional[torch.Tensor] = None
) -> torch.Tensor: 
    """
    Compute the sum of the persistence of points in the PD of X. 
    If you specify `topk`, the loss is computed using only the top-k most persistent points.

    Parameters:
        X (torch.Tensor) : point cloud. shape=(# of points, dim)
        dims (list[int]) : list of dimensions of the persistent homology.
        order (int) : the order of Wasserstein distance.
        topk (int | None): If not None, sum only the top-k most persistent points (largest death - birth) in each persistence diagram.
        grad_type (str) : the method to compute the gradient. One of ['standard', 'bigstep', 'continuation', 'diffeo'].
        clip_grad ("l2" | "linf" | "none") : Gradient clipping mode. "l2" clips by L2 norm, "linf" clips elementwise by L^\infty, and "none" disables clipping.
        sigma (float) : the bandwidth of the Gaussian kernel. Only used when `method` is 'diffeo'.
        lmbd (float): regularization parameter for kernel ridge regression. Only used when `method` is `diffeo`.
        eps (float): the maximum distance searched in Dijkstra's algorithm. Only used when `method` is 'stratified'.
        n_strata (int): the number of strata to consider. Only used when `method` is 'stratified'.
        all_X (torch.Tensor) : the point cloud for diffeomorphic interpolation. shape=(# of points, dim). Only used when `method` is 'diffeo' and `all_X` is not `None`.
        
    Returns:
        loss: the sum of the persistence.
    """

    # if (topk is not None) and len(dims) > 1:
    #     raise ValueError("Need to specify a unique dimension if topk is not None.")

    return _topk_persistence_loss_with_improved_grad.apply(
        X, dims, order, topk, grad_type, clip_grad, sigma, lmbd, eps, n_strata, all_X
    )

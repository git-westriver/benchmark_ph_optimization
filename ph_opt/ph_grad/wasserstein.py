from ..ph_compute.ph_computation_library import RipsPH, Bar
from .singleton import singleton_loss_from_bar_to_target
from .lib.wasserstein import _powered_wasserstein_distance_one_sided_from_rph_with_standard_grad
from .lib.stratified_gd import aux_loss_for_stratified_gradient

import torch
from torch.autograd import Function
from gudhi.wasserstein import wasserstein_distance
from typing import Optional

def _get_rph_loss_targets_for_wasserstein(X: torch.Tensor, ref_pd: list[torch.Tensor], dims: list[int], 
                                          order: int = 2, need_V_and_W: bool = False) -> list[tuple[int, Bar, torch.Tensor]]:
    target_info = []

    # compute RipsPH
    maxdim = max(dims)
    rph = RipsPH(X, maxdim=maxdim)

    # if need_V_and_W is True, compute PH with access to V and W
    if need_V_and_W:
        rph.compute_ph(clearing_opt=False, get_inv=True)
        rph.compute_ph_right(get_inv=True)

    for dim, _ref_pd in zip(dims, ref_pd):
        # get bars and matching to ref_pd
        bars = rph.get_bar_objects(dim)
        barcode = torch.tensor([[bar.birth_time, bar.death_time] for bar in bars])
        loss, matching = wasserstein_distance(barcode, _ref_pd, order=order, matching=True, keep_essential_parts=False)

        # obtain the targets
        for i, j in matching:
            if i == -1:
                continue
            elif j == -1:
                bar_to_move: Bar = bars[i]
                _target = (bar_to_move.birth_time + bar_to_move.death_time) / 2
                target = torch.tensor([_target, _target])
            else:
                bar_to_move = bars[i]
                target = _ref_pd[j]
            target_info.append((dim, bar_to_move, target))

    return rph, loss, target_info

def _get_standard_gradient_for_wasserstein(X: torch.Tensor, rph: RipsPH, 
                                           ref_pd: list[torch.Tensor], dims: list[int], 
                                           order: int = 2):
    with torch.enable_grad():
        _X = X.detach().clone().requires_grad_()
        _loss = _powered_wasserstein_distance_one_sided_from_rph_with_standard_grad(rph, ref_pd, dims, order, _X)
    
    if _loss.requires_grad:
        standard_df_dX, = torch.autograd.grad(outputs=_loss, inputs=(_X,), retain_graph=False, create_graph=False)
        standard_df_dX = torch.nan_to_num(standard_df_dX, nan=0.0)
    else:
        standard_df_dX = torch.zeros_like(_X)

    return standard_df_dX

def _get_improved_gradient_for_wasserstein(X: torch.Tensor, rph: RipsPH, 
                                           dims: list[int], ref_pds: list[torch.Tensor],
                                           target_info: list[tuple[int, Bar, torch.Tensor]], 
                                           grad_type: str, order: int = 2, 
                                           sigma: float = 0.1, lmbd: float = 1e-5, 
                                           eps: float = 1., n_strata: int = 5,    
                                           all_X: Optional[torch.Tensor] = None):
    if grad_type == "stratified":
        with torch.enable_grad():
            _X = X.detach().clone().requires_grad_()
            _loss = aux_loss_for_stratified_gradient(_X, rph, dims, ref_pds, 
                                                     order, eps, n_strata)
    else:
        with torch.enable_grad():
            _X = X.detach().clone().requires_grad_()
            _loss = torch.tensor(0., device=_X.device, dtype=_X.dtype)
            for dim, bar_to_move, target in target_info:
                _loss += singleton_loss_from_bar_to_target(_X, bar_to_move, target, grad_type=grad_type, order=order, normalize_grad=False, 
                                                           dim=dim, rph=rph, sigma=sigma, lmbd=lmbd, all_X=all_X)

    if _loss.requires_grad:
        improved_df_dX, = torch.autograd.grad(outputs=_loss, inputs=(_X,), retain_graph=False, create_graph=False)
    else:
        improved_df_dX = torch.zeros_like(_X)

    return improved_df_dX

class _powered_wasserstein_distance_one_sided_with_improved_grad(Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, ref_pd: list[torch.Tensor], dims: list[int], 
                order: int = 2, grad_type: Optional[str] = None, 
                sigma: float = 0.1, lmbd: float = 1e-5, 
                eps: float = 1., n_strata: int = 5,
                all_X: Optional[torch.Tensor] = None):
        # save the variables in the context
        if all_X is not None:
            ctx.save_for_backward(X, all_X)
        else:
            ctx.save_for_backward(X)

        ctx.ref_pd    = ref_pd
        ctx.dims      = dims
        ctx.order     = order
        ctx.grad_type = grad_type
        ctx.sigma     = sigma
        ctx.lmbd      = lmbd
        ctx.eps       = eps
        ctx.n_strata  = n_strata

        # get RipsPH and loss
        rph, loss, target_info = _get_rph_loss_targets_for_wasserstein(X, ref_pd, dims, order, 
                                                                       need_V_and_W=(grad_type == 'bigstep'))

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
        
        ref_pd, dims = ctx.ref_pd, ctx.dims             # information of reference PD
        order, grad_type = ctx.order, ctx.grad_type     # basic information of the loss
        sigma, lmbd = ctx.sigma, ctx.lmbd               # parameters for Diffeo
        eps, n_strata = ctx.eps, ctx.n_strata           # parameters for Stratified GD
        rph, target_info = ctx.rph, ctx.target_info     # information of current PD and directions to move

        # compute the un-normalized gradient
        improved_df_dX = _get_improved_gradient_for_wasserstein(X, rph, 
                                                                dims, ref_pd, 
                                                                target_info, 
                                                                grad_type, order, 
                                                                sigma, lmbd, 
                                                                eps, n_strata,
                                                                all_X)

        # if improved_df_dX is not zero, normalize it to have the same norm as standard_dF_df
        if (improved_df_dX.norm() > 0) and (grad_type != 'standard'):
            standard_df_dX = _get_standard_gradient_for_wasserstein(X, rph, ref_pd, dims, order)
            improved_df_dX = (improved_df_dX / improved_df_dX.norm()) * standard_df_dX.norm()

        # compute the gradient
        dF_dX = dF_df * improved_df_dX

        if all_X_provided and (grad_type == 'diffeo'):
            return None, None, None, None, None, None, None, None, None, dF_dX
        else:
            return dF_dX, None, None, None, None, None, None, None, None, None
        

def powered_wasserstein_distance_one_sided(X: torch.Tensor, ref_pd: list[torch.Tensor], dims: list[int], 
                                           order: int = 2, grad_type: str = "standard", 
                                           sigma: float = 0.1, lmbd: float = 1e-5, 
                                           eps: float = 1., n_strata: int = 5,
                                           all_X: Optional[torch.Tensor] = None):
    """
    Compute the Wasserstein distance between the persistent diagram of `X` and `ref_pd`
    with improved gradient using specialized method.

    Parameters:
        X (torch.Tensor) : point cloud. shape=(# of points, dim)
        ref_pd (list[torch.Tensor]) : the reference persistent diagram. shape=(#bars, 2)
        dims (list[int]) : list of dimensions of the persistent homology.
        order (int) : the order of Wasserstein distance.
        grad_type (str) : the method to compute the gradient. One of ['standard', 'bigstep', 'continuation', 'diffeo'].
        sigma (float) : the bandwidth of the Gaussian kernel. Only used when `method` is 'diffeo'.
        lmbd (float): regularization parameter for kernel ridge regression. Only used when `method` is `diffeo`.
        eps (float): the maximum distance searched in Dijkstra's algorithm. Only used when `method` is 'stratified'.
        n_strata (int): the number of strata to consider. Only used when `method` is 'stratified'.
        all_X (torch.Tensor) : the point cloud for diffeomorphic interpolation. shape=(# of points, dim). Only used when `method` is 'diffeo' and `all_X` is not `None`.
        
    Returns:
        loss: the Wasserstein distance.
    """

    return _powered_wasserstein_distance_one_sided_with_improved_grad.apply(X, ref_pd, dims, 
                                                                            order, grad_type, 
                                                                            sigma, lmbd, 
                                                                            eps, n_strata,
                                                                            all_X)

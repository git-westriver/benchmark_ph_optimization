from ..ph_compute.ph_computation_library import Bar, RipsPH

import torch
from torch.autograd import Function
from typing import Optional

def _singleton_loss_from_bar_to_target(X: torch.Tensor, bar: Bar, target: torch.Tensor, order=2, 
                                       distance_matrix: bool = False) -> torch.Tensor:
    """
    Computes the singleton loss from a bar to a target point with standard gradient.
    
    Parameters
    ----------
    X : torch.Tensor
        The tensor that represents the point cloud.
    bar : ph_opt.Bar
        The bar to compute the loss from.
    target : torch.Tensor of shape (2,)
        The target point to compute the loss to.
    order : int
        The order of the loss. Default is 2.
    distance_matrix: bool
        If True, X is regarded as a distance matrix. Default is False.

    Returns
    -------
    loss: torch.Tensor
        The singleton loss.
    """
    def get_diameter(v1, v2):
        if distance_matrix:
            return X[v1, v2]
        else:
            return torch.norm(X[v1] - X[v2], p=2)

    # Get differentiable birth and death time
    if bar.birth_v2 is not None:
        birth_time = get_diameter(bar.birth_v1, bar.birth_v2)
    else:
        birth_time = torch.tensor(0.)
    death_time = get_diameter(bar.death_v1, bar.death_v2)

    # Compute the singleton loss
    loss = (birth_time - target[0]) ** order + (death_time - target[1]) ** order

    return loss

def _get_standard_gradient_for_singleton(X: torch.Tensor, bar: Bar, target: torch.Tensor, order=2, 
                           distance_matrix: bool = False):
    with torch.enable_grad():
        _X = X.detach().clone().requires_grad_()
        _loss = _singleton_loss_from_bar_to_target(_X, bar, target, order, distance_matrix)

    if _loss.requires_grad:
        standard_df_dX, = torch.autograd.grad(outputs=_loss, inputs=(_X,), retain_graph=False, create_graph=False)
    else:
        standard_df_dX = torch.zeros_like(_X)

    return standard_df_dX

class _singleton_loss_from_bar_to_target_with_bigstep_grad(Function):
    @staticmethod
    def forward(ctx, dist_mat: torch.Tensor, bar: Bar, target: torch.Tensor, 
                order: int = 2, normalize_grad: bool = True, 
                dim: Optional[int] = None, rph: Optional[RipsPH] = None):
        # check the value and the state of dim and rph
        if dim is None:
            raise ValueError("dim must be specified.")
        if rph is None:
            raise ValueError("rph must be specified.")
        if not rph.get_ph_left:
            rph.compute_ph(clearing_opt=False, get_inv=True)
        if not rph.get_ph_right:
            rph.compute_ph_right(get_inv=True)

        # save the variables in the context
        ctx.save_for_backward(dist_mat, target)
        ctx.bar, ctx.order, ctx.normalize_grad, ctx.dim, ctx.rph = bar, order, normalize_grad, dim, rph

        # Compute the singleton loss
        loss = _singleton_loss_from_bar_to_target(dist_mat, bar, target, order, distance_matrix=True)

        return loss

    @staticmethod
    def backward(ctx, dF_df): # F: whole network, f: output of this function, D: dist_mat
        # retrieve the saved tensors
        dist_mat: torch.Tensor
        target: torch.Tensor
        dist_mat, target = ctx.saved_tensors

        # retrieve the saved context
        bar: Bar
        order: int
        normalize_grad: bool
        dim: int
        rph: RipsPH
        bar, order, normalize_grad, dim, rph = ctx.bar, ctx.order, ctx.normalize_grad, ctx.dim, ctx.rph

        # compute the target value for each simplex
        high_target_value = {dim: {}, dim+1: {}}
        low_target_value = {dim: {}, dim+1: {}}
        b_simp, d_simp = bar.birth_simp, bar.death_simp
        b_direc = torch.sign(dF_df) * (target[0] - bar.birth_time)
        d_direc = torch.sign(dF_df) * (target[1] - bar.death_time)
        b_target, d_target = bar.birth_time + b_direc, bar.death_time + d_direc # can be different from target
        if b_direc > 0: # increase birth
            for simp in rph.W[dim][b_simp]:
                if simp in high_target_value[dim]:
                    high_target_value[dim][simp] = max(b_target, high_target_value[dim][simp])
                else:
                    high_target_value[dim][simp] = b_target
        elif b_direc < 0: # decrease birth
            for simp in rph.invW[dim][b_simp]:
                if simp in low_target_value[dim]:
                    low_target_value[dim][simp] = min(b_target, low_target_value[dim][simp])
                else:
                    low_target_value[dim][simp] = b_target
        if d_direc < 0: # decrease death
            for simp in rph.V[dim][d_simp]:
                if simp in low_target_value[dim+1]:
                    low_target_value[dim+1][simp] = min(d_target, low_target_value[dim+1][simp])
                else:
                    low_target_value[dim+1][simp] = d_target
        elif d_direc > 0: # increase death
            for simp in rph.invV[dim][d_simp]:
                if simp in high_target_value[dim+1]:
                    high_target_value[dim+1][simp] = max(d_target, high_target_value[dim+1][simp])
                else:
                    high_target_value[dim+1][simp] = d_target

        # compute df_dD (more precisely, df_dD * sign(dF_df))
        df_dD = torch.zeros_like(dist_mat)
        for _dim in [dim, dim+1]:
            moving_simplices = set(high_target_value[_dim].keys()) | set(low_target_value[_dim].keys())
            for simp in moving_simplices:
                # get the max edge and the diameter of the simplex
                v1, v2 = rph.get_max_edge(_dim, simp)
                diam = dist_mat[v1, v2]

                # get the target value
                if (simp in high_target_value[_dim]) and (simp in low_target_value[_dim]):
                    # target is the farthest one
                    high_diff = torch.relu(high_target_value[_dim][simp] - diam)
                    low_diff = torch.relu(diam - low_target_value[_dim][simp])
                    if high_diff > low_diff:
                        target_value = high_target_value[_dim][simp]
                    else:
                        target_value = low_target_value[_dim][simp]
                elif simp in high_target_value[_dim]:
                    target_value = high_target_value[_dim][simp]
                elif simp in low_target_value[_dim]:
                    target_value = low_target_value[_dim][simp]

                # compute the gradient (note: doubled by the symmetry)
                df_dD[v1, v2] += diam - target_value
                df_dD[v2, v1] += diam - target_value

        # normalize the gradient to have the same norm as the standard gradient
        if normalize_grad:
            standard_df_dD = _get_standard_gradient_for_singleton(dist_mat, bar, target, order, distance_matrix=True)
            standard_df_dD_norm = standard_df_dD.norm()
            df_dD_norm = df_dD.norm()
            if df_dD_norm > 0:
                # normalize the gradient to have the same norm as the standard gradient
                df_dD = df_dD * (standard_df_dD_norm / df_dD_norm)

        # compute dF_dD
        dF_dD = torch.abs(dF_df) * df_dD

        return dF_dD, None, None, None, None, None, None
    
def singleton_loss_from_bar_to_target_with_bigstep_grad(X: torch.Tensor, bar: Bar, target: torch.Tensor, 
                                                        order: int = 2, normalize_grad: bool = True, 
                                                        dim: Optional[int] = None, rph: Optional[RipsPH] = None) -> torch.Tensor:
    """
    Computes the singleton loss from a bar to a target point with bigstep gradient.

    Parameters
    ----------
    X : torch.Tensor
        The tensor that represents the point cloud.
    bar : ph_opt.Bar
        The bar to compute the loss from.
    target : torch.Tensor of shape (2,)
        The target point to compute the loss to.
    order : int, optional
        The order of the loss. Default is 2.
    normalize_grad : bool, optional
        If True, the gradient is normalized so that it has the same norm as the standard gradient.
        Default is True.
    dim : int, optional
        The dimension of the bar.
        Set as optional, but when it is not provided, an error will be raised.
    rph : ph_opt.RipsPH, optional
        The Rips persistent homology object.
        Set as optional, but when it is not provided, an error will be raised.

    Returns
    -------
    loss: torch.Tensor
        The singleton loss.
    """
    dist_mat = torch.cdist(X, X)
    return _singleton_loss_from_bar_to_target_with_bigstep_grad.apply(
        dist_mat, bar, target, order, normalize_grad, dim, rph
    )


class _singleton_loss_from_bar_to_target_with_continuation_grad(Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, bar: Bar, target: torch.Tensor, 
                order: int = 2, normalize_grad: bool = True):
        # save the variables in the context
        ctx.save_for_backward(X, target)
        ctx.bar, ctx.order, ctx.normalize_grad = bar, order, normalize_grad

        # Compute the singleton loss
        loss = _singleton_loss_from_bar_to_target(X, bar, target, order)

        return loss
    
    @staticmethod
    def backward(ctx, dF_df):
        # retrieve the saved tensors
        X: torch.Tensor
        target: torch.Tensor
        X, target = ctx.saved_tensors

        # retrieve the saved context
        bar: Bar
        order: int
        normalize_grad: bool
        bar, order, normalize_grad = ctx.bar, ctx.order, ctx.normalize_grad

        # Get \phi(X, ref_pd) (i.e., minus of the direction to move
        phi_birth = - (target[0] - bar.birth_time)
        phi_death = - (target[1] - bar.death_time)

        # compute d(b_time - b_target) / dX = d(b_time) / dX and d(d_time - d_target) / dX = d(d_time) / dX
        bv1, bv2 = bar.birth_v1, bar.birth_v2
        dv1, dv2 = bar.death_v1, bar.death_v2
        with torch.enable_grad():
            b_time: torch.Tensor = torch.norm(X[bv1] - X[bv2])
            jacobi_birth = torch.autograd.grad(b_time, X)[0]
            d_time: torch.Tensor = torch.norm(X[dv1] - X[dv2])
            jacobi_death = torch.autograd.grad(d_time, X)[0]

        # get phi and jacobi as torch.Tensor
        phi = torch.stack([phi_birth, phi_death], dim=0) # (2,)
        jacobi = torch.stack([jacobi_birth, jacobi_death], dim=0) # (2, num_pts, pts_dim)

        # compute (something like) gradient using the pseudo inverse of the jacobian
        jacobi_flatten = jacobi.view(-1, jacobi.size(1) * jacobi.size(2)) # (2, num_pts * pts_dim)
        jacobi_pinv = torch.linalg.pinv(jacobi_flatten).transpose(0, 1).view_as(jacobi) # (2, num_pts, pts_dim)
        df_dX = torch.tensordot(jacobi_pinv, phi, dims=([0], [0])) # (num_pts, pts_dim)

        # normalize the gradient to have the same norm as the standard gradient
        if normalize_grad:
            standard_df_dX = _get_standard_gradient_for_singleton(X, bar, target, order)
            standard_df_dX_norm = standard_df_dX.norm()
            df_dX_norm = df_dX.norm()
            if df_dX_norm > 0:
                df_dX = df_dX * (standard_df_dX_norm / df_dX_norm)

        # compute dF_dX
        dF_dX = dF_df * df_dX

        return dF_dX, None, None, None, None
    
def singleton_loss_from_bar_to_target_with_continuation_grad(X: torch.Tensor, bar: Bar,
                                                             target: torch.Tensor, order: int = 2,
                                                             normalize_grad: bool = True) -> torch.Tensor:
    """
    Computes the singleton loss from a bar to a target point with continuation gradient.

    Parameters
    ----------
    X : torch.Tensor
        The tensor that represents the point cloud.
    bar : ph_opt.Bar
        The bar to compute the loss from.
    target : torch.Tensor of shape (2,)
        The target point to compute the loss to.
    order : int, optional
        The order of the loss. Default is 2.
    normalize_grad : bool, optional
        If True, the gradient is normalized so that it has the same norm as the standard gradient.
        Default is True.

    Returns
    -------
    loss: torch.Tensor
        The singleton loss.
    """
    return _singleton_loss_from_bar_to_target_with_continuation_grad.apply(
        X, bar, target, order, normalize_grad
    )

class _singleton_loss_from_bar_to_target_with_diffeo_grad(Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, bar: Bar, target: torch.Tensor, 
                order: int = 2, normalize_grad: bool = True, 
                sigma: float = 0.1, lmbd: float=1e-5, all_X: Optional[torch.Tensor] = None) -> torch.Tensor:
        # save the variables in the context
        if all_X is not None:
            ctx.save_for_backward(X, target, all_X)
        else:
            ctx.save_for_backward(X, target)
        ctx.bar, ctx.order, ctx.normalize_grad, ctx.sigma, ctx.lmbd = bar, order, normalize_grad, sigma, lmbd

        # Compute the singleton loss
        loss = _singleton_loss_from_bar_to_target(X, bar, target, order)

        return loss
    
    @staticmethod
    def backward(ctx, dF_df): # F: whole network, f: output of this function, X: **all_X**
        # retrieve the saved tensors
        X: torch.Tensor
        target: torch.Tensor
        all_X: torch.Tensor
        all_X_provided = len(ctx.saved_tensors) == 3
        if all_X_provided:
            X, target, all_X = ctx.saved_tensors
        else:
            X, target = ctx.saved_tensors
            all_X = X

        # retrieve the saved context
        bar: torch.Tensor
        order: int
        normalize_grad: bool
        sigma: float
        lmbd: float
        bar, order, normalize_grad, sigma, lmbd = ctx.bar, ctx.order, ctx.normalize_grad, ctx.sigma, ctx.lmbd

        # compute the standard gradient
        standard_df_dX = _get_standard_gradient_for_singleton(X, bar, target, order)

        # no gradient if the standard gradient is zero
        if standard_df_dX.norm() == 0:
            if all_X_provided:
                dF_dX = torch.zeros_like(all_X)
                return None, None, None, None, None, None, None, dF_dX
            else:
                dF_dX = torch.zeros_like(X)
                return dF_dX, None, None, None, None, None, None, None

        # compute the new gradient using diffeomorphic interpolation
        num_pts, pts_dim = X.size()
        nonzero_grad_idx = [idx for idx in range(num_pts) if standard_df_dX[idx, :].norm() > 0]
        nonzero_grad_X = X[nonzero_grad_idx, :]
        a_vec = torch.cat([standard_df_dX[i, :] for i in nonzero_grad_idx], dim=0) # corresponds to `a` in the paper
        _K_mat = torch.exp(- torch.cdist(nonzero_grad_X, nonzero_grad_X) ** 2 / (2 * sigma ** 2)) # the Gaussian kernel matrix
        K_mat = torch.kron(_K_mat, torch.eye(pts_dim)) # corresponds to `K` in the paper
        _alpha: torch.Tensor = torch.linalg.solve(K_mat + lmbd * torch.eye(K_mat.size(0)), a_vec)
        alpha = _alpha.view(len(nonzero_grad_idx), pts_dim) # corresponds to `alpha` in the paper
        rho_mat = torch.exp(- torch.cdist(all_X, nonzero_grad_X) ** 2 / (2 * sigma ** 2)) # shape = (num_all_pts, num_nonzero_grad_pts)
        df_dX = torch.matmul(rho_mat, alpha) # "ij,jk->ik", shape = (num_all_pts, pts_dim)

        # normalize_grad the gradient to have the same norm as the standard gradient
        df_dX_norm = df_dX.norm()
        if (df_dX_norm > 0) and normalize_grad:
            standard_df_dX_norm = standard_df_dX.norm()
            df_dX = df_dX * (standard_df_dX_norm / df_dX_norm)

        # compute dF_dX
        dF_dX = dF_df * df_dX
        
        if all_X_provided:
            return None, None, None, None, None, None, None, dF_dX
        else:
            return dF_dX, None, None, None, None, None, None, None

def singleton_loss_from_bar_to_target_with_diffeo_grad(X: torch.Tensor, bar: Bar, target: torch.Tensor, 
                                                       order: int = 2, normalize_grad: bool = True, 
                                                       sigma: float = 0.1, lmbd: float=1e-5, all_X: Optional[torch.Tensor] = None
                                                       ) -> torch.Tensor:
    """
    Computes the singleton loss from a bar to a target point with diffeomorphic gradient.

    Parameters
    ----------
    X : torch.Tensor
        The tensor that represents the point cloud.
    bar : ph_opt.Bar
        The bar to compute the loss from.
    target : torch.Tensor of shape (2,)
        The target point to compute the loss to.
    order : int, optional
        The order of the loss. Default is 2.
    normalize_grad : bool, optional
        If True, the gradient is normalized so that it has the same norm as the standard gradient.
        Default is True.
    sigma : float, optional
        The bandwidth of the Gaussian kernel. Default is 0.1.
    lmbd: float, optional
        The kernel regularization parameter. Default is 1e-5.
    all_X : torch.Tensor, optional
        The tensor that represents the point cloud. If None, X is used. Default is None.

    Returns
    -------
    loss: torch.Tensor
        The singleton loss.
    """
    return _singleton_loss_from_bar_to_target_with_diffeo_grad.apply(
        X, bar, target, order, normalize_grad, sigma, lmbd, all_X
    )

def singleton_loss_from_bar_to_target(X: torch.Tensor, bar: Bar, target: torch.Tensor, 
                                      grad_type: str = "standard",
                                      order: int = 2, normalize_grad: bool = True, 
                                      dim: Optional[int] = None, rph: Optional[RipsPH] = None,
                                      sigma: float = 0.1, lmbd: float = 1e-5, all_X: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the singleton loss from a bar to a target point.

    Parameters
    ----------
    X : torch.Tensor
        The tensor that represents the point cloud.
    bar : ph_opt.Bar
        The bar to compute the loss from.
    target : torch.Tensor of shape (2,)
        The target point to compute the loss to. 
    grad_type : str, optional
        The type of gradient to use.
    order : int, optional
        The order of the loss. Default is 2.
    normalize_grad : bool, optional
        If True, the gradient is normalized so that it has the same norm as the standard gradient.
        Default is True.
    dim : int, optional
        The dimension of the bar. Only used when `method` is 'bigstep'.
    rph : ph_opt.RipsPH, optional
        The Rips persistent homology object. Only used when `method` is 'bigstep'.
    sigma : float, optional
        the bandwidth of the Gaussian kernel. Only used when `method` is 'diffeo'.
    lmbd: float, optional
        The regularization parameter. Only used when `method` is 'diffeo'.
    all_X : torch.Tensor of size=(# of points, dim), optional
        The whole point cloud for diffeomorphic interpolation.
        Only used when `method` is 'diffeo' and `all_X` is not `None`.

    Returns
    -------
    loss: torch.Tensor
        The singleton loss.

    Notes
    -----
    - If `grad_type` is "bigstep", the attributes `birth_simp` and `death_simp` of the bar are necessary since they are used to compute the loss.
    - If `grad_type` is "bigstep", `dim` and `rph` must be provided.
    """
    if grad_type == "standard":
        return _singleton_loss_from_bar_to_target(X, bar, target, order)
    elif grad_type == "bigstep":
        return singleton_loss_from_bar_to_target_with_bigstep_grad(X, bar, target, order, normalize_grad, 
                                                                   dim, rph)
    elif grad_type == "continuation":
        return singleton_loss_from_bar_to_target_with_continuation_grad(X, bar, target, order, normalize_grad)
    elif grad_type == "diffeo":
        return singleton_loss_from_bar_to_target_with_diffeo_grad(X, bar, target, order, normalize_grad, 
                                                                  sigma, lmbd, all_X)
    else:
        raise ValueError(f"Unknown grad_type: {grad_type}. Must be one of ['standard', 'bigstep', 'continuation', 'diffeo'].")
                                                                  
from ..ph_compute.ph_computation_library import RipsPH, Bar

import torch
from torch.autograd import Function
from gudhi.wasserstein import wasserstein_distance

def _powered_wasserstein_distance_from_rph(rph: RipsPH, ref_pd: list[torch.Tensor], dims: list[int], order: int = 2):
    # get the barcode
    differentiable_barcode = [rph.get_differentiable_barcode(dim) for dim in dims]

    # compute the loss
    loss = torch.tensor(0.)
    for dim_idx, dim in enumerate(dims):
        loss += wasserstein_distance(differentiable_barcode[dim], ref_pd[dim_idx], 
                                     order=order, enable_autodiff=True, keep_essential_parts=False) ** 2
        
    return loss

def _get_rph_and_powered_wasserstein_distance_one_sided(X: torch.Tensor, ref_pd: list[torch.Tensor], dims: list[int], order: int = 2, 
                                                        distance_matrix: bool = False, need_V_and_W: bool = False):
    # compute RipsPH
    maxdim = max(dims)
    rph = RipsPH(X, maxdim=maxdim, distance_matrix=distance_matrix)

    # if need_V_and_W is True, compute PH with access to V and W
    if need_V_and_W:
        rph.compute_ph(clearing_opt=False, get_inv=True)
        rph.compute_ph_right(get_inv=True)

    # compute the loss
    loss = _powered_wasserstein_distance_from_rph(rph, ref_pd, dims, order)

    return rph, loss

def _powered_wasserstein_distance_one_sided(X: torch.Tensor, ref_pd: list[torch.Tensor], dims: list[int], order: int = 2, 
                                            distance_matrix: bool = False):
    _, loss = _get_rph_and_powered_wasserstein_distance_one_sided(X, ref_pd, dims, order, distance_matrix)
    return loss

def _get_standard_gradient(X: torch.Tensor, rph: RipsPH, ref_pd: list[torch.Tensor], dims: list[int], order: int = 2, 
                           distance_matrix: bool = False):
    with torch.enable_grad():
        _X = X.detach().clone().requires_grad_()
        _loss = _powered_wasserstein_distance_one_sided(_X, ref_pd, dims, order, distance_matrix)
    standard_df_dX, = torch.autograd.grad(outputs=_loss, inputs=(_X,), retain_graph=False, create_graph=False)

    return standard_df_dX

def _get_direction_for_wasserstein(rph: RipsPH, ref_pd: list[torch.Tensor], dims: list[int], order: int) -> list[tuple[int, Bar, torch.Tensor]]:
    ret = []
    for dim, _ref_pd in zip(dims, ref_pd):
        bars = rph.get_bar_object_list(dim)
        barcode = torch.tensor([[bar.birth_time, bar.death_time] for bar in bars])
        _, matching = wasserstein_distance(barcode, _ref_pd, order=order, matching=True, keep_essential_parts=False)
        for i, j in matching:
            if i == -1:
                continue
            elif j == -1:
                bar_to_move = bars[i]
                persistence = barcode[i, 1] - barcode[i, 0]
                direction = 0.5 * persistence * torch.tensor([1., -1.])
            else:
                bar_to_move = bars[i]
                direction = _ref_pd[j] - barcode[i]
            ret.append((dim, bar_to_move, direction))

    return ret

class _powered_wasserstein_distance_one_sided_with_bigstep_grad(Function):
    @staticmethod
    def forward(ctx, dist_mat: torch.Tensor, ref_pd: list[torch.Tensor], dims: list[int], order: int = 2):
        ctx.save_for_backward(dist_mat)
        ctx.ref_pd, ctx.dims, ctx.order = ref_pd, dims, order

        # get rph and the loss
        rph, loss = _get_rph_and_powered_wasserstein_distance_one_sided(dist_mat, ref_pd, dims, order, 
                                                                        distance_matrix=True, need_V_and_W=True)

        # save the rph in the context
        ctx.rph = rph
        
        return loss

    @staticmethod
    def backward(ctx, dF_df): # F: whole network, f: output of this function, D: dist_mat
        # retrieve the saved tensors
        dist_mat: torch.Tensor
        dist_mat, = ctx.saved_tensors

        # retrieve the saved context
        ref_pd: list[torch.Tensor]
        dims: list[int]
        order: int
        rph: RipsPH
        ref_pd, dims, order, rph = ctx.ref_pd, ctx.dims, ctx.order, ctx.rph

        # get the directions of the bars to move
        direction_info = _get_direction_for_wasserstein(rph, ref_pd, dims, order)

        # compute the standard gradient
        standard_df_dD = _get_standard_gradient(dist_mat, rph, ref_pd, dims, order, distance_matrix=True)

        # compute the target value for each simplex
        maxdim = max(dims)
        low_target_value = [{} for dim in range(maxdim+2)]
        high_target_value = [{} for dim in range(maxdim+2)]
        for dim, bar, direction in direction_info:
            b_simp, d_simp = bar.birth_simp, bar.death_simp
            b_direc, d_direc = direction * torch.sign(dF_df)
            b_target, d_target = bar.birth_time + b_direc, bar.death_time + d_direc
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
        for dim in range(maxdim+2):
            moving_simplices = set(high_target_value[dim].keys()) | set(low_target_value[dim].keys())
            for simp in moving_simplices:
                # get the max edge and the diameter of the simplex
                v1, v2 = rph.get_max_edge(dim, simp)
                diam = dist_mat[v1, v2]

                # get the target value
                if (simp in high_target_value[dim]) and (simp in low_target_value[dim]):
                    # target is the farthest one
                    high_diff = torch.relu(high_target_value[dim][simp] - diam)
                    low_diff = torch.relu(diam - low_target_value[dim][simp])
                    if high_diff > low_diff:
                        target_value = high_target_value[dim][simp]
                    else:
                        target_value = low_target_value[dim][simp]
                elif simp in high_target_value[dim]:
                    target_value = high_target_value[dim][simp]
                elif simp in low_target_value[dim]:
                    target_value = low_target_value[dim][simp]

                # compute the gradient (note: doubled by the symmetry)
                df_dD[v1, v2] += diam - target_value
                df_dD[v2, v1] += diam - target_value

        # normalize the gradient to have the same norm as the standard gradient
        standard_df_dD_norm = standard_df_dD.norm()
        df_dD_norm = df_dD.norm()
        if df_dD_norm > 0:
            # normalize the gradient to have the same norm as the standard gradient
            df_dD = df_dD * (standard_df_dD_norm / df_dD_norm)

        # compute dF_dD
        dF_dD = torch.abs(dF_df) * df_dD

        return dF_dD, None, None, None

def powered_wasserstein_distance_one_sided_with_bigstep_grad(X: torch.Tensor, ref_pd: list[torch.Tensor], dims: list[int], order: int = 2):
    """
    Compute the Wasserstein distance between the persistent diagram of `X` and `ref_pd`
    with improved gradient using big step method.

    Parameters:
        X (torch.Tensor) : point cloud. shape=(# of points, dim)
        ref_pd (list[torch.Tensor]) : the reference persistent diagram. shape=(#bars, 2)
        dims (list[int]) : list of dimensions of the persistent homology.
        order (int) : the order of Wasserstein distance.
        
    Returns:
        loss (torch.Tensor) : the Wasserstein distance.
    """
    dist_mat = torch.cdist(X, X)
    return _powered_wasserstein_distance_one_sided_with_bigstep_grad.apply(dist_mat, ref_pd, dims, order)
    
class _powered_wasserstein_distance_one_sided_with_continuation_grad(Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, ref_pd: list[torch.Tensor], dims: list[int], order: int = 2):
        ctx.save_for_backward(X)
        ctx.ref_pd, ctx.dims, ctx.order = ref_pd, dims, order

        # get rph and the loss
        rph, loss = _get_rph_and_powered_wasserstein_distance_one_sided(X, ref_pd, dims, order)

        # save the rph in the context
        ctx.rph = rph
        
        return loss
    
    @staticmethod
    def backward(ctx, dF_df):
        # retrieve the saved tensors
        X: torch.Tensor
        X, = ctx.saved_tensors

        # retrieve the saved context
        ref_pd: list[torch.Tensor]
        dims: list[int]
        order: int
        rph: RipsPH
        ref_pd, dims, order, rph = ctx.ref_pd, ctx.dims, ctx.order, ctx.rph

        # compute the standard gradient
        standard_df_dX = _get_standard_gradient(X, rph, ref_pd, dims, order)

        # get the directions of the bars to move
        direction_info = _get_direction_for_wasserstein(rph, ref_pd, dims, order)

        # compute \phi(X, ref_pd) = Pers(\Phi(X)) - ref_pd and its jacobian matrix
        phi: list[torch.Tensor] = []
        _jacobi: list[torch.Tensor] = []
        for dim, bar, direction in direction_info:
            bv1, bv2 = bar.birth_v1, bar.birth_v2
            dv1, dv2 = bar.death_v1, bar.death_v2
            b_direc, d_direc = direction

            # append the values of \phi(X, ref_pd) (i.e., - b_direc, - d_direc for the birth and death, respectively)
            phi.append(b_direc)
            phi.append(d_direc)

            # compute d(b_time - b_target) / dX = d(b_time) / dX and append it to _jacobi
            b_time: torch.Tensor = torch.norm(X[bv1] - X[bv2])
            _jacobi.append(torch.autograd.grad(b_time, X)[0])

            # compute d(d_time - d_target) / dX = d(d_time) / dX and append it to _jacobi
            d_time: torch.Tensor = torch.norm(X[dv1] - X[dv2])
            _jacobi.append(torch.autograd.grad(d_time, X)[0])

        # get phi and jacobi as torch.Tensor
        phi = torch.stack(phi, dim=0).detach() # (2 * #pairs,)
        jacobi = torch.stack(_jacobi, dim=0) # (2 * #pairs, num_pts, pts_dim)

        # compute (something like) gradient using the pseudo inverse of the jacobian
        jacobi_flatten = jacobi.view(-1, jacobi.size(1) * jacobi.size(2)) # (2 * #pairs, num_pts * pts_dim)
        jacobi_pinv = torch.linalg.pinv(jacobi_flatten).transpose(0, 1).view_as(jacobi) # (2 * #pairs, num_pts, pts_dim)
        df_dX = torch.tensordot(jacobi_pinv, phi, dims=([0], [0])) # (num_pts, pts_dim)

        # normalize the gradient to have the same norm as the standard gradient
        standard_df_dX_norm = standard_df_dX.norm()
        df_dX_norm = df_dX.norm()
        if df_dX_norm > 0:
            df_dX = df_dX * (standard_df_dX_norm / df_dX_norm)

        # compute dF_dX
        dF_dX = dF_df * df_dX

        return dF_dX, None, None, None
    
def powered_wasserstein_distance_one_sided_with_continuation_grad(X: torch.Tensor, ref_pd: list[torch.Tensor], dims: list[int], order: int = 2):
    """
    Compute the Wasserstein distance between the persistent diagram of `X` and `ref_pd`
    with improved gradient using continuation method.

    Parameters:
        X (torch.Tensor) : point cloud. shape=(# of points, dim)
        ref_pd (list[torch.Tensor]) : the reference persistent diagram. shape=(#bars, 2)
        dims (list[int]) : list of dimensions of the persistent homology.
        order (int) : the order of Wasserstein distance.
        
    Returns:
        loss (torch.Tensor) : the Wasserstein distance.
    """
    return _powered_wasserstein_distance_one_sided_with_continuation_grad.apply(X, ref_pd, dims, order)

class _powered_wasserstein_distance_one_sided_with_diffeo_grad(Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, ref_pd: list[torch.Tensor], dims: list[int], order: int = 2, sigma: float = 0.1):
        ctx.save_for_backward(X)
        ctx.ref_pd, ctx.dims, ctx.order, ctx.sigma = ref_pd, dims, order, sigma

        # get rph and the loss
        rph, loss = _get_rph_and_powered_wasserstein_distance_one_sided(X, ref_pd, dims, order)

        # save the rph in the context
        ctx.rph = rph
        
        return loss
    
    @staticmethod
    def backward(ctx, dF_df): # F: whole network, f: output of this function, X: input point cloud
        # retrieve the saved tensors
        X: torch.Tensor
        X, = ctx.saved_tensors

        # retrieve the saved context
        ref_pd: list[torch.Tensor]
        dims: list[int]
        order: int
        sigma: float
        rph: RipsPH
        ref_pd, dims, order, sigma, rph = ctx.ref_pd, ctx.dims, ctx.order, ctx.sigma, ctx.rph

        # compute the standard gradient
        standard_df_dX = _get_standard_gradient(X, rph, ref_pd, dims, order)

        # compute the new gradient using diffeomorphic interpolation
        num_pts, pts_dim = X.size()
        nonzero_grad_idx = [idx for idx in range(num_pts) if standard_df_dX[idx, :].norm() > 0]
        nonzero_grad_X = X[nonzero_grad_idx, :]
        a_vec = torch.cat([standard_df_dX[i, :] for i in nonzero_grad_idx], dim=0) # corresponds to `a` in the paper
        _K_mat = torch.exp(- torch.cdist(nonzero_grad_X, nonzero_grad_X) ** 2 / (2 * sigma ** 2)) # the Gaussian kernel matrix
        K_mat = torch.kron(_K_mat, torch.eye(pts_dim)) # corresponds to `K` in the paper
        _alpha: torch.Tensor = torch.linalg.solve(K_mat, a_vec)
        alpha = _alpha.view(len(nonzero_grad_idx), pts_dim) # corresponds to `alpha` in the paper
        rho_mat = torch.exp(- torch.cdist(X, nonzero_grad_X) ** 2 / (2 * sigma ** 2)) # shape = (num_pts, num_nonzero_grad_pts)
        df_dX = torch.matmul(rho_mat, alpha) # "ij,jk->ik", shape = (num_pts, pts_dim)

        # normalize the gradient to have the same norm as the standard gradient
        standard_df_dX_norm = standard_df_dX.norm()
        df_dX_norm = df_dX.norm()
        if df_dX_norm > 0:
            # normalize the gradient to have the same norm as the standard gradient
            df_dX = df_dX * (standard_df_dX_norm / df_dX_norm)

        # compute dF_dX
        dF_dX = dF_df * df_dX
        
        return dF_dX, None, None, None, None 
    
def powered_wasserstein_distance_one_sided_with_diffeo_grad(X: torch.Tensor, ref_pd: list[torch.Tensor], dims: list[int],
                                                            order: int = 2, sigma: float = 0.1):
    """
    Compute the Wasserstein distance between the persistent diagram of `X` and `ref_pd`
    with improved gradient using diffeomorphic interpolation.

    Parameters:
        X (torch.Tensor) : point cloud. shape=(# of points, dim)
        ref_pd (list[torch.Tensor]) : the reference persistent diagram. shape=(#bars, 2)
        dims (list[int]) : list of dimensions of the persistent homology.
        order (int) : the order of Wasserstein distance.
        sigma (float) : the bandwidth of the Gaussian kernel.

    Returns:
        loss (torch.Tensor) : the Wasserstein distance.
    """
    return _powered_wasserstein_distance_one_sided_with_diffeo_grad.apply(X, ref_pd, dims, order, sigma)

def powered_wasserstein_distance_one_sided_with_improved_grad(X: torch.Tensor, ref_pd: list[torch.Tensor], dims: list[int], order: int = 2, 
                                                              grad_type: str = "standard", sigma: float = 0.1):
    """
    Compute the Wasserstein distance between the persistent diagram of `X` and `ref_pd`
    with improved gradient using specialized method.

    Parameters:
        - X (torch.Tensor) : point cloud. shape=(# of points, dim)
        - ref_pd (list[torch.Tensor]) : the reference persistent diagram. shape=(#bars, 2)
        - dims (list[int]) : list of dimensions of the persistent homology.
        - order (int) : the order of Wasserstein distance.
        - grad_type (str) : the method to compute the gradient. 
            One of ['standard', 'bigstep', 'continuation', 'diffeo'].
        - sigma (float) : the bandwidth of the Gaussian kernel. Only used when `method` is 'diffeo'.
        
    Returns:
        - loss: the Wasserstein distance.
    """

    if grad_type == 'standard':
        loss = _powered_wasserstein_distance_one_sided(X, ref_pd, dims, order)
    elif grad_type == 'bigstep':
        loss = powered_wasserstein_distance_one_sided_with_bigstep_grad(X, ref_pd, dims, order)
    elif grad_type == 'continuation':
        loss = powered_wasserstein_distance_one_sided_with_continuation_grad(X, ref_pd, dims, order)
    elif grad_type == 'diffeo':
        loss = powered_wasserstein_distance_one_sided_with_diffeo_grad(X, ref_pd, dims, order, sigma)
    else:
        raise ValueError(f"Unknown grad_type: {grad_type}. Must be one of ['standard', 'bigstep', 'continuation', 'diffeo'].")
    
    return loss

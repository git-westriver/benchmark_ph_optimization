from ...ph_compute.ph_computation_library import RipsPH
from gudhi.wasserstein import wasserstein_distance
from typing import Optional
import torch

def _powered_wasserstein_distance_one_sided_from_rph_with_standard_grad(rph: RipsPH, ref_pd: list[torch.Tensor], dims: list[int], 
                                                                        order: int = 2, X: Optional[torch.Tensor] = None):
    # get the barcode
    differentiable_barcode = [rph.get_differentiable_barcode(dim, X=X) for dim in dims]

    # compute the loss
    loss = torch.tensor(0.)
    for dim_idx, dim in enumerate(dims):
        loss += wasserstein_distance(differentiable_barcode[dim_idx], ref_pd[dim_idx], 
                                     order=order, enable_autodiff=True, keep_essential_parts=False) ** order
        
    return loss
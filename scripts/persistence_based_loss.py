from typing import Optional
import torch
from gudhi.wasserstein import wasserstein_distance
from lib.ph_computation_library import RipsPH, Bar

# decorator to get/process rph
def get_rph(f):
    def _get_rph(*args, **kwargs):
        rph_given = False
        # process the arguments
        _self = args[0]
        if len(args) >= 2:
            X = args[1]
        else:
            X = kwargs['X']
        if len(args) >= 3:
            rph = args[2]
        elif 'rph' in kwargs:
            rph = kwargs['rph']
        else:
            rph = None
        # define rph if not given
        if rph is None:
            rph = RipsPH(X, _self.maxdim)
        else:
            rph_given = True
        ret = f(_self, X, rph)
        if not rph_given:
            del rph
        return ret
    return _get_rph

class PersistenceBasedLoss:
    """
    Base class for persistence-based loss.
    """
    def __init__(self, dim_list: list[int], **kwargs):
        self.dim_list = dim_list
        self.maxdim = max(dim_list)

    @get_rph
    def __call__(self, X: torch.Tensor, rph=Optional[RipsPH]) -> torch.Tensor:
        """
        Compute the loss.
        Note that when implementing this method, you need to apply the get_rph decorator.

        Parameters:
            - X(torch.Tensor): point cloud. shape=(# of points, dim)
            - rph(Optional[RipsPH]): if not `None`, the persistent homology is computed in advance.

        Returns:
            - loss(torch.Tensor): the loss value.
        """
        raise NotImplementedError

    @get_rph
    def get_direction(self, X, rph=Optional[RipsPH]) -> list[tuple[int, list[Bar], torch.Tensor]]:
        """
        Get the direction to which the points in PD should be moved.
        This function is used in BigStep and Continuation.
        Note that when implementing this method, you need to apply the get_rph decorator.
        
        Parameters:
            - X(torch.Tensor): point cloud. shape=(# of points, dim)
            - rph(Optional[RipsPH]): if not `None`, the persistent homology is computed in advance.

        Returns:  list of tuples. The `i`-th tuple corresponds to `dim_list[i]`-dimensional persistent homology. 
        Each tuple contains:
            - dim(int): equal to `dim_list[i]`
            - list of `Bar` (can be obtained through `RipsPH.get_bar_object_list`): the bars to move
            - direction to move (torch.Tensor)
        """
        raise NotImplementedError

class WassersteinLoss(PersistenceBasedLoss):
    """
    Loss to minimize the Wasserstein distance between the persistent diagram of the input and a desirable persistent diagram.

    Parameters:
        - dim_list(list[int]): list of dimensions of the persistent homology.
        - desirable_pd(list[torch.Tensor]): the desirable persistent diagram. shape=(#bars, 2)
        - order(int, default=2): the order of Wasserstein distance.
    """
    def __init__(self, dim_list: list[int], desirable_pd: list[torch.Tensor], order=2):
        super().__init__(dim_list)
        self.desirable_pd = desirable_pd
        self.order = order

    @get_rph
    def __call__(self, X: torch.Tensor, rph=Optional[RipsPH]) -> torch.Tensor:
        loss = torch.tensor(0.)
        for dim_idx, dim in enumerate(self.dim_list):
            barcode: torch.Tensor = rph.get_differentiable_barcode(dim)
            loss += wasserstein_distance(barcode, self.desirable_pd[dim_idx], order=self.order, enable_autodiff=True, keep_essential_parts=False)
        return loss
    
    @get_rph
    def get_direction(self, X, rph=Optional[RipsPH]) -> list[tuple[int, list[Bar], torch.Tensor]]:
        ret = []
        for dim_idx, dim in enumerate(self.dim_list):
            bars_to_move, direction = [], []
            bar_list = rph.get_bar_object_list(dim)
            barcode = torch.tensor([[bar.birth_time, bar.death_time] for bar in bar_list])
            _, matching = wasserstein_distance(barcode, self.desirable_pd[dim_idx], order=self.order, matching=True, keep_essential_parts=False)
            for i, j in matching:
                if i == -1:
                    continue
                elif j == -1:
                    bars_to_move.append(bar_list[i])
                    diff = barcode[i, 1] - barcode[i, 0]
                    direction.append(diff * torch.tensor([0.5, -0.5]))
                else:
                    bars_to_move.append(bar_list[i])
                    direction.append(self.desirable_pd[dim_idx][j] - barcode[i])

            if direction:
                direction_stack = torch.stack(direction, dim=0)
            else:
                direction_stack = torch.empty(0, 2)
            ret.append((dim, bars_to_move, direction_stack))
            
        return ret

class ExpandLoss(PersistenceBasedLoss):
    """
    Loss to expand the holes in the point cloud, i.e., make the points in the PD far from the diagonal.
    - Parameters
        - dim_list(list[int]): list of dimensions of the persistent homology.
        - order(int, default=1): the order of Wasserstein distance.
        - eps(float, default=0.): if not `None`, the points in the PD with lifetime less than `eps` will be ignored.
        - topk(Optional[int], default=None): if not `None`, the points in PD are sorted by lifetime in descending order, and only the top k points are considered.
    """
    def __init__(self, dim_list: list[int], order: int=1, eps: float=0., topk: Optional[int]=None):
        super().__init__(dim_list)
        self.order = order
        self.eps = eps
        self.topk = topk

        self.max_dim = max(dim_list)

    @get_rph
    def __call__(self, X: torch.Tensor, rph=Optional[RipsPH]) -> torch.Tensor:
        loss = torch.tensor(0.)
        for dim in self.dim_list:
            barcode: torch.Tensor = rph.get_differentiable_barcode(dim)
            bar_list = [barcode[i] for i in range(barcode.size(0)) 
                        if self.eps is None or barcode[i, 1] - barcode[i, 0] > self.eps]
            if self.topk is not None:
                bar_list = sorted(bar_list, key=lambda x: x[1] - x[0], reverse=True)[:self.topk]
            loss -= sum([(bar[1] - bar[0]) ** self.order for bar in bar_list])
        return loss

    @get_rph
    def get_direction(self, X, rph=Optional[RipsPH]):
        ret: list[tuple[int, Bar, torch.Tensor]] = []
        for dim in self.dim_list:
            bar_list = rph.get_bar_object_list(dim)
            # filter bars by eps and topk
            bar_list = [bar for bar in bar_list if bar.death_time - bar.birth_time > self.eps]
            if self.topk is not None:
                bar_list = sorted(bar_list, key=lambda bar: bar.death_time - bar.birth_time, reverse=True)[:self.topk]
            # specify the direction for target bars
            direction = torch.tensor([[-1., 1.] for _ in range(len(bar_list))])
            # add to the return value
            ret.append((dim, bar_list, direction))
        return ret

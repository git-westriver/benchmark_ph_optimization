from typing import Optional
import torch

from ..ph_compute.ph_computation_library import RipsPH, Bar
from .persistence_based_loss import PersistenceBasedLoss
from .regularization import Regularization
from ph_opt.scheduler import TransformerLR

class PHOptimization:
    def __init__(self, X: torch.Tensor, loss_obj: PersistenceBasedLoss, reg_obj: Optional[Regularization]=None, reg_proj: bool=False):
        self.num_pts = X.shape[0]
        self.X = X
        if not self.X.requires_grad:
            raise ValueError("X.requires_grad must be True")
        self.loss_obj = loss_obj
        self.reg_obj = reg_obj
        self.reg_proj = reg_proj
    
    def get_loss(self) -> torch.Tensor:
        loss = self.loss_obj(self.X)
        if (self.reg_obj is not None) and (not self.reg_proj):
            loss += self.reg_obj(self.X)
        return loss

class GradientDescent(PHOptimization):
    def __init__(self, X: torch.Tensor, loss_obj: PersistenceBasedLoss, reg_obj: Optional[Regularization]=None, reg_proj: bool=False, 
                 lr: float=1e-1, optimizer_conf: Optional[dict[str]]=None, scheduler_conf: Optional[dict[str]]=None):
        super().__init__(X, loss_obj, reg_obj, reg_proj)
        ### optimizer ###
        if optimizer_conf is None:
            optimizer_conf = {"name": "SGD"}
        optimizer_name = optimizer_conf.get("name", "SGD")
        if optimizer_name == "SGD":
            self.optimizer = torch.optim.SGD([self.X], lr=lr)
        elif optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam([self.X], lr=lr)
        ### scheduler ###
        if scheduler_conf is None:
            scheduler_conf = {"name": "const"}
        scheduler_name = scheduler_conf.get("name", "const")
        if scheduler_name == "const":
            self.scheduler = None
        elif scheduler_name == "TransformerLR":
            self.scheduler = TransformerLR(self.optimizer, warmup_epochs=scheduler_conf.get("warmup_epochs", 100))

    def update(self) -> None:
        self.optimizer.zero_grad()
        loss = self.get_loss()
        if loss.grad_fn is None:
            return
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        ### project to the region of regularization = 0 ###
        if self.reg_proj:
            self.do_projection()

class BigStep(PHOptimization):
    def __init__(self, X: torch.Tensor, loss_obj: PersistenceBasedLoss, reg_obj: Optional[Regularization]=None, reg_proj: bool=False, 
                 lr: float=1e-1, optimizer_conf: Optional[dict[str]]=None, scheduler_conf: Optional[dict[str]]=None):
        super().__init__(X, loss_obj, reg_obj, reg_proj)
        ### optimizer ###
        if optimizer_conf is None:
            optimizer_conf = {"name": "SGD"}
        optimizer_name = optimizer_conf.get("name", "SGD")
        if optimizer_name == "SGD":
            self.optimizer = torch.optim.SGD([self.X], lr=lr)
        elif optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam([self.X], lr=lr)
        ### scheduler ###
        if scheduler_conf is None:
            scheduler_conf = {"name": "const"}
        scheduler_name = scheduler_conf.get("name", "const")
        if scheduler_name == "const":
            self.scheduler = None
        elif scheduler_name == "TransformerLR":
            self.scheduler = TransformerLR(self.optimizer, warmup_epochs=scheduler_conf.get("warmup_epochs", 100))

    def update(self) -> None:
        self.optimizer.zero_grad()
        ### compute the gradient of the persistence based loss at the current point ###
        loss = self.loss_obj(self.X)
        if loss.grad_fn is None:
            return
        loss.backward()
        gen_grad_norm = torch.norm(self.X.grad)
        self.optimizer.zero_grad()
        ### get the target value of each simplex ###
        rph = RipsPH(self.X, self.loss_obj.maxdim)
        rph.compute_ph(clearing_opt=False, get_inv=True)
        rph.compute_ph_right(get_inv=True)
        low_target_value = [{} for dim in range(self.loss_obj.maxdim+2)]
        high_target_value = [{} for dim in range(self.loss_obj.maxdim+2)]
        direction_info = self.loss_obj.get_direction(self.X, rph)
        for dim, bar_list, direction in direction_info:
            for i in range(len(bar_list)):
                b_simp, d_simp = bar_list[i].birth_simp, bar_list[i].death_simp
                b_direc, d_direc = direction[i]
                b_target, d_target = bar_list[i].birth_time + b_direc, bar_list[i].death_time + d_direc
                if b_direc > 0: # increase birth
                    print(dim)
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
        ### Compute Auxilary Loss ###
        aux_loss = 0
        for dim in range(self.loss_obj.maxdim+2):
            moving_simp = set(high_target_value[dim].keys()) | set(low_target_value[dim].keys())
            for simp in moving_simp:
                diam: torch.Tensor = rph.get_differentiable_diameter(dim, simp) # this should be differentiable
                if (simp in high_target_value[dim]) and (simp in low_target_value[dim]):
                    # target is the farthest from the current value
                    high_diff = torch.relu(high_target_value[dim][simp] - diam)
                    low_diff = torch.relu(diam - low_target_value[dim][simp])
                    if high_diff > low_diff:
                        aux_loss += torch.relu(high_target_value[dim][simp] - diam) ** 2
                    elif high_diff < low_diff:
                        aux_loss += torch.relu(diam - low_target_value[dim][simp]) ** 2
                elif simp in high_target_value[dim]:
                    aux_loss += torch.relu(high_target_value[dim][simp] - diam) ** 2
                elif simp in low_target_value[dim]:
                    aux_loss += torch.relu(diam - low_target_value[dim][simp]) ** 2
        ### Rescale Auxilary Loss to make the gradient norm of the auxilary loss equal to that of the persistence based loss ###
        if aux_loss.grad_fn is None:
            return
        aux_loss.backward(retain_graph=True)
        aux_grad_norm = torch.norm(self.X.grad)
        self.optimizer.zero_grad()
        aux_loss = aux_loss * gen_grad_norm / aux_grad_norm if aux_grad_norm > 0 else aux_loss
        ### add the regularization, compute the gradient, and update the parameters ###
        aux_loss = aux_loss + self.reg_obj(self.X)
        aux_loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        ### we must delete rph to avoid memory leak ###
        del rph
        ### project to the region of regularization = 0 ###
        if self.reg_proj:
            self.X = self.reg_obj.projection(self.X)

class Diffeo(PHOptimization):
    def __init__(self, X: torch.Tensor, loss_obj: PersistenceBasedLoss, reg_obj: Optional[Regularization]=None, reg_proj: bool=False, 
                 lr: float=1e-1, sigma: float=0.1, optimizer_conf: Optional[dict[str]]=None, scheduler_conf: Optional[dict[str]]=None):
        super().__init__(X, loss_obj, reg_obj, reg_proj)
        ### optimizer ###
        if optimizer_conf is None:
            optimizer_conf = {"name": "SGD"}
        optimizer_name = optimizer_conf.get("name", "SGD")
        if optimizer_name == "SGD":
            self.optimizer = torch.optim.SGD([self.X], lr=lr)
        elif optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam([self.X], lr=lr)
        ### scheduler ###
        if scheduler_conf is None:
            scheduler_conf = {"name": "const"}
        scheduler_name = scheduler_conf.get("name", "const")
        if scheduler_name == "const":
            self.scheduler = None
        elif scheduler_name == "TransformerLR":
            self.scheduler = TransformerLR(self.optimizer, warmup_epochs=scheduler_conf.get("warmup_epochs", 100))
        ### unique parameter ###
        self.sigma = sigma # the standard deviation of the Gaussian kernel

    def update(self) -> None:
        self.optimizer.zero_grad()
        ### compute the gradient of the persistence based loss at the current point ###
        loss = self.loss_obj(self.X)
        if loss.grad_fn is None:
            return
        loss.backward()
        gen_grad = self.X.grad.clone()
        gen_grad_norm = torch.norm(self.X.grad)
        ### compute the new gradient using diffeomorphic interpolation ###
        nonzero_grad_idx = [idx for idx in range(self.num_pts) if gen_grad[idx, :].norm() > 0]
        nonzero_grad_X = self.X[nonzero_grad_idx, :]
        a_vec = torch.cat([gen_grad[i, :] for i in nonzero_grad_idx], dim=0) # corresponds to `a` in the paper
        _K_mat = torch.exp(- torch.cdist(nonzero_grad_X, nonzero_grad_X) ** 2 / (2 * self.sigma ** 2)) # the Gaussian kernel matrix
        K_mat = torch.kron(_K_mat, torch.eye(self.X.shape[1])) # corresponds to `K` in the paper
        _alpha: torch.Tensor = torch.linalg.solve(K_mat, a_vec)
        alpha = _alpha.view(len(nonzero_grad_idx), self.X.shape[1]) # corresponds to `alpha` in the paper
        rho_mat = torch.exp(- torch.cdist(self.X, nonzero_grad_X) ** 2 / (2 * self.sigma ** 2)) # shape = (num_pts, num_nonzero_grad_pts)
        new_grad = torch.einsum("ij,jk->ik", rho_mat, alpha).detach().clone() # shape = (num_pts, pts_dim)
        ### obtain auxiliary loss based on the new gradient ###
        self.optimizer.zero_grad()
        target_X = self.X.detach() -  new_grad
        aux_loss = (torch.norm(target_X - self.X) ** 2) * gen_grad_norm / (2 * torch.norm(new_grad))
        ### add the regularization, compute the gradient, and update the parameters ###
        aux_loss = aux_loss + self.reg_obj(self.X)
        aux_loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        ### project to the region of regularization = 0 ###
        if self.reg_proj:
            self.X = self.reg_obj.projection(self.X)

class Continuation(PHOptimization):
    def __init__(self, X: torch.Tensor, loss_obj: PersistenceBasedLoss, reg_obj: Optional[Regularization]=None, 
                 lr: float = 1e-1, in_iter_num: int=1):
        super().__init__(X, loss_obj, reg_obj, False)
        self.lr = lr
        self.in_iter_num = in_iter_num
    
    @staticmethod
    def move_forward(X: torch.Tensor, bar_list: list[Bar], direction: torch.Tensor, scale: float=1.0) -> torch.Tensor:
        """
        staticmethod to move PD to the desirable direction.
        Note that bar_list can be obtained through RipsPH.get_bar_object_list.

        Parameters:
            - X(torch.Tensor): shape=(num_pts, 2)
            - bar_list(list[Bar]): list of bars to move. You do not need to specify the dimension of each bar.
            - direction_list(torch.Tensor): specify the direction to move of each bar. shape=(num_pts, 2)
            - scale(float, default=1.0): the scale of the movement.
        """
        if not X.requires_grad:
            raise ValueError("X.requires_grad must be True")
        if not bar_list: # nothing to do
            return X
        ### compute \phi(x, y) = Pers(\Phi(x)) - y and its jacobian matrix ###
        phi: list[torch.Tensor] = []
        _jacobi: list[torch.Tensor] = []
        for i in range(len(bar_list)):
            bv1, bv2, dv1, dv2 = bar_list[i].birth_v1, bar_list[i].birth_v2, bar_list[i].death_v1, bar_list[i].death_v2
            b_direction, d_direction = direction[i]
            # gradient of b_time - b_target w.t.t X = gradient of b_time w.t.t X (one element of the jacobian matrix)
            b_time: torch.Tensor = torch.norm(X[bv1] - X[bv2])
            _jacobi.append(torch.autograd.grad(b_time, X)[0])
            # gradient of d_time - d_target w.t.t X = gradient of d_time w.t.t X (one element of the jacobian matrix)
            d_time: torch.Tensor = torch.norm(X[dv1] - X[dv2])
            _jacobi.append(torch.autograd.grad(d_time, X)[0])
            # contain the value of Pers(\Phi(x)) - y in phi
            phi.append(-b_direction); phi.append(-d_direction) # NOTE: `phi` represents the value `current - target`
        phi_tensor: torch.Tensor = torch.stack(phi, dim=0).detach() # (2 * num_pairs)
        jacobi: torch.Tensor = torch.stack(_jacobi, dim=0) # (2 * num_pairs) x num_pts x pts_dim
        # update X using the pseudo-inverse of the jacobian matrix
        jacobi_flatten = jacobi.view(-1, jacobi.shape[1] * jacobi.shape[2]) # (2 * num_pairs) x (num_pts * pts_dim)
        jacobi_pinv = torch.linalg.pinv(jacobi_flatten).transpose(0, 1).view_as(jacobi) # (2 * num_pairs) x num_pts x pts_dim 
        X = X - scale * torch.einsum("ijk,i->jk", jacobi_pinv, phi_tensor) # num_pts x pts_dim
        return X
    
    def update(self) -> None:
        ### get the direction to move for each bar ###
        direction_info = self.loss_obj.get_direction(self.X) # we can do this using giotto-ph
        # define bar_list and direction 
        bar_list: list[Bar] = []
        direction: list[torch.Tensor] = []
        for dim, bar_list_dim, direction_dim in direction_info:
            bar_list += bar_list_dim
            direction += direction_dim
        if not bar_list: # nothing to do
            return
        direction = torch.stack(direction, dim=0)
        ### inner loop ###
        for in_iter in range(self.in_iter_num):
            self.X = self.move_forward(self.X, bar_list, direction, self.lr)
            self.X = self.reg_obj.projection(self.X)
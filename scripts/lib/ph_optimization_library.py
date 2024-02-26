from typing import Union, Optional, Any
from gudhi.wasserstein import wasserstein_distance
import torch
import numpy as np
import sys, os

from regularization import Regularization
from persistence_based_loss import PersistenceBasedLoss
from lib.ph_computation_library import RipsPH, Bar
from lib.scheduler import TransformerLR

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
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        ### 正則化領域に投影 ###
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
        ### 現在の点における，persistence based loss の勾配のノルムを計算 ###
        loss = self.loss_obj(self.X)
        loss.backward()
        gen_grad_norm = torch.norm(self.X.grad)
        self.optimizer.zero_grad()
        ### 各単体の目標値を計算 ###
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
        for dim in range(self.loss_obj.maxdim+2): # すべての dim
            moving_simp = set(high_target_value[dim].keys()) | set(low_target_value[dim].keys())
            for simp in moving_simp:
                diam: torch.Tensor = rph.get_differentiable_diameter(dim, simp) # 微分可能である必要がある
                if (simp in high_target_value[dim]) and (simp in low_target_value[dim]):
                    # diam から遠い方を目標値にする
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
        aux_loss.backward(retain_graph=True)
        aux_grad_norm = torch.norm(self.X.grad)
        self.optimizer.zero_grad()
        aux_loss = aux_loss * gen_grad_norm / aux_grad_norm if aux_grad_norm > 0 else aux_loss
        ### 正則化を加算して勾配を計算し，パラメータを更新 ###
        aux_loss = aux_loss + self.reg_obj(self.X)
        aux_loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        ### 必ず rph を削除 ###
        del rph
        ### 正則化領域に投影 ###
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
        ### \phi(x, y) = Pers(\Phi(x)) - y とそのヤコビ行列を求める．xは現在の点群．y は目標とする PD．###
        phi: list[torch.Tensor] = []
        _jacobi: list[torch.Tensor] = []
        for i in range(len(bar_list)):
            bv1, bv2, dv1, dv2 = bar_list[i].birth_v1, bar_list[i].birth_v2, bar_list[i].death_v1, bar_list[i].death_v2
            b_direction, d_direction = direction[i]
            # b_time - b_target の X に対する微分 = b_time の X に対する微分 = ヤコビ行列の1成分
            b_time: torch.Tensor = torch.norm(X[bv1] - X[bv2])
            _jacobi.append(torch.autograd.grad(b_time, X)[0])
            # d_time - d_target の X に対する微分 = d_time の X に対する微分 = ヤコビ行列の1成分
            d_time: torch.Tensor = torch.norm(X[dv1] - X[dv2])
            _jacobi.append(torch.autograd.grad(d_time, X)[0])
            # \phi(x, y) の2成分を格納
            phi.append(-b_direction); phi.append(-d_direction) # NOTE: `phi` represents the value `current - target`
        phi_tensor: torch.Tensor = torch.stack(phi, dim=0).detach() # (2 * num_pairs)
        jacobi: torch.Tensor = torch.stack(_jacobi, dim=0) # (2 * num_pairs) x num_pts x pts_dim
        # jacobi の擬似逆行列を使って X を更新
        jacobi_flatten = jacobi.view(-1, jacobi.shape[1] * jacobi.shape[2]) # (2 * num_pairs) x (num_pts * pts_dim)
        jacobi_pinv = torch.linalg.pinv(jacobi_flatten).transpose(0, 1).view_as(jacobi) # (2 * num_pairs) x num_pts x pts_dim 
        X = X - scale * torch.einsum("ijk,i->jk", jacobi_pinv, phi_tensor) # num_pts x pts_dim
        return X
    
    def update(self) -> None:
        ### 各単体を動かす向きを計算 ###
        direction_info = self.loss_obj.get_direction(self.X) # 注意：rph は必要ない．giotto-ph を使える．
        # bar_list と direction を作成
        bar_list: list[Bar] = []
        direction: list[torch.Tensor] = []
        for dim, bar_list_dim, direction_dim in direction_info:
            bar_list += bar_list_dim
            direction += direction_dim
        if not bar_list: # nothing to do
            return
        direction = torch.stack(direction, dim=0)
        ### 内部ループ ###
        for in_iter in range(self.in_iter_num):
            self.X = self.move_forward(self.X, bar_list, direction, self.lr)
            self.X = self.reg_obj.projection(self.X)
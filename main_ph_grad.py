from ph_opt import PHTrainerConfig, PHTrainer, RipsPH, singleton_loss_from_bar_to_target
import torch
from functools import partial

def sampled_expand_loss_with_regtangle_regularization(X: torch.Tensor, grad_type, n_trials, n_samples,
                                                      sigma: float, 
                                                      lamb: float, x_min: float, y_min: float, x_max: float, y_max: float):
    loss = torch.tensor(0.)

    # Sampled expand loss
    for _ in range(n_trials):
        # sample random points
        random_idx = torch.randint(0, X.size(0), (n_samples, ))
        random_X = X[random_idx]
        rph = RipsPH(random_X, maxdim=1)

        # if grad_type is "bigstep", call `compute_ph` and `compute_ph_right` beforehand
        if grad_type == "bigstep":
            rph.compute_ph()
            rph.compute_ph_right()

        # get bar with maximum lifetime
        bars = rph.get_bar_object_list(1)
        if not bars:
            continue
        bar = max(rph.get_bar_object_list(1), key=lambda b: b.death_time - b.birth_time)

        # get target and compute loss
        _target = (bar.birth_time + bar.death_time) / 2
        target = torch.tensor([_target, _target], device=X.device, dtype=X.dtype)
        loss -= singleton_loss_from_bar_to_target(random_X, bar, target, grad_type=grad_type, 
                                                  dim=1, rph=rph, 
                                                  sigma=sigma, all_X=X)
    loss /= n_trials

    # Rectangle regularization
    penalty_x = torch.relu(X[:, 0] - x_max) + torch.relu(x_min - X[:, 0])
    penalty_y = torch.relu(X[:, 1] - y_max) + torch.relu(y_min - X[:, 1])
    loss += lamb * (torch.sum(penalty_x ** 2) + torch.sum(penalty_y ** 2))

    return loss

if __name__ == "__main__":
    loss_obj = partial(sampled_expand_loss_with_regtangle_regularization, n_trials=10, n_samples=40, 
                       sigma=0.1, lamb=1., x_min=-2., y_min=-2., x_max=2., y_max=2.)
    
    method_list = ["gd", "continuation", "bigstep", "diffeo"]
    lr_list = [(4**i) * 1e-3 for i in range(6)]
    for method in method_list:
        for lr in lr_list:
            config = PHTrainerConfig(loss_obj=loss_obj, 
                                     exp_name=f"{method}_lr={lr:.3f}", 
                                     method=method, lr=lr, num_epoch=200)
            pht = PHTrainer(config, viz_dims=[1])
            pht.train()
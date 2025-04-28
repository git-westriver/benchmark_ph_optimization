from ph_opt import PHTrainerConfig, PHTrainer, powered_wasserstein_distance_one_sided_with_improved_grad
import torch
from functools import partial

def sampled_expand_loss_with_regtangle_regularization(X: torch.Tensor, grad_type, n_trials, n_samples,
                                                      sigma: float, 
                                                      lamb: float, x_min: float, y_min: float, x_max: float, y_max: float):
    loss = torch.tensor(0.)

    # Sampled expand loss
    for _ in range(n_trials):
        random_idx = torch.randint(0, X.size(0), (n_samples, ))
        random_X = X[random_idx]
        loss -= powered_wasserstein_distance_one_sided_with_improved_grad(random_X, ref_pd=[torch.empty(0, 2)], dims=[1], 
                                                                          grad_type=grad_type, sigma=sigma, all_X=X)
    loss /= n_trials

    # Rectangle regularization
    penalty_x = torch.relu(X[:, 0] - x_max) + torch.relu(x_min - X[:, 0])
    penalty_y = torch.relu(X[:, 1] - y_max) + torch.relu(y_min - X[:, 1])
    loss += lamb * (torch.sum(penalty_x ** 2) + torch.sum(penalty_y ** 2))

    return loss

if __name__ == "__main__":
    loss_obj = partial(sampled_expand_loss_with_regtangle_regularization, n_trials=10, n_samples=40, 
                       sigma=0.1, lamb=1., x_min=-2., y_min=-2., x_max=2., y_max=2.)
    
    method_list = ["diffeo"] # ["gd", "continuation", "bigstep", "diffeo"]
    lr_list = [(4**i) * 1e-3 for i in range(6)]
    for method in method_list:
        for lr in lr_list:
            config = PHTrainerConfig(loss_obj=loss_obj, 
                                     exp_name=f"{method}_lr={lr:.3f}", 
                                     method=method, lr=lr, num_epoch=200)
            pht = PHTrainer(config, viz_dims=[1])
            pht.train()
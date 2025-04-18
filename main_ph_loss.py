from ph_opt import PHTrainerConfig, PHTrainer, ExpandLoss, RectangleRegularization

if __name__ == "__main__":
    loss_obj = ExpandLoss([1], 1, topk=1)
    regularization_obj = RectangleRegularization(-2., -2., 2., 2., 1., 2)
    
    method_list = ["gd", "continuation", "bigstep", "diffeo"]
    lr_list = [(4**i) * 1e-3 for i in range(6)]
    for method in method_list:
        for lr in lr_list:
            config = PHTrainerConfig(loss_obj=loss_obj, 
                                     regularization_obj=regularization_obj,
                                     exp_name=f"{method}_lr={lr:.3f}", 
                                     method=method, lr=lr, num_epoch=100)
            pht = PHTrainer(config)
            pht.train()
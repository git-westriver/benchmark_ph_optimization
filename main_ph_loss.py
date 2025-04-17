from ph_opt import PHTrainerConfig, PHTrainer

if __name__ == "__main__":
    method_list = ["gd", "continuation", "bigstep", "diffeo"]
    lr_list = [(4**i) * 1e-3 for i in range(6)]
    for method in method_list:
        for lr in lr_list:
            config = PHTrainerConfig(exp_name=f"{method}_lr={lr:.3f}", method=method, lr=lr, num_epoch=100)
            pht = PHTrainer(config)
            pht.train()
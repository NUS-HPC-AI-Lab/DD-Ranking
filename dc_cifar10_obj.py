import os
import torch
from dd_ranking.metrics import Hard_Label_Objective_Metrics


root = "/home/wangkai/"
device = "cuda:7"
method_name = "DC"
dataset = "CIFAR10"
data_dir = os.path.join(root, "datasets")
model_name = "ConvNet-3"
im_size = (32, 32)

for ipc in [1, 10, 50]:
    print(f"Evaluating {method_name} on {dataset} with ipc{ipc}")
    syn_images = torch.load(os.path.join(root, f"DD-Ranking/{method_name}/{dataset}/IPC{ipc}/images.pt"), map_location='cpu')

    save_path_hard = f"./results/{dataset}/{model_name}/IPC{ipc}/dc_hard_scores.csv"
    convd3_hard_obj = Hard_Label_Objective_Metrics(
        dataset=dataset, 
        real_data_path=data_dir, 
        ipc=ipc,
        model_name=model_name,
        im_size=im_size,
        data_aug_func=None,
        aug_params=None,
        device=device,
        save_path=save_path_hard
    )
    print(convd3_hard_obj.compute_metrics(syn_images))

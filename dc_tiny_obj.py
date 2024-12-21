import os
import torch
from dd_ranking.metrics import Hard_Label_Objective_Metrics


root = "/home/wangkai/"
device = "cuda:6"
method_name = "DC"
dataset = "TinyImageNet"
im_size = (64, 64)
data_dir = os.path.join(root, "datasets/tiny-imagenet-200")
model_name = "ConvNet-4"

for ipc in [1, 10]:
    print(f"Evaluating {method_name} on {dataset} with ipc{ipc}")
    syn_images = torch.load(os.path.join(root, f"DD-Ranking/{method_name}/{dataset}/IPC{ipc}/images.pt"), map_location='cpu')

    save_path_hard = f"./results/{dataset}/{model_name}/IPC{ipc}/dc_hard_scores.csv"
    convd4_hard_obj = Hard_Label_Objective_Metrics(
        dataset=dataset, 
        real_data_path=data_dir,
        ipc=ipc, 
        model_name=model_name,
        data_aug_func=None,
        aug_params=None,
        im_size=im_size,
        device=device,
        save_path=save_path_hard
    )
    print(convd4_hard_obj.compute_metrics(syn_images))

import os
import torch
from dd_ranking.metrics import Hard_Label_Objective_Metrics


root = "/home/wangkai/"
device = "cuda:7"
method_name = "DM"
dataset = "TinyImageNet"
im_size = (64, 64)
data_dir = os.path.join(root, "datasets/tiny-imagenet-200")
model_name = "ConvNet-4"
dsa_params = {
    "prob_flip": 0.5,
    "ratio_rotate": 15.0,
    "saturation": 2.0,
    "brightness": 1.0,
    "contrast": 0.5,
    "ratio_scale": 1.2,
    "ratio_crop_pad": 0.125,
    "ratio_cutout": 0.5,
}

for ipc in [1, 10]:
    print(f"Evaluating {method_name} on {dataset} with ipc{ipc}")
    syn_images = torch.load(os.path.join(root, f"DD-Ranking/{method_name}/{dataset}/IPC{ipc}/images.pt"), map_location='cpu')

    save_path_hard = f"./results/{dataset}/{model_name}/IPC{ipc}/dm_hard_scores.csv"
    convd3_hard_obj = Hard_Label_Objective_Metrics(
        dataset=dataset, 
        real_data_path=data_dir, 
        ipc=ipc, 
        model_name=model_name,
        data_aug_func='dsa',
        aug_params=dsa_params,
        im_size=im_size,
        device=device,
        save_path=save_path_hard
    )
    print(convd3_hard_obj.compute_metrics(syn_images))
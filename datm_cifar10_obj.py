import os
import torch
from dd_ranking.metrics import Soft_Label_Objective_Metrics


root = "/home/wangkai/"
device = "cuda:0"
method_name = "DATM"
dataset = "CIFAR10"
im_size = (32, 32)
data_dir = os.path.join(root, "datasets")
model_name = "ConvNet-3"
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

for ipc in [10, 50]:
    print(f"Evaluating {method_name} on {dataset} with ipc{ipc}")
    syn_images = torch.load(os.path.join(root, f"DD-Ranking/baselines/{method_name}/{dataset}/IPC{ipc}/images.pt"), map_location='cpu')
    soft_labels = torch.load(os.path.join(root, f"DD-Ranking/baselines/{method_name}/{dataset}/IPC{ipc}/labels.pt"), map_location='cpu')
    syn_lr = torch.load(os.path.join(root, f"DD-Ranking/baselines/{method_name}/{dataset}/IPC{ipc}/lr.pt"), map_location='cpu')

    save_path_soft = f"./results/{dataset}/{model_name}/IPC{ipc}/datm_soft_scores.csv"
    convd3_soft_obj = Soft_Label_Objective_Metrics(
        dataset=dataset, 
        real_data_path=data_dir, 
        ipc=ipc,
        soft_label_criterion='sce',
        soft_label_mode='S',
        optimizer='SGD',
        lr_scheduler='StepLR',
        weight_decay=0.0005,
        momentum=0.9,
        model_name=model_name,
        data_aug_func='dsa',
        aug_params=dsa_params,
        use_zca=True,
        im_size=im_size,
        device=device,
        save_path=save_path_soft
    )
    print(convd3_soft_obj.compute_metrics(syn_images, soft_labels, syn_lr=syn_lr))
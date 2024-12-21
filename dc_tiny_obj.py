import os
import torch
from dd_ranking.metrics import SCE_Objective_Metrics, KL_Objective_Metrics


root = "/home/wangkai/"
device = "cuda:1"
method_name = "DC"
dataset = "TinyImageNet"
im_size = (64, 64)
data_dir = os.path.join(root, "datasets/tiny-imagenet-200")
model_name = "ConvNet-4"

for ipc in [1, 10, 50]:
    print(f"Evaluating {method_name} on {dataset} with ipc{ipc}")
    syn_images = torch.load(os.path.join(root, f"DD-Ranking/{method_name}/{dataset}/IPC{ipc}/images.pt"), map_location='cpu')

    save_path_sce = f"./results/{dataset}/{model_name}/IPC{ipc}/dc_sce_scores.csv"
    convd3_sce_obj = SCE_Objective_Metrics(dataset=dataset, 
                                          im_size=im_size,
                                          real_data_path=data_dir, 
                                          ipc=ipc, 
                                          model_name=model_name, 
                                          device=device, 
                                          save_path=save_path_sce)
    print(convd3_sce_obj.compute_metrics(syn_images))

    save_path_kl = f"./results/{dataset}/{model_name}/IPC{ipc}/dc_kl_scores.csv"
    convd3_kl_obj = KL_Objective_Metrics(dataset=dataset, 
                                         im_size=im_size, 
                                         real_data_path=data_dir, 
                                         ipc=ipc, 
                                         model_name=model_name, 
                                         device=device, 
                                         save_path=save_path_kl)
    print(convd3_kl_obj.compute_metrics(syn_images))

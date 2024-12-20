import os
import torch
from dd_ranking.metrics import SCE_Objective_Metrics, KL_Objective_Metrics


root = "/home/wangkai/"
device = "cuda:4"
method_name = "datm"
dataset = "CIFAR10"
data_dir = os.path.join(root, "datasets")
model_name = "ConvNet-3"
ipc = 10

syn_images = torch.load(os.path.join(root, f"DD-Ranking/{method_name}/{dataset}/ipc{ipc}/images.pt"), map_location='cpu')
soft_labels = torch.load(os.path.join(root, f"DD-Ranking/{method_name}/{dataset}/ipc{ipc}/labels.pt"), map_location='cpu')
syn_lr = torch.load(os.path.join(root, f"DD-Ranking/{method_name}/{dataset}/ipc{ipc}/lr.pt"), map_location='cpu')

convd3_sl_obj = SCE_Objective_Metrics(dataset=dataset, real_data_path=data_dir, ipc=ipc, model_name=model_name, device=device)
print(convd3_sl_obj.compute_metrics(syn_images, soft_labels=soft_labels))

convd3_kl_obj = KL_Objective_Metrics(dataset=dataset, real_data_path=data_dir, ipc=ipc, model_name=model_name, device=device)
print(convd3_kl_obj.compute_metrics(syn_images, soft_labels=soft_labels))
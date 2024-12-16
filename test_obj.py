import os
import torch
from dd_ranking import DD_Ranking_Objective, Soft_Label_Objective, KL_Divergence_Objective

syn_images = torch.load("./datm/images_best.pt", map_location='cpu')
syn_labels = torch.load("./datm/labels_best.pt", map_location='cpu')
syn_lr = torch.load("./datm/lr_best.pt", map_location='cpu')

device = "cuda:1"
convd3_sl_obj = Soft_Label_Objective(dataset="CIFAR10", real_data_path="../DATASET", ipc=10, model_name="ConvNet-3", device=device)
print(convd3_sl_obj.compute_metrics(syn_images, syn_labels, syn_lr))

convd3_kl_obj = KL_Divergence_Objective(dataset="CIFAR10", real_data_path="../DATASET", ipc=10, model_name="ConvNet-3", device=device)
print(convd3_kl_obj.compute_metrics(syn_images, syn_labels, syn_lr))

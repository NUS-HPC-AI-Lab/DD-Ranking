import os
import torch
from dd_ranking import DD_Ranking_Objective, Soft_Label_Objective, KL_Divergence_Objective

# syn_images = torch.load("./datm/cifar10/ipc10/images_best.pt", map_location='cpu')
# syn_labels = torch.load("./datm/cifar10/ipc10/labels_best.pt", map_location='cpu')
# syn_lr = torch.load("./datm/cifar10/ipc10/lr_best.pt", map_location='cpu')
root = "/home/wangkai/"
device = "cuda:7"
method_name = "datm"
dataset = "CIFAR10"
data_dir = os.path.join(root, "datasets")
model_name = "ConvNet-3"
ipc = 10

syn_images = torch.load(os.path.join(root, f"DD-Ranking/{method_name}/{dataset}/ipc{ipc}/images.pt"), map_location='cpu')
soft_labels = torch.load(os.path.join(root, f"DD-Ranking/{method_name}/{dataset}/ipc{ipc}/labels.pt"), map_location='cpu')
syn_lr = torch.load(os.path.join(root, f"DD-Ranking/{method_name}/{dataset}/ipc{ipc}/lr.pt"), map_location='cpu')

# convd3_sl_obj = Soft_Label_Objective(dataset=dataset, real_data_path=data_dir, ipc=ipc, model_name=model_name, device=device)
# print(convd3_sl_obj.compute_metrics(syn_images, soft_labels=soft_labels, syn_lr=syn_lr))

convd3_kl_obj = KL_Divergence_Objective(dataset=dataset, real_data_path=data_dir, ipc=ipc, stu_model_name=model_name, tea_model_name=model_name, device=device)
print(convd3_kl_obj.compute_metrics(syn_images))

import os
import torch
from dd_ranking import DD_Ranking_Objective, Soft_Label_Objective, KL_Divergence_Objective

# syn_images = torch.load("./datm/images_best.pt", map_location='cpu')
# syn_labels = torch.load("./datm/labels_best.pt", map_location='cpu')
# syn_lr = torch.load("./datm/lr_best.pt", map_location='cpu')
root = "/home/wangkai/"
syn_data = torch.load(os.path.join(root, "DD-Ranking/dc/images.pt"), map_location='cpu')['data']
syn_images = syn_data[0][0]
hard_labels = syn_data[0][1]
print(syn_images.shape)
print(hard_labels.shape)

device = "cuda:7"
dataset = "CIFAR10"
data_dir = os.path.join(root, "datasets")
model_name = "ConvNet-3"
lr = 0.01
ipc = 10

convd3_sl_obj = Soft_Label_Objective(dataset=dataset, real_data_path=data_dir, ipc=ipc, model_name=model_name, lr=lr, device=device)
print(convd3_sl_obj.compute_metrics(syn_images, hard_labels=hard_labels))

convd3_kl_obj = KL_Divergence_Objective(dataset=dataset, real_data_path=data_dir, ipc=ipc, model_name=model_name, lr=lr, device=device)
print(convd3_kl_obj.compute_metrics(syn_images, hard_labels=hard_labels))

import os
import torch
from dd_ranking.metrics import DSA_Augmentation_Metrics, ZCA_Whitening_Augmentation_Metrics, Mixup_Augmentation_Metrics, Cutmix_Augmentation_Metrics

# syn_images = torch.load("./datm/cifar10/ipc10/images_best.pt", map_location='cpu')
# syn_labels = torch.load("./datm/cifar10/ipc10/labels_best.pt", map_location='cpu')
# syn_lr = torch.load("./datm/cifar10/ipc10/lr_best.pt", map_location='cpu')
root = "/home/wangkai/"
device = "cuda:0"
method_name = "RDED"
dataset = "ImageNette"
data_dir = os.path.join(root, "datasets/imagenet")
model_name = "ConvNet-5-BN"
ipc = 10


syn_images = torch.load(os.path.join(root, f"DD-Ranking/{method_name}/{dataset}/IPC{ipc}/images.pt"), map_location='cpu')
# soft_labels = torch.load(os.path.join(root, f"DD-Ranking/{method_name}/{dataset}/IPC{ipc}/labels.pt"), map_location='cpu')
# syn_lr = torch.load(os.path.join(root, f"DD-Ranking/{method_name}/{dataset}/IPC{ipc}/lr.pt"), map_location='cpu')

# func_names = ['rotate', 'saturation', 'scale', 'crop', 'cutout']
# dsa_params = {
#     "prob_flip": 0.5,
#     "ratio_rotate": 15.0,
#     "saturation": 2.0,
#     "brightness": 1.0,
#     "contrast": 0.5,
#     "ratio_scale": 1.2,
#     "ratio_crop_pad": 0.125,
#     "ratio_cutout": 0.5,
# }

# convd3_dsa_obj = DSA_Augmentation_Metrics(dataset=dataset, real_data_path=data_dir, ipc=ipc, model_name=model_name, device=device, 
#                                  func_names=func_names, params=dsa_params, aug_mode="S")
# print(convd3_dsa_obj.compute_metrics(syn_images))

convd5bn_cutmix_obj = Cutmix_Augmentation_Metrics(dataset=dataset, real_data_path=data_dir, ipc=ipc, model_name=model_name, im_size=(128, 128), device=device)
print(convd5bn_cutmix_obj.compute_metrics(syn_images))

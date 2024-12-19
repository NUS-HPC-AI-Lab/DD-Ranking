import os
import torch
from dd_ranking import Augmentation, DSA_Augmentation, ZCA_Whitening_Augmentation, Mixup_Augmentation, Cutmix_Augmentation

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

func_names = ['rotate', 'saturation', 'scale', 'crop', 'cutout']
params = {
    "prob_flip": 0.5,
    "ratio_rotate": 15.0,
    "saturation": 2.0,
    "brightness": 1.0,
    "contrast": 0.5,
    "ratio_scale": 1.2,
    "ratio_crop_pad": 0.125,
    "ratio_cutout": 0.5,
}
convd3_sl_obj = DSA_Augmentation(dataset=dataset, real_data_path=data_dir, ipc=ipc, model_name=model_name, device=device, 
                                 func_names=func_names, params=params, aug_mode="S")
print(convd3_sl_obj.compute_metrics(syn_images, syn_lr=syn_lr))


import os
import torch
from dd_ranking.metrics import Soft_Label_Objective_Metrics
from dd_ranking.config import Config


"""Use config file to specify the parameters (Recommended)"""
config = Config.from_file("./configs/Demo_Soft_Label.yaml")
convd3_soft_obj = Soft_Label_Objective_Metrics(config)
syn_images = torch.load(os.path.join(syn_data_dir, f"images.pt"), map_location='cpu')
soft_labels = torch.load(os.path.join(syn_data_dir, f"labels.pt"), map_location='cpu')
print(convd3_soft_obj.compute_metrics(syn_images, soft_labels, syn_lr=0.01))


"""Use hardcoded parameters"""
device = "cuda"
method_name = "DATM"                    # Specify your method name
ipc = 10                                # Specify your IPC
dataset = "CIFAR10"                     # Specify your dataset name
syn_data_dir = "./DATM/CIFAR10/IPC10/"  # Specify your synthetic data path
data_dir = "./datasets"                 # Specify your dataset path
model_name = "ConvNet-3"                # Specify your model name
im_size = (32, 32)                      # Specify your image size
dsa_params = {                          # Specify your data augmentation parameters
    "prob_flip": 0.5,
    "ratio_rotate": 15.0,
    "saturation": 2.0,
    "brightness": 1.0,
    "contrast": 0.5,
    "ratio_scale": 1.2,
    "ratio_crop_pad": 0.125,
    "ratio_cutout": 0.5
}

syn_images = torch.load(os.path.join(syn_data_dir, f"images.pt"), map_location='cpu')
soft_labels = torch.load(os.path.join(syn_data_dir, f"labels.pt"), map_location='cpu')
save_path = f"./results/{dataset}/{model_name}/IPC{ipc}/dm_hard_scores.csv"
convd3_hard_obj = Soft_Label_Objective_Metrics(
    dataset=dataset,
    real_data_path=data_dir, 
    ipc=ipc, 
    model_name=model_name,
    soft_label_criterion='sce',  # Use Soft Cross Entropy Loss
    soft_label_mode='S',         # Use one-to-one image to soft label mapping
    default_lr=0.01,
    optimizer='sgd',             # Use SGD optimizer
    lr_scheduler='step',         # Use StepLR learning rate scheduler
    weight_decay=0.0005,         
    momentum=0.9,                
    use_zca=True,                # Use ZCA whitening (please disable it if you didn't use it to distill synthetic data)
    num_eval=5,                  
    data_aug_func='dsa',         # Use DSA data augmentation
    aug_params=dsa_params,       # Specify dsa parameters
    im_size=im_size,
    num_epochs=1000,
    num_workers=4,
    device=device,
    save_path=save_path
)
print(convd3_hard_obj.compute_metrics(syn_images, soft_labels, syn_lr=0.01))
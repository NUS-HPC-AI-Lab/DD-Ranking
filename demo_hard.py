import os
import torch
import warnings
from ddranking.metrics import LabelRobustScoreHard
from ddranking.config import Config
warnings.filterwarnings("ignore")


""" Use config file to specify the arguments (Recommended) """
config = Config.from_file("./configs/Demo_LRS_Hard_Label.yaml")
hard_label_evaluator = LabelRobustScoreHard(config)

syn_data_dir = "./baselines/DM/CIFAR10/IPC10/"
syn_images = torch.load(os.path.join(syn_data_dir, f"images.pt"), map_location='cpu')
syn_lr = 0.01
print(hard_label_evaluator.compute_metrics(image_tensor=syn_images, syn_lr=syn_lr))


""" Use keyword arguments """
device = "cuda"
method_name = "DM"                    # Specify your method name
ipc = 10                              # Specify your IPC
dataset = "CIFAR10"                   # Specify your dataset name
data_dir = "./datasets"               # Specify your dataset path
syn_data_dir = "./DM/CIFAR10/IPC10/"  # Specify your synthetic data path
model_name = "ConvNet-3"              # Specify your model name
im_size = (32, 32)                    # Specify your image size

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

syn_images = torch.load(os.path.join(syn_data_dir, f"images.pt"), map_location='cpu')
save_path = f"./results/{dataset}/{model_name}/IPC{ipc}/dm_hard_scores.csv"
hard_label_evaluator = LabelRobustScoreHard(
    dataset=dataset, 
    real_data_path=data_dir, 
    ipc=ipc, 
    model_name=model_name,
    optimizer='sgd',             # Use SGD optimizer
    lr_scheduler='step',         # Use StepLR learning rate scheduler
    weight_decay=0.0005,
    momentum=0.9,               
    use_zca=False,              
    num_eval=5,                 
    data_aug_func='dsa',         # Use DSA data augmentation
    aug_params=dsa_params,       # Specify DSA parameters
    im_size=im_size,
    num_epochs=1000,
    num_workers=4,
    use_torchvision=False,
    syn_batch_size=256,
    real_batch_size=256,
    custom_train_trans=None,
    custom_val_trans=None,
    device=device,
    dist=True,
    save_path=save_path,
    random_data_format='tensor',
    random_data_path='./random_data',
    eval_full_data=True,
)
print(hard_label_evaluator.compute_metrics(image_tensor=syn_images, syn_lr=0.01))

import os
import torch
import warnings
from ddranking.metrics import LabelRobustScoreSoft
from ddranking.config import Config
warnings.filterwarnings("ignore")


""" Use config file to specify the arguments (Recommended) """
config = Config.from_file("./configs/Demo_LRS_Soft_Label.yaml")
soft_label_evaluator = LabelRobustScoreSoft(config)

syn_data_dir = "./baselines/DATM/CIFAR10/IPC10/"
syn_images = torch.load(os.path.join(syn_data_dir, f"images.pt"), map_location='cpu')
soft_labels = torch.load(os.path.join(syn_data_dir, f"labels.pt"), map_location='cpu')
syn_lr = torch.load(os.path.join(syn_data_dir, f"lr.pt"), map_location='cpu')
print(soft_label_evaluator.compute_metrics(image_tensor=syn_images, soft_labels=soft_labels, syn_lr=syn_lr))


""" Use keyword arguments """
device = "cuda"
method_name = "DATM"                    # Specify your method name
ipc = 10                                # Specify your IPC
dataset = "CIFAR10"                     # Specify your dataset name
syn_data_dir = "./DATM/CIFAR10/IPC10/"  # Specify your synthetic data path
data_dir = "./datasets"                 # Specify your dataset path
model_name = "ConvNet-3"                # Specify your model name
im_size = (32, 32)                      # Specify your image size
dsa_params = {                          # Specify your data augmentation parameters
    "flip": 0.5,
    "rotate": 15.0,
    "saturation": 2.0,
    "brightness": 1.0,
    "contrast": 0.5,
    "scale": 1.2,
    "crop": 0.125,
    "cutout": 0.5
}

syn_images = torch.load(os.path.join(syn_data_dir, f"images.pt"), map_location='cpu')
soft_labels = torch.load(os.path.join(syn_data_dir, f"labels.pt"), map_location='cpu')
syn_lr = torch.load(os.path.join(syn_data_dir, f"lr.pt"), map_location='cpu')
save_path = f"./results/{dataset}/{model_name}/IPC{ipc}/dm_hard_scores.csv"
soft_label_evaluator = LabelRobustScoreSoft(
    dataset=dataset,
    real_data_path=data_dir, 
    ipc=ipc, 
    model_name=model_name,
    soft_label_criterion='sce',  # Use Soft Cross Entropy Loss
    soft_label_mode='S',         # Use one-to-one image to soft label mapping
    loss_fn_kwargs={'temperature': 1.0, 'scale_loss': False},
    optimizer='sgd',             # Use SGD optimizer
    lr_scheduler='step',         # Use StepLR learning rate scheduler
    step_size=500,
    weight_decay=0.0005,         
    momentum=0.9,                
    use_zca=True,                # Use ZCA whitening (please disable it if you didn't use it to distill synthetic data)
    num_eval=5,                  
    data_aug_func='dsa',         # Use DSA data augmentation
    aug_params=dsa_params,       # Specify dsa parameters
    im_size=im_size,
    num_epochs=1000,
    num_workers=4,
    eval_full_data=True,
    stu_use_torchvision=False,
    tea_use_torchvision=False,
    random_data_format='tensor',
    random_data_path='./random_data',
    custom_train_trans=None,
    custom_val_trans=None,
    syn_batch_size=256,
    real_batch_size=256,
    teacher_dir='./teacher_models',
    teacher_model_name=['ConvNet-3'],
    device=device,
    dist=True,
    save_path=save_path
)
print(soft_label_evaluator.compute_metrics(image_tensor=syn_images, soft_labels=soft_labels, syn_lr=syn_lr))

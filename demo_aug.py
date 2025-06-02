import os
import argparse
import torch
from ddranking.metrics import AugmentationRobustScore
from ddranking.config import Config

""" Use config file to specify the arguments (Recommended) """
config = Config.from_file("./configs/Demo_ARS.yaml")
aug_evaluator = AugmentationRobustScore(config)

syn_data_dir = "./baselines/SRe2L/ImageNet1K/IPC10/"
print(aug_evaluator.compute_metrics(image_path=syn_data_dir, syn_lr=0.001))


""" Use keyword arguments """
from torchvision import transforms
device = "cuda"
method_name = "SRe2L"                    # Specify your method name
ipc = 10                                # Specify your IPC
dataset = "ImageNet1K"                     # Specify your dataset name
syn_data_dir = "./SRe2L/ImageNet1K/IPC10/"  # Specify your synthetic data path
data_dir = "./datasets"                 # Specify your dataset path
model_name = "ResNet-18-BN"                # Specify your model name
im_size = (224, 224)                      # Specify your image size
cutmix_params = {                          # Specify your data augmentation parameters
    "beta": 1.0
}

syn_images = torch.load(os.path.join(syn_data_dir, f"images.pt"), map_location='cpu')
soft_labels = torch.load(os.path.join(syn_data_dir, f"labels.pt"), map_location='cpu')
syn_lr = torch.load(os.path.join(syn_data_dir, f"lr.pt"), map_location='cpu')
save_path = f"./results/{dataset}/{model_name}/IPC{ipc}/dm_hard_scores.csv"

custom_train_trans = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
custom_val_trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

aug_evaluator = AugmentationRobustScore(
    dataset=dataset,
    real_data_path=data_dir, 
    ipc=ipc, 
    model_name=model_name,
    label_type='soft',
    soft_label_criterion='kl',  # Use Soft Cross Entropy Loss
    soft_label_mode='M',         # Use one-to-one image to soft label mapping
    loss_fn_kwargs={'temperature': 1.0, 'scale_loss': False},
    optimizer='adamw',             # Use SGD optimizer
    lr_scheduler='cosine',         # Use StepLR learning rate scheduler
    weight_decay=0.01,         
    momentum=0.9,                
    num_eval=5,                  
    data_aug_func='cutmix',         # Use DSA data augmentation
    aug_params=cutmix_params,       # Specify dsa parameters
    im_size=im_size,
    num_epochs=300,
    num_workers=4,
    stu_use_torchvision=True,
    tea_use_torchvision=True,
    random_data_format='tensor',
    random_data_path='./random_data',
    custom_train_trans=custom_train_trans,
    custom_val_trans=custom_val_trans,
    batch_size=256,
    teacher_dir='./teacher_models',
    teacher_model_name=['ResNet-18-BN'],
    device=device,
    dist=True,
    save_path=save_path
)
print(aug_evaluator.compute_metrics(image_path=syn_data_dir, syn_lr=0.001))

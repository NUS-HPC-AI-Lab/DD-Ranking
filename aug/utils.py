import torch
import numpy as np
import random
from torchvision import transforms


def set_seed(seed):
    """
    setting random seed
    """
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # in case of distributed setting 

    np.random.seed(seed)

    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def Default_Augmentation(images,transform_params):
    # you can also add your own implementation here
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=transform_params["flip"]["prob"]),
        transforms.RandomAffine(
            degrees=transform_params["affine"]["degrees"],
            shear=transform_params["affine"]["shear"],
            interpolation=transform_params["affine"]["interpolation"]),
        transforms.RandomGrayscale(p=transform_params["gray"]["prob"]),
        transforms.ColorJitter(
            brightness=transform_params["colorjitter"]["brightness"],
            contrast=transform_params["colorjitter"]["contrast"],
            saturation=transform_params["colorjitter"]["saturation"],
            hue=transform_params["colorjitter"]["hue"])
        ])
    p = transform_params["transform"]["prob"]
    for i,img in enumerate(images):
        for t in transform.transforms:
            if p > torch.rand(1):
                images[i]=t(img)
    return images

def calculate_acc(model,images,labels,transform):
    pass
    



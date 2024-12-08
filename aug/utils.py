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
    

def ZCA_Whitening(x, epsilon=1e-6):
    
    # mean centering
    mean = torch.mean(x, dim=[0, 2, 3], keepdim=True)  # 
    x = x - mean
    
    
    B, C, H, W = x.shape
    x_flattened = x.view(B, -1)
    
    cov_matrix = torch.cov(x_flattened.T)  # covariance (C*H*W, C*H*W)
    
    # Step 4: eigen values and eigen vectors
    eig_vals, eig_vecs = torch.linalg.eigh(cov_matrix)  
    
    # Step 5: whitening matrix
    eig_vals_inv_sqrt = torch.diag(1.0 / torch.sqrt(eig_vals + epsilon))  
    whitening_matrix = eig_vecs @ eig_vals_inv_sqrt @ eig_vecs.T  
    
    # Step 6: transform the data
    x_white = x_flattened @ whitening_matrix.T 
    x_white = x_white.view(B, C, H, W)  
    
    return x_white


def CutMix(x,y):
    pass


def Mixup(x,y):
    pass


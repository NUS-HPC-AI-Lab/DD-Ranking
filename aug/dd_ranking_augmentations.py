import os
import torch
from typing import Dict
from torch import Tensor
from torchvision import transforms
from utils import set_seed, Default_Augmentation, calculate_acc
import random
import utils

class Augmentation:
    def __init__(self, images:Tensor=None, transform_params:Dict=None, model:torch.nn.Module=None):
        self.images=images
        self.transform_params=transform_params
        self.model=model
        self.transform=None
        
        
    def __call__(self):
        p = self.transform_params["transform"]["prob"]
        for i,img in enumerate(self.images):
            img = img.view(1,-1,-1,-1)
            for t in self.transform.transforms:
                if p > torch.rand(1):
                    self.images[i]=t(img).view(-1,-1,-1)
        return self.images
    
    def compute_metric(self,labels,distilled_dataset):
        
        set_seed(self.transform_params["seed"])
        
        q_d = calculate_acc(self.model, distilled_dataset, labels, Default_Augmentation)
        q_e = calculate_acc(self.model, distilled_dataset, labels, self.transform)
        p_f = calculate_acc(self.model, self.images, labels, Default_Augmentation)
        
        select = random.sample(range(self.images.shape[0]), self.transform_params["subset_size"])
        subset = self.images[select]
        r_e = calculate_acc(self.model, subset, labels, self.transform)
        
        return abs((p_f-r_e)/(q_d-q_e))
        
        
    
class DSA_Augmentation(Augmentation):
    def __init__(self, images = None, transform_type = None, transform_params = None):
        super().__init__(images, transform_type, transform_params)
        self.transform = transforms.Compose(
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
        hue=transform_params["colorjitter"]["hue"])]
        )
    def __call__(self):
        return self.transform(images)
        
    
class ZCA_Whitening_Augmentation(Augmentation):
    def __init__(self, images = None, transform_params = None, model = None):
        super().__init__(images, transform_params, model)
        self.transform=transforms.Compose([utils.ZCA_Whitening])
        
    def __call__(self):
        return self.transform(self.images)
        
        
class Mixup_Augmentation(Augmentation):
    def __init__(self, labels, images = None, transform_params = None, model = None):
        super().__init__(images, transform_params, model)
        self.labels=labels
        self.transform = utils.Mixup
        
    def __call__(self):
        return self.transform(self.images, self.labels)


class Cutmix_Augmentation(Augmentation):
    def __init__(self, labels, images = None, transform_params = None, model = None):
        super().__init__(images, transform_params, model)
        self.labels=labels
        self.transform = utils.CutMix
        
    def __call__(self):
        return self.transform(self.images, self.labels)
    
    
class Erasing_Augmentation(Augmentation):
    def __init__(self, images = None, transform_params = None, model = None):
        super().__init__(images, transform_params, model)
        self.transformt = transforms.RandomErasing(
        p=self.transform_params["transform"]["prob"],     
        scale=self.transform_params["erasing"]["scale"],   
        ratio=self.transform_params["erasing"]["ratio"], 
        value=self.transform_params["erasing"]["value"],      
        inplace=False)
        
        
        
class Guassian_Blur_Augmentation(Augmentation):
    def __init__(self, images = None, transform_params = None, model = None):
        super().__init__(images, transform_params, model)
        self.transform = transforms.Compose([transforms.GaussianBlur
        (kernel_size=self.transform_params["gb"]["kernel_size"],sigma=self.transform_params["gb"]["sigma"])])
        
        
class Crop_Augmentation(Augmentation):
    def __init__(self, images = None, transform_params = None, model = None):
        super().__init__(images, transform_params, model)
        self.transform = transforms.Compose([transforms.RandomCrop(self.transform_params["crop_size"])])
        
        
if __name__ == "__main__":
    images = torch.randn(10, 3, 32, 32)
    torch.nn.init.constant_(images, 1.0)
    transform_params={
        "flip":{"prob":0.5},
        "affine":{"degrees":15,
                  "shear":20,
                  "interpolation":transforms.InterpolationMode.BILINEAR},
        "gray":{"prob":0.5},
        "colorjitter":{"brightness":0.03,
                       "contrast":0.01,
                       "saturation":0.03,
                       "hue":0.0},
        "transform":{"prob":0.5}
    }
    images= Default_Augmentation(images=images,transform_params=transform_params)
    
    for img in images:
        print(img==images)
    
    
        
    
    

        